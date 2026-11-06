"""Viser geometry viewer; no OpenGL or display server required on the Spark."""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import viser


class BrowserViewer:
    def __init__(self, env, port):
        self.env = env
        self.server = viser.ViserServer(host="127.0.0.1", port=port)
        # Include both the robot at the origin and targets roughly one metre ahead.
        self.server.initial_camera.position = (1.3, -1.6, 0.8)
        self.server.initial_camera.look_at = (0.45, 0.0, 0.2)
        self.server.initial_camera.up = (0, 0, 1)
        self.handles = []
        model = env.model
        for i in range(model.ngeom):
            if int(model.geom_type[i]) != int(mujoco.mjtGeom.mjGEOM_MESH):
                continue
            mesh_id = int(model.geom_dataid[i])
            start_v, count_v = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
            start_f, count_f = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
            vertices = model.mesh_vert[start_v:start_v + count_v]
            faces = model.mesh_face[start_f:start_f + count_f]
            material = int(model.geom_matid[i])
            rgba = model.mat_rgba[material] if material >= 0 else model.geom_rgba[i]
            color = tuple((255 * rgba[:3]).astype(int).tolist())
            handle = self.server.scene.add_mesh_simple(f"/robot/geom_{i}", vertices=vertices,
                                                        faces=faces, color=color, opacity=float(rgba[3]))
            self.handles.append((i, handle))
        self.target = self.server.scene.add_icosphere("/moving_target", radius=0.035, color=(255, 80, 60))
        self.server.scene.add_grid("/grid", width=2, height=2)
        self.status = self.server.gui.add_markdown("Synthetic gaze-tracking task. Red sphere = target.")
        print(f"Viewer: http://localhost:{port} (use SSH port forwarding from your Mac)")

    def update(self, info):
        with self.server.atomic():
            for i, handle in self.handles:
                handle.position = self.env.data.geom_xpos[i]
                xyzw = Rotation.from_matrix(self.env.data.geom_xmat[i].reshape(3, 3)).as_quat()
                handle.wxyz = xyzw[[3, 0, 1, 2]]
            self.target.position = self.env.target_position
            self.status.content = (f"**Simulation only**\n\nCamera-plane error: {info['tracking_error']:.3f}"
                                   f"\n\nTarget visible: {bool(info['visible'])}"
                                   f"\n\nCentered: {bool(info['centered'])}")

    def close(self):
        self.server.stop()
