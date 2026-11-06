"""Render a checkpoint replay to MP4 without a browser or physical robot."""
from __future__ import annotations

import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import mujoco
import numpy as np

from .env import ReachyTrackingEnv, TaskConfig, proportional_action


class MP4Writer:
    """Stream RGB frames to system FFmpeg; publish only a successfully encoded file."""
    def __init__(self, output, width, height, fps, overwrite=False):
        self.output = Path(output).expanduser().resolve()
        self.overwrite = overwrite
        if self.output.suffix.lower() != ".mp4":
            raise ValueError("Output must have an .mp4 extension")
        if self.output.exists() and not overwrite:
            raise ValueError(f"Output already exists: {self.output}; choose another name or use --overwrite")
        executable = shutil.which("ffmpeg")
        if executable is None:
            raise ValueError("FFmpeg is required. On the Spark: sudo apt install ffmpeg")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".reachy-record-", suffix=".mp4", dir=self.output.parent)
        os.close(fd)
        self.temporary = Path(name)
        self.errors = tempfile.TemporaryFile()
        self.frames = 0
        self.shape = (height, width, 3)
        try:
            self.process = subprocess.Popen([
                executable, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
                "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(self.temporary),
            ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.errors)
        except BaseException:
            self.errors.close()
            self.temporary.unlink(missing_ok=True)
            raise

    def write(self, frame):
        if frame.shape != self.shape or frame.dtype != np.uint8:
            raise ValueError(f"Expected uint8 RGB frame of shape {self.shape}")
        self.process.stdin.write(frame.tobytes())
        self.frames += 1

    def close(self, publish=True):
        try:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
            code = self.process.wait()
            self.errors.seek(0)
            message = self.errors.read().decode(errors="replace")
            if publish:
                if code != 0 or not self.frames:
                    raise ValueError(f"FFmpeg recording failed: {message or 'no frames encoded'}")
                if self.overwrite:
                    os.replace(self.temporary, self.output)
                else:
                    # Atomic no-clobber publication, even if another process created output.
                    os.link(self.temporary, self.output)
        finally:
            self.errors.close()
            self.temporary.unlink(missing_ok=True)


def record(args):
    # Importing here avoids a cycle with the CLI checkpoint helpers.
    from .cli import checkpoint_path, load_config
    from stable_baselines3 import PPO

    if not math.isfinite(args.seconds) or args.seconds <= 0:
        raise ValueError("seconds must be positive and finite")
    if not 1 <= args.fps <= 120:
        raise ValueError("fps must be between 1 and 120")
    if min(args.width, args.height) < 2 or args.width % 2 or args.height % 2:
        raise ValueError("width and height must be positive even numbers, at least 2")
    if args.distance <= 0 or not math.isfinite(args.distance):
        raise ValueError("distance must be positive and finite")
    if args.controller == "ppo":
        checkpoint = checkpoint_path(args.checkpoint)
        config = load_config(checkpoint)
        policy = PPO.load(str(checkpoint), device="cpu")
    else:
        config, policy = TaskConfig(), None
    env = ReachyTrackingEnv(config)
    renderer = None
    writer = None
    completed = False
    try:
        env.model.vis.global_.offwidth = max(args.width, env.model.vis.global_.offwidth)
        env.model.vis.global_.offheight = max(args.height, env.model.vis.global_.offheight)
        # FFmpeg/output errors should be reported before initializing a graphics context.
        writer = MP4Writer(args.output, args.width, args.height, args.fps, args.overwrite)
        renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = [0.45, 0, 0.2]
        camera.distance = args.distance
        camera.azimuth = args.azimuth
        camera.elevation = args.elevation
        episode = 0
        observation, _ = env.reset(seed=args.seed)
        elapsed = 0.0
        for index in range(math.ceil(args.seconds * args.fps)):
            # Resample policy time to video time, preserving playback speed at any FPS.
            while elapsed + 1e-9 < index / args.fps:
                if policy is not None:
                    action, _ = policy.predict(observation, deterministic=True)
                elif args.controller == "proportional":
                    action = proportional_action(env, observation)
                else:
                    action = np.zeros(3)
                observation, _, terminated, truncated, info = env.step(action)
                elapsed += config.control_dt
                if terminated:
                    raise ValueError(f"Simulation terminated during recording: {info}")
                if truncated:
                    episode += 1
                    observation, _ = env.reset(seed=args.seed + episode)
            renderer.update_scene(env.data, camera=camera)
            scene = renderer.scene
            if scene.ngeom < scene.maxgeom:
                mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                   np.array([0.035, 0, 0]), env.target_position,
                                   np.eye(3).ravel(), np.array([1.0, 0.25, 0.15, 1.0]))
                scene.ngeom += 1
            writer.write(renderer.render())
        completed = True
    except BrokenPipeError as error:
        raise ValueError("FFmpeg stopped accepting frames; verify your FFmpeg build supports libx264") from error
    finally:
        if renderer is not None:
            renderer.close()
        env.close()
        if writer is not None:
            writer.close(publish=completed)
    print(f"Saved {writer.output} ({writer.frames} frames, {writer.frames / args.fps:.2f}s, {args.fps} fps)")
