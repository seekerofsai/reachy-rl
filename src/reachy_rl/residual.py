"""Bounded learned pose-rate corrections on top of proportional tracking."""
from dataclasses import dataclass
import math

import gymnasium as gym
import numpy as np

from .env import TaskConfig, proportional_action
from .occlusion import ReacquisitionConfig, ReacquisitionEnv


@dataclass(frozen=True)
class ResidualConfig:
    scale: float = 0.25

    def validate(self):
        if not math.isfinite(self.scale) or not 0 <= self.scale <= 1:
            raise ValueError("Residual scale must be finite and between 0 and 1")


class ResidualTrackingEnv(gym.Wrapper):
    def __init__(self, task: TaskConfig | None = None,
                 experiment: ReacquisitionConfig | None = None,
                 residual: ResidualConfig | None = None):
        self.residual = residual or ResidualConfig()
        self.residual.validate()
        super().__init__(ReacquisitionEnv(task, experiment))
        self.last_observation = None

    def reset(self, *, seed=None, options=None):
        self.last_observation, info = self.env.reset(seed=seed, options=options)
        return self.last_observation.copy(), info

    def baseline_action(self):
        # Zero learned correction means exactly the proportional controller.
        return np.zeros(3, dtype=np.float32)

    def step(self, action):
        if self.last_observation is None:
            raise RuntimeError("Call reset before step")
        action = np.asarray(action, dtype=float)
        if action.shape != (3,) or not np.isfinite(action).all():
            raise ValueError("Expected three finite residual action values")
        residual = np.clip(action, -1, 1)
        baseline = proportional_action(self.unwrapped, self.last_observation[-12:])
        requested = baseline + self.residual.scale * residual
        applied = np.clip(requested, -1, 1)
        observation, reward, terminated, truncated, info = self.env.step(applied)
        self.last_observation = observation.copy()
        info = dict(info)
        info["residual"] = {
            "baseline": baseline.tolist(), "correction": (self.residual.scale * residual).tolist(),
            "applied": applied.tolist(), "clipped": bool(np.any(np.abs(requested) > 1)),
        }
        info["correction_norm"] = float(np.linalg.norm(applied - baseline))
        info["residual_clipped"] = float(info["residual"]["clipped"])
        # Preserve the original reward, including effort/smoothness of APPLIED motion.
        return observation, reward, terminated, truncated, info
