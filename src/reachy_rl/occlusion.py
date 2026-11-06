"""A separate occlusion/history wrapper that preserves the original tracking task."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import gymnasium as gym
import numpy as np

from .env import ReachyTrackingEnv, TaskConfig


@dataclass(frozen=True)
class ReacquisitionConfig:
    history_steps: int = 64
    start_min_seconds: float = 2.0
    start_max_seconds: float = 5.0
    duration_min_seconds: float = 1.0
    duration_max_seconds: float = 2.0
    recovery_timeout_seconds: float = 2.0
    hold_seconds: float = 0.2
    center_threshold: float = 0.15

    def validate(self, task):
        if not isinstance(self.history_steps, int) or not 1 <= self.history_steps <= 256:
            raise ValueError("history_steps must be an integer from 1 to 256")
        values = [value for key, value in vars(self).items() if key != "history_steps"]
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("Occlusion times and center threshold must be positive and finite")
        if self.start_min_seconds > self.start_max_seconds or self.duration_min_seconds > self.duration_max_seconds:
            raise ValueError("Occlusion minimum must not exceed maximum")
        if self.hold_seconds > self.recovery_timeout_seconds:
            raise ValueError("Recovery hold duration must fit within the timeout")
        if self.start_max_seconds + self.duration_max_seconds + self.recovery_timeout_seconds > task.episode_seconds:
            raise ValueError("Occlusion plus the complete recovery window must fit in the episode")
        dt = task.control_dt
        for minimum, maximum in ((self.start_min_seconds, self.start_max_seconds),
                                 (self.duration_min_seconds, self.duration_max_seconds)):
            if math.ceil(minimum / dt - 1e-9) > math.floor(maximum / dt + 1e-9):
                raise ValueError("Time range contains no complete policy timestep")


class ReacquisitionEnv(gym.Wrapper):
    """One scheduled detection outage per episode; optionally stack past observations.

    Target motion continues while hidden. Only detector error/flag are masked;
    commanded and measured pose remain available. No future samples are exposed.
    """
    def __init__(self, task: TaskConfig | None = None, experiment: ReacquisitionConfig | None = None):
        task = task or TaskConfig()
        self.experiment = experiment or ReacquisitionConfig()
        self.experiment.validate(task)
        super().__init__(ReachyTrackingEnv(task))
        self.observation_space = gym.spaces.Box(-2.0, 2.0, (12 * self.experiment.history_steps,), np.float32)
        self.frames = deque(maxlen=self.experiment.history_steps)
        self.schedule_rng = None
        self.sensor_rng = None

    @property
    def occluded(self):
        return self.start_step <= self.unwrapped.steps < self.end_step

    def _sample_steps(self, minimum, maximum):
        dt = self.unwrapped.config.control_dt
        low, high = math.ceil(minimum / dt - 1e-9), math.floor(maximum / dt + 1e-9)
        return int(self.schedule_rng.integers(low, high + 1))

    def _frame(self):
        base = self.unwrapped
        error, in_view = base.tracking_error()
        if base.config.randomize:
            # Draw every time, independent of visibility/actions: shared evaluation seeds
            # produce matching sensor perturbations even when controllers point differently.
            error = error + self.sensor_rng.normal(0, base.config.observation_noise, 2)
            dropped = self.sensor_rng.random() < base.config.dropout_probability
        else:
            dropped = False
        detected = bool(in_view and not self.occluded and not dropped)
        if detected:
            self.last_error = error.copy()
        frame = np.concatenate([self.last_error, [float(detected)], base.command / base.limits,
                                base._measurement() / base.limits, base.last_action])
        return np.clip(frame, -2, 2).astype(np.float32), detected

    def _stack(self, frame):
        self.frames.append(frame.copy())
        return np.concatenate(self.frames).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        _, info = self.env.reset(seed=seed, options=options)
        if seed is not None or self.schedule_rng is None:
            schedule_seed, sensor_seed = np.random.SeedSequence(seed).spawn(2)
            self.schedule_rng = np.random.default_rng(schedule_seed)
            self.sensor_rng = np.random.default_rng(sensor_seed)
        c = self.experiment
        self.start_step = self._sample_steps(c.start_min_seconds, c.start_max_seconds)
        self.end_step = self.start_step + self._sample_steps(c.duration_min_seconds, c.duration_max_seconds)
        self.timeout_steps = math.ceil(c.recovery_timeout_seconds / self.unwrapped.config.control_dt - 1e-9)
        self.hold_steps = math.ceil(c.hold_seconds / self.unwrapped.config.control_dt - 1e-9)
        self.status = "waiting"
        self.stable_since = None
        self.latency = None
        self.centered_before_occlusion = False
        self.last_error = np.zeros(2)
        frame, _ = self._frame()
        self.frames.clear()
        self.frames.extend(frame.copy() for _ in range(c.history_steps))
        return np.concatenate(self.frames).astype(np.float32), {**info, "reacquisition": self.report()}

    def report(self):
        dt = self.unwrapped.config.control_dt
        return {
            "occlusion_start_s": self.start_step * dt,
            "occlusion_end_s": self.end_step * dt,
            "occlusion_duration_s": (self.end_step - self.start_step) * dt,
            "status": self.status,
            "success": self.status == "recovered",
            "latency_s": self.latency,
            "timeout_s": self.timeout_steps * dt,
            "centered_before_occlusion": self.centered_before_occlusion,
        }

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        if terminated:
            frame, detected = np.zeros(12, dtype=np.float32), False
        else:
            frame, detected = self._frame()
        info = dict(info)
        info["geometric_visible"] = info["visible"]
        info["visible"] = float(detected)
        info["occluded"] = float(self.occluded)
        info["centered"] = float(detected and info["tracking_error"] < self.experiment.center_threshold)
        step = self.unwrapped.steps
        if step < self.start_step:
            self.centered_before_occlusion = bool(info["centered"])
        if step >= self.end_step and self.status not in ("recovered", "failed"):
            self.status = "pending"
            if info["centered"]:
                if self.stable_since is None:
                    self.stable_since = step
                if step - self.stable_since >= self.hold_steps:
                    self.status = "recovered"
                    self.latency = (step - self.end_step) * self.unwrapped.config.control_dt
            else:
                self.stable_since = None
            if self.status != "recovered" and step >= self.end_step + self.timeout_steps:
                self.status = "failed"
        if (terminated or truncated) and self.status not in ("recovered", "failed"):
            self.status = "failed"
        info["reacquisition"] = self.report()
        # The original simulator's true target error shapes training, including during
        # occlusion. It is privileged reward information, never an observation feature.
        return self._stack(frame), reward, terminated, truncated, info
