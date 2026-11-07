import json

import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from reachy_rl.advanced import rollout, summarize
from reachy_rl.env import TaskConfig, proportional_action
from reachy_rl.occlusion import ReacquisitionConfig, ReacquisitionEnv
from reachy_rl.photo import PhotoConfig, PhotoFramingEnv
from reachy_rl.residual import ResidualConfig, ResidualTrackingEnv


def test_zero_residual_exactly_matches_proportional_controller():
    experiment = ReacquisitionConfig(history_steps=8)
    direct = ReacquisitionEnv(experiment=experiment)
    residual = ResidualTrackingEnv(experiment=experiment)
    try:
        obs, _ = direct.reset(seed=21)
        memory, _ = residual.reset(seed=21)
        np.testing.assert_array_equal(obs, memory)
        for _ in range(250):
            expected = direct.step(proportional_action(direct.unwrapped, obs[-12:]))
            actual = residual.step(np.zeros(3))
            obs = expected[0]
            np.testing.assert_array_equal(expected[0], actual[0])
            assert expected[1:4] == actual[1:4]
            assert expected[4]["reacquisition"] == actual[4]["reacquisition"]
            assert actual[4]["correction_norm"] == 0
    finally:
        direct.close()
        residual.close()


def test_residual_corrections_are_bounded_and_history_uses_applied_motion():
    env = ResidualTrackingEnv(experiment=ReacquisitionConfig(history_steps=4))
    try:
        env.reset(seed=2)
        for action in ([10, -10, 10], [-10, 10, -10]):
            obs, _, terminated, _, info = env.step(action)
            assert not terminated
            baseline = np.array(info["residual"]["baseline"])
            applied = np.array(info["residual"]["applied"])
            assert np.all(np.abs(applied) <= 1)
            assert np.all(np.abs(applied - baseline) <= 0.25 + 1e-9)
            np.testing.assert_allclose(obs[-3:], applied, atol=1e-7)
        with pytest.raises(ValueError):
            env.step([np.nan, 0, 0])
        with pytest.raises(ValueError):
            ResidualConfig(scale=1.5).validate()
    finally:
        env.close()


def test_photo_gym_contract_and_seeded_reset():
    env = PhotoFramingEnv()
    try:
        check_env(env)
        obs_a, info_a = env.reset(seed=19)
        step_a = env.step([1, 1, 1, 0])
        obs_b, info_b = env.reset(seed=19)
        step_b = env.step([1, 1, 1, 0])
        assert obs_a.shape == (24,) and env.action_space.shape == (4,)
        np.testing.assert_array_equal(obs_a, obs_b)
        np.testing.assert_array_equal(step_a[0], step_b[0])
        assert step_a[1:] == step_b[1:]
        assert info_a == info_b
    finally:
        env.close()


def test_photo_target_pauses_without_teleporting():
    env = PhotoFramingEnv(task=TaskConfig(randomize=False))
    try:
        env.reset(seed=3)
        base = env.unwrapped
        base.pause_start, base.pause_duration = 1.0, 1.0
        positions = []
        for step in (24, 25, 26, 49, 50, 51):
            base.steps = step
            base._update_target()
            positions.append(base.target_position.copy())
        for position in positions[1:5]:
            np.testing.assert_allclose(position, positions[1], atol=1e-12)
        assert np.linalg.norm(positions[0] - positions[1]) < 0.05
        assert np.linalg.norm(positions[-1] - positions[-2]) < 0.05
    finally:
        env.close()


def test_photo_baseline_can_capture_and_never_capture_policy_times_out():
    env = PhotoFramingEnv()
    try:
        baseline = rollout(env, None, 0)
        assert baseline["outcome"]["success"]
        assert baseline["outcome"]["first_capture_valid"]
        env.reset(seed=0)
        for _ in range(250):
            _, _, terminated, truncated, info = env.step([1, 1, 1, 0])
            assert not terminated
            if truncated:
                break
        assert info["photo"]["attempts"] == 0
        assert not info["photo"]["success"]
        assert info["photo"]["capture_time_s"] is None
        assert info["photo"]["failure_reason"] == "timeout"
        json.dumps(info, allow_nan=False)
    finally:
        env.close()


def test_shutter_is_edge_triggered_and_enforces_cooldown_and_attempt_budget():
    photo = PhotoConfig(minimum_capture_seconds=0, capture_cooldown_seconds=0.2)
    env = PhotoFramingEnv(TaskConfig(randomize=False), photo)
    try:
        env.reset(seed=0)
        env.projection = lambda: (np.array([0.9, 0.9, 0.2, 0.2]), True, False)
        for _ in range(10):
            _, _, _, _, info = env.step([1, 1, 1, 1])
        assert info["photo"]["attempts"] == 1
        env.step([1, 1, 1, 0])
        _, _, _, _, info = env.step([1, 1, 1, 1])
        assert info["photo"]["attempts"] == 2
        env.step([1, 1, 1, 0])
        _, _, _, _, info = env.step([1, 1, 1, 1])
        assert info["photo"]["attempts"] == 2  # Still within cooldown.
        for _ in range(6):
            env.step([1, 1, 1, 0])
        _, _, terminated, _, info = env.step([1, 1, 1, 1])
        assert terminated and info["photo"]["attempts"] == 3
        assert info["photo"]["failure_reason"] == "attempt_budget"
    finally:
        env.close()


def test_capture_requires_stable_framing_not_just_centered_position():
    config = PhotoConfig(minimum_capture_seconds=0)
    env = PhotoFramingEnv(TaskConfig(randomize=False), config)
    try:
        env.reset(seed=0)
        env.goal[:] = 0
        env.last_observation[:2] = 0
        env.last_observation[5:7] = 0
        env.projection = lambda: (np.array([0, 0, 0.2, 0.3]), True, True)
        env.previous_true_center[:] = 0.8
        _, _, _, _, info = env.step([1, 1, 1, 1])
        shot = info["photo"]["shots"][0]
        assert shot["framing_error"] == 0
        assert shot["image_speed"] > config.image_speed_threshold
        assert not shot["valid"]
    finally:
        env.close()


def test_photo_synthetic_preview_and_failed_outcome_summary(tmp_path):
    env = PhotoFramingEnv()
    try:
        env.reset(seed=1)
        preview = env.preview()
        assert preview.shape == (360, 640, 3) and preview.dtype == np.uint8
        from PIL import Image
        path = tmp_path / "synthetic.png"
        Image.fromarray(preview).save(path)
        assert Image.open(path).size == (640, 360)
        class NeverCapture:
            def predict(self, obs, deterministic=True):
                return np.array([1, 1, 1, 0]), None
        episode = rollout(env, NeverCapture(), 1)
        summary = summarize("photo", [episode], 10)
        assert summary["success_rate"] == 0
        assert summary["mean_successful_capture_time_s"] is None
        assert summary["mean_capped_capture_time_s"] == 10
    finally:
        env.close()


def test_invalid_photo_settings_and_actions_are_rejected():
    for photo in (PhotoConfig(residual_scale=2), PhotoConfig(max_attempts=0),
                  PhotoConfig(baseline_wait_seconds=0), PhotoConfig(pause_start_max=9)):
        with pytest.raises(ValueError):
            photo.validate(TaskConfig())
    env = PhotoFramingEnv()
    try:
        env.reset(seed=0)
        with pytest.raises(ValueError):
            env.step([0, 0, 0])
        with pytest.raises(ValueError):
            env.step([0, 0, 0, np.nan])
        with pytest.raises(ValueError):
            env.step([0.5, 1, 1, 0])
    finally:
        env.close()
