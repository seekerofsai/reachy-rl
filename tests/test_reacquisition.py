import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from reachy_rl.env import TaskConfig
from reachy_rl.occlusion import ReacquisitionConfig, ReacquisitionEnv


def test_history_environment_contract_and_reset_padding():
    env = ReacquisitionEnv(experiment=ReacquisitionConfig(history_steps=64))
    try:
        check_env(env)
        first, report_a = env.reset(seed=12)
        assert first.shape == (768,)
        np.testing.assert_array_equal(first.reshape(64, 12), np.tile(first[-12:], (64, 1)))
        transition_a = env.step([0.2, -0.1, 0.05])
        again, report_b = env.reset(seed=12)
        transition_b = env.step([0.2, -0.1, 0.05])
        np.testing.assert_array_equal(first, again)
        np.testing.assert_array_equal(transition_a[0], transition_b[0])
        assert report_a == report_b
        assert transition_a[1:] == transition_b[1:]
    finally:
        env.close()


def test_history_does_not_change_targets_outages_or_current_observations():
    one = ReacquisitionEnv(experiment=ReacquisitionConfig(history_steps=1))
    memory = ReacquisitionEnv(experiment=ReacquisitionConfig(history_steps=64))
    try:
        obs_one, info_one = one.reset(seed=77)
        obs_memory, info_memory = memory.reset(seed=77)
        assert info_one == info_memory
        np.testing.assert_array_equal(obs_one, obs_memory[-12:])
        for _ in range(one.unwrapped.episode_steps):
            obs_one, _, ended, truncated, info_one = one.step(np.zeros(3))
            obs_memory, _, ended_other, truncated_other, info_memory = memory.step(np.zeros(3))
            np.testing.assert_array_equal(obs_one, obs_memory[-12:])
            np.testing.assert_array_equal(one.unwrapped.target_position, memory.unwrapped.target_position)
            assert info_one["reacquisition"] == info_memory["reacquisition"]
            assert (ended, truncated) == (ended_other, truncated_other)
    finally:
        one.close()
        memory.close()


def test_forced_occlusion_hides_detections_but_target_keeps_moving():
    env = ReacquisitionEnv(task=TaskConfig(randomize=False))
    try:
        obs, info = env.reset(seed=4)
        report = info["reacquisition"]
        assert 1.0 <= report["occlusion_duration_s"] <= 2.0
        last_error = obs[-12:-10].copy()
        target_positions = []
        for _ in range(env.end_step):
            obs, _, terminated, _, info = env.step(np.zeros(3))
            assert not terminated
            frame = obs[-12:]
            if info["occluded"]:
                assert frame[2] == 0 and info["visible"] == 0
                np.testing.assert_array_equal(frame[:2], last_error)
                target_positions.append(env.unwrapped.target_position.copy())
            elif frame[2]:
                last_error = frame[:2].copy()
        assert len(target_positions) == env.end_step - env.start_step
        assert np.linalg.norm(target_positions[0] - target_positions[-1]) > 0.01
        assert not info["occluded"]
    finally:
        env.close()


def tiny_experiment():
    return ReacquisitionConfig(history_steps=1, start_min_seconds=0.08, start_max_seconds=0.08,
                               duration_min_seconds=0.08, duration_max_seconds=0.08,
                               recovery_timeout_seconds=0.12, hold_seconds=0.04)


def test_recovery_requires_stable_recentering_after_reappearance():
    env = ReacquisitionEnv(TaskConfig(randomize=False, episode_seconds=0.48), tiny_experiment())
    try:
        env.reset(seed=0)
        env.unwrapped.tracking_error = lambda: (np.zeros(2), True)
        for step in range(1, 6):
            _, _, _, _, info = env.step(np.zeros(3))
            if step < 5:
                assert not info["reacquisition"]["success"]
        report = info["reacquisition"]
        assert report["success"] and report["status"] == "recovered"
        assert report["latency_s"] == pytest.approx(0.04)
    finally:
        env.close()


def test_failure_remains_in_report_with_no_invented_latency():
    env = ReacquisitionEnv(TaskConfig(randomize=False, episode_seconds=0.48), tiny_experiment())
    try:
        env.reset(seed=0)
        env.unwrapped.tracking_error = lambda: (np.array([0.8, 0]), True)
        for _ in range(7):
            _, _, _, _, info = env.step(np.zeros(3))
        report = info["reacquisition"]
        assert report["status"] == "failed" and not report["success"]
        assert report["latency_s"] is None
        assert report["timeout_s"] == pytest.approx(0.12)
    finally:
        env.close()


def test_invalid_history_and_recovery_windows_are_rejected():
    for config in (ReacquisitionConfig(history_steps=0),
                   ReacquisitionConfig(start_max_seconds=9),
                   ReacquisitionConfig(hold_seconds=3),
                   ReacquisitionConfig(duration_min_seconds=3, duration_max_seconds=2)):
        with pytest.raises(ValueError):
            config.validate(TaskConfig())
