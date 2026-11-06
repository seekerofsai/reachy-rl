import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from reachy_rl.env import ReachyTrackingEnv, TaskConfig, proportional_action


def test_gym_contract_and_seeded_reset():
    env = ReachyTrackingEnv()
    try:
        check_env(env)
        obs_a, _ = env.reset(seed=23)
        transition_a = env.step(np.array([0.1, -0.2, 0.05]))
        obs_b, _ = env.reset(seed=23)
        transition_b = env.step(np.array([0.1, -0.2, 0.05]))
        np.testing.assert_array_equal(obs_a, obs_b)
        np.testing.assert_array_equal(transition_a[0], transition_b[0])
        assert transition_a[1:] == transition_b[1:]
    finally:
        env.close()


def test_random_actions_stay_finite_and_within_pose_bounds():
    env = ReachyTrackingEnv()
    rng = np.random.default_rng(9)
    try:
        for seed in range(3):
            env.reset(seed=seed)
            for step in range(env.episode_steps):
                obs, reward, terminated, truncated, info = env.step(rng.uniform(-1, 1, 3))
                assert np.isfinite(reward)
                assert env.observation_space.contains(obs)
                assert not terminated, info
                assert np.all(np.abs(env.command) <= env.limits + 1e-10)
                assert truncated == (step == env.episode_steps - 1)
    finally:
        env.close()


def test_bad_action_and_unreachable_pose_are_rejected():
    env = ReachyTrackingEnv()
    try:
        env.reset(seed=0)
        with pytest.raises(ValueError):
            env.step([np.nan, 0, 0])
        with pytest.raises(ValueError):
            env.step([0, 0])
        with pytest.raises(ValueError):
            env._joint_targets([np.deg2rad(40), 0, np.deg2rad(-30)])
    finally:
        env.close()


def test_proportional_baseline_beats_no_motion():
    # This also checks camera/action sign conventions against actual MuJoCo motion.
    config = TaskConfig(randomize=False)
    means = []
    for use_baseline in (False, True):
        errors = []
        env = ReachyTrackingEnv(config)
        try:
            for seed in range(4):
                obs, _ = env.reset(seed=seed)
                for _ in range(env.episode_steps):
                    action = proportional_action(env, obs) if use_baseline else np.zeros(3)
                    obs, _, terminated, truncated, info = env.step(action)
                    assert not terminated
                    errors.append(info["tracking_error"])
                    if truncated:
                        break
        finally:
            env.close()
        means.append(np.mean(errors))
    assert means[1] < 0.8 * means[0], means
