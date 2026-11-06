"""Train and compare controllers in simulation. No hardware commands."""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np

from .assets import ROOT, asset_dir, fetch
from .env import ReachyTrackingEnv, TaskConfig, proportional_action


def checkpoint_path(value: str) -> Path:
    if value == "latest":
        paths = list((ROOT / "runs").glob("*/final.zip"))
        if not paths:
            raise FileNotFoundError("No final.zip found under runs/. Train first or supply --checkpoint PATH.")
        return max(paths, key=lambda p: p.stat().st_mtime)
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_config(checkpoint: Path) -> TaskConfig:
    path = checkpoint.parent / "env.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; keep env.json beside your checkpoint.")
    return TaskConfig(**json.loads(path.read_text()))


def new_run(name: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = ROOT / "runs" / f"{stamp}-{name}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def train(args):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv

    if args.timesteps <= 0 or args.num_envs <= 0:
        raise ValueError("timesteps and num-envs must be positive")
    if args.n_steps * args.num_envs < 2 or args.batch_size < 2:
        raise ValueError("PPO rollout and batch size must be at least two")
    if args.batch_size > args.n_steps * args.num_envs or (args.n_steps * args.num_envs) % args.batch_size:
        raise ValueError("batch-size must divide n-steps * num-envs")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu or check the Spark CUDA PyTorch build")
    torch.set_num_threads(args.torch_threads)
    resume = checkpoint_path(args.resume) if args.resume else None
    config = load_config(resume) if resume else TaskConfig()
    run_dir = new_run("smoke" if args.smoke else "ppo")
    (run_dir / "env.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    manifest = {
        "arguments": {key: value for key, value in vars(args).items() if key != "func"},
        "model": json.loads((asset_dir() / "source.json").read_text()),
        "packages": {name: version(name) for name in ("mujoco", "numpy", "torch", "stable-baselines3", "gymnasium", "reachy-mini-rust-kinematics")},
        "observation_size": 12, "action_size": 3,
    }
    (run_dir / "training.json").write_text(json.dumps(manifest, indent=2) + "\n")

    class MetricsCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.values = defaultdict(lambda: deque(maxlen=1000))
        def _on_step(self):
            for info in self.locals["infos"]:
                for key in ("tracking_error", "visible", "centered", "action_change", "invalid_pose", "numerical_failure"):
                    if key in info:
                        self.values[key].append(info[key])
            for key, values in self.values.items():
                self.logger.record(f"tracking/{key}", float(np.mean(values)))
            return True

    vec_kwargs = {}
    if args.vectorization == "subproc" and args.num_envs > 1:
        vec_kwargs = {"vec_env_cls": SubprocVecEnv, "vec_env_kwargs": {"start_method": "spawn"}}
    env = make_vec_env(ReachyTrackingEnv, n_envs=args.num_envs, seed=args.seed,
                       env_kwargs={"config": config}, monitor_dir=str(run_dir / "monitor"), **vec_kwargs)
    try:
        if resume:
            model = PPO.load(str(resume), env=env, device=args.device)
            # Explicitly retain checkpoint optimizer/rollout hyperparameters when resuming.
            model.tensorboard_log = str(run_dir / "tensorboard")
            manifest["resumed_from"] = str(resume)
            manifest["resume_uses_saved_hyperparameters"] = True
            (run_dir / "training.json").write_text(json.dumps(manifest, indent=2) + "\n")
        else:
            model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=args.n_steps,
                        batch_size=args.batch_size, n_epochs=args.epochs, gamma=0.99,
                        policy_kwargs={"net_arch": [64, 64]}, seed=args.seed,
                        tensorboard_log=str(run_dir / "tensorboard"), device=args.device, verbose=1)
        callbacks = [MetricsCallback(), CheckpointCallback(
            save_freq=max(1, args.checkpoint_every // args.num_envs),
            save_path=str(run_dir), name_prefix="checkpoint")]
        print(f"Run directory: {run_dir}\nAdditional requested transitions: {args.timesteps}", flush=True)
        try:
            model.learn(total_timesteps=args.timesteps, callback=callbacks, reset_num_timesteps=resume is None)
        except KeyboardInterrupt:
            print("Interrupted; saving current policy.")
        model.save(str(run_dir / "final.zip"))
        print(f"Saved {run_dir / 'final.zip'}")
    finally:
        env.close()


def evaluate(args):
    from stable_baselines3 import PPO
    path = checkpoint_path(args.checkpoint)
    config = load_config(path)
    model = PPO.load(str(path), device="cpu")
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    result = {"checkpoint": str(path), "seed": args.seed, "episodes": args.episodes, "controllers": {}}
    for controller in ("ppo", "proportional", "zero"):
        env = ReachyTrackingEnv(config)
        episodes = []
        try:
            for episode in range(args.episodes):
                observation, _ = env.reset(seed=args.seed + episode)
                stats = defaultdict(list)
                reward_sum = 0.0
                while True:
                    if controller == "ppo":
                        action, _ = model.predict(observation, deterministic=True)
                    elif controller == "proportional":
                        action = proportional_action(env, observation)
                    else:
                        action = np.zeros(3)
                    observation, reward, terminated, truncated, info = env.step(action)
                    reward_sum += reward
                    for key, value in info.items():
                        stats[key].append(value)
                    if terminated or truncated:
                        break
                episodes.append({"return": reward_sum, "length": env.steps,
                                 **{key: float(np.mean(values)) for key, values in stats.items()}})
        finally:
            env.close()
        summary = {key: {"mean": float(np.mean([ep[key] for ep in episodes])),
                         "std": float(np.std([ep[key] for ep in episodes]))} for key in episodes[0]}
        result["controllers"][controller] = {"summary": summary, "episodes": episodes}
        print(f"{controller:12s} return={summary['return']['mean']:8.2f} "
              f"error={summary['tracking_error']['mean']:.4f} "
              f"centered={summary['centered']['mean']:.1%} "
              f"visible={summary['visible']['mean']:.1%}")
    output = Path(args.output) if args.output else path.parent / f"evaluation-{args.seed}-{args.episodes}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Report: {output}")


def play(args):
    from stable_baselines3 import PPO
    from .viewer import BrowserViewer
    if args.controller == "ppo":
        path = checkpoint_path(args.checkpoint)
        config = load_config(path)
        model = PPO.load(str(path), device="cpu")
    else:
        config, model = TaskConfig(), None
    env = ReachyTrackingEnv(config)
    viewer = BrowserViewer(env, args.port) if not args.headless else None
    try:
        for episode in range(args.episodes):
            observation, _ = env.reset(seed=args.seed + episode)
            while True:
                start = time.monotonic()
                if model is not None:
                    action, _ = model.predict(observation, deterministic=True)
                elif args.controller == "proportional":
                    action = proportional_action(env, observation)
                else:
                    action = np.zeros(3)
                observation, _, terminated, truncated, info = env.step(action)
                if viewer:
                    viewer.update(info)
                    time.sleep(max(0, config.control_dt - (time.monotonic() - start)))
                if terminated or truncated:
                    print(f"Episode {episode + 1}: {info}")
                    break
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        if viewer:
            viewer.close()


def check(args):
    import torch
    from stable_baselines3.common.env_checker import check_env
    env = ReachyTrackingEnv(TaskConfig(randomize=False))
    try:
        check_env(env, warn=True)
        for seed in (0, 1, 2):
            obs, _ = env.reset(seed=seed)
            for _ in range(100):
                obs, _, terminated, truncated, info = env.step(proportional_action(env, obs))
                if terminated:
                    raise RuntimeError(f"Simulation failed: {info}")
                if truncated:
                    break
        print(f"Environment OK: obs={env.observation_space.shape}, actions={env.action_space.shape}")
        print(f"MuJoCo actuators: {env.model.nu}; CUDA available: {torch.cuda.is_available()}")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    assets = commands.add_parser("fetch-assets", help="Download hash-verified official model assets")
    assets.add_argument("--version", help="Explicitly refresh model.lock.json to this SDK release")
    assets.set_defaults(func=lambda args: fetch(args.version))
    commands.add_parser("check").set_defaults(func=check)
    for name in ("train", "smoke"):
        p = commands.add_parser(name)
        p.add_argument("--timesteps", type=int, default=2048 if name == "smoke" else 500_000)
        p.add_argument("--num-envs", type=int, default=2 if name == "smoke" else 8)
        p.add_argument("--n-steps", type=int, default=64 if name == "smoke" else 256)
        p.add_argument("--batch-size", type=int, default=64 if name == "smoke" else 256)
        p.add_argument("--epochs", type=int, default=2 if name == "smoke" else 10)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--device", default="cpu")
        p.add_argument("--torch-threads", type=int, default=1)
        p.add_argument("--vectorization", choices=("dummy", "subproc"), default="dummy" if name == "smoke" else "subproc")
        p.add_argument("--checkpoint-every", type=int, default=50_000)
        p.add_argument("--resume", help="Checkpoint path or latest; timesteps are additional")
        p.set_defaults(func=train, smoke=name == "smoke")
    p = commands.add_parser("evaluate")
    p.add_argument("--checkpoint", default="latest")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--output")
    p.set_defaults(func=evaluate)
    p = commands.add_parser("play")
    p.add_argument("--checkpoint", default="latest")
    p.add_argument("--controller", choices=("ppo", "proportional", "zero"), default="ppo")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--port", type=int, default=8081)
    p.add_argument("--headless", action="store_true")
    p.set_defaults(func=play)
    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
