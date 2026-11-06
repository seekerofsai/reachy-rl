"""Train, compare and view the target-reacquisition experiment."""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np

from .assets import ROOT, asset_dir
from .cli import checkpoint_path, load_config
from .env import TaskConfig, proportional_action
from .occlusion import ReacquisitionConfig, ReacquisitionEnv


def experiment_checkpoint(value, history=None):
    if value != "latest":
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    candidates = []
    for path in (ROOT / "runs" / "reacquisition").glob("*/final.zip"):
        config = read_experiment(path)
        if history is None or config.history_steps == history:
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No reacquisition checkpoint for history={history}; train it first or supply a path")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def read_experiment(path):
    metadata = path.parent / "experiment.json"
    if not metadata.is_file():
        raise FileNotFoundError(f"Missing {metadata}; use --tracking for an original tracking checkpoint")
    return ReacquisitionConfig(**json.loads(metadata.read_text()))


def validate_policy(model, history):
    if model.observation_space.shape != (12 * history,) or model.action_space.shape != (3,):
        raise ValueError(f"Checkpoint interface does not match {history} frames of 12 observations and 3 actions")


def train(args):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv

    if min(args.timesteps, args.num_envs, args.n_steps, args.torch_threads) <= 0 or args.batch_size < 2:
        raise ValueError("Training counts must be positive and batch-size at least 2")
    if args.batch_size > args.num_envs * args.n_steps or (args.num_envs * args.n_steps) % args.batch_size:
        raise ValueError("batch-size must divide num-envs * n-steps")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; use --device cpu or check the Spark PyTorch build")
    torch.set_num_threads(args.torch_threads)
    resume = experiment_checkpoint(args.resume) if args.resume else None
    task = load_config(resume) if resume else TaskConfig()
    config = read_experiment(resume) if resume else ReacquisitionConfig(history_steps=args.history)
    config.validate(task)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = ROOT / "runs" / "reacquisition" / f"{stamp}-h{config.history_steps}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "env.json").write_text(json.dumps(asdict(task), indent=2) + "\n")
    (run_dir / "experiment.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    manifest = {
        "experiment": "reacquisition-v1",
        "arguments": {key: value for key, value in vars(args).items() if key != "func"},
        "model": json.loads((asset_dir() / "source.json").read_text()),
        "packages": {name: version(name) for name in ("mujoco", "numpy", "torch", "gymnasium", "stable-baselines3")},
        "observation_size": 12 * config.history_steps,
        "resumed_from": str(resume) if resume else None,
        "resume_uses_saved_ppo_hyperparameters": bool(resume),
    }
    (run_dir / "training.json").write_text(json.dumps(manifest, indent=2) + "\n")

    class Metrics(BaseCallback):
        def __init__(self):
            super().__init__()
            self.steps = defaultdict(lambda: deque(maxlen=1000))
            self.successes, self.capped_latencies = deque(maxlen=100), deque(maxlen=100)
        def _on_step(self):
            for done, info in zip(self.locals["dones"], self.locals["infos"]):
                for key in ("tracking_error", "visible", "occluded", "centered", "action_change", "numerical_failure"):
                    self.steps[key].append(info[key])
                if done:
                    report = info["reacquisition"]
                    self.successes.append(float(report["success"]))
                    self.capped_latencies.append(report["latency_s"] if report["success"] else report["timeout_s"])
            for key, values in self.steps.items():
                self.logger.record(f"reacquisition/{key}", float(np.mean(values)))
            if self.successes:
                self.logger.record("reacquisition/success_rate", float(np.mean(self.successes)))
                self.logger.record("reacquisition/capped_latency_s", float(np.mean(self.capped_latencies)))
            return True

    vectorization = {}
    if args.vectorization == "subproc" and args.num_envs > 1:
        vectorization = {"vec_env_cls": SubprocVecEnv, "vec_env_kwargs": {"start_method": "spawn"}}
    env = make_vec_env(ReacquisitionEnv, n_envs=args.num_envs, seed=args.seed,
                       env_kwargs={"task": task, "experiment": config},
                       monitor_dir=str(run_dir / "monitor"), **vectorization)
    try:
        if resume:
            model = PPO.load(str(resume), env=env, device=args.device)
            model.tensorboard_log = str(run_dir / "tensorboard")
        else:
            model = PPO("MlpPolicy", env, n_steps=args.n_steps, batch_size=args.batch_size,
                        n_epochs=args.epochs, learning_rate=3e-4, gamma=0.99,
                        policy_kwargs={"net_arch": [128, 128]}, seed=args.seed, device=args.device,
                        tensorboard_log=str(run_dir / "tensorboard"), verbose=1)
        callbacks = [Metrics(), CheckpointCallback(save_freq=max(1, args.checkpoint_every // args.num_envs),
                                                   save_path=str(run_dir), name_prefix="checkpoint")]
        print(f"Run: {run_dir}\nObservation history: {config.history_steps} frames\nAdditional transitions: {args.timesteps}", flush=True)
        try:
            model.learn(total_timesteps=args.timesteps, callback=callbacks, reset_num_timesteps=resume is None)
        except KeyboardInterrupt:
            print("Interrupted; saving the current policy.")
        model.save(str(run_dir / "final.zip"))
        print(f"Saved {run_dir / 'final.zip'}")
    finally:
        env.close()


def evaluate(args):
    from stable_baselines3 import PPO

    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    memory_path = experiment_checkpoint(args.memory, history=64)
    task, benchmark = load_config(memory_path), read_experiment(memory_path)
    if benchmark.history_steps <= 1:
        raise ValueError("--memory must select a policy with more than one observation frame")
    controllers = [("history_ppo", memory_path, benchmark.history_steps)]
    if args.feedforward:
        path = experiment_checkpoint(args.feedforward, history=1)
        if read_experiment(path).history_steps != 1:
            raise ValueError("--feedforward requires a history=1 reacquisition checkpoint")
        controllers.append(("occlusion_ppo_no_history", path, 1))
    if args.tracking:
        controllers.append(("original_tracking_ppo", checkpoint_path(args.tracking), 1))
    controllers.append(("proportional", None, 1))
    result = {"experiment": "reacquisition-v1", "seed": args.seed, "episodes": args.episodes,
              "benchmark_config": asdict(benchmark), "task_config": asdict(task), "controllers": {}}
    for name, path, history in controllers:
        if path and asdict(load_config(path)) != asdict(task):
            raise ValueError(f"Base environment differs for {path}; compare checkpoints trained with the same env.json")
        model = PPO.load(str(path), device="cpu") if path else None
        if model:
            validate_policy(model, history)
        env = ReacquisitionEnv(task, replace(benchmark, history_steps=history))
        episodes = []
        try:
            for index in range(args.episodes):
                obs, _ = env.reset(seed=args.seed + index)
                errors, post_errors, changes = [], [], []
                episode_return = 0.0
                numerical_failure = False
                while True:
                    action = model.predict(obs, deterministic=True)[0] if model else proportional_action(env.unwrapped, obs[-12:])
                    obs, reward, terminated, truncated, info = env.step(action)
                    errors.append(info["tracking_error"])
                    changes.append(info["action_change"])
                    report = info["reacquisition"]
                    now = env.unwrapped.steps * task.control_dt
                    if report["occlusion_end_s"] <= now <= report["occlusion_end_s"] + report["timeout_s"] + 1e-9:
                        post_errors.append(info["tracking_error"])
                    episode_return += reward
                    numerical_failure = numerical_failure or bool(info["numerical_failure"])
                    if terminated or truncated:
                        break
                episodes.append({
                    "seed": args.seed + index, **report,
                    "return": episode_return, "mean_tracking_error": float(np.mean(errors)),
                    "post_occlusion_error": float(np.mean(post_errors)) if post_errors else None,
                    "mean_action_change": float(np.mean(changes)), "numerical_failure": numerical_failure,
                })
        finally:
            env.close()
        successes = [row["latency_s"] for row in episodes if row["success"]]
        post_errors = [row["post_occlusion_error"] for row in episodes if row["post_occlusion_error"] is not None]
        summary = {
            "success_rate": float(np.mean([row["success"] for row in episodes])),
            "mean_successful_latency_s": float(np.mean(successes)) if successes else None,
            "mean_capped_latency_s": float(np.mean([row["latency_s"] if row["success"] else row["timeout_s"] for row in episodes])),
            "centered_before_occlusion_rate": float(np.mean([row["centered_before_occlusion"] for row in episodes])),
            "mean_post_occlusion_error": float(np.mean(post_errors)) if post_errors else None,
            "mean_tracking_error": float(np.mean([row["mean_tracking_error"] for row in episodes])),
            "mean_action_change": float(np.mean([row["mean_action_change"] for row in episodes])),
            "numerical_failure_rate": float(np.mean([row["numerical_failure"] for row in episodes])),
        }
        result["controllers"][name] = {"checkpoint": str(path) if path else None, "history_steps": history,
                                        "summary": summary, "episodes": episodes}
        latency = summary["mean_successful_latency_s"]
        display = f"{latency:.3f}s" if latency is not None else "n/a"
        print(f"{name:26s} success={summary['success_rate']:.1%} successful_latency={display} "
              f"capped_latency={summary['mean_capped_latency_s']:.3f}s")
    output = Path(args.output).expanduser() if args.output else memory_path.parent / f"reacquisition-evaluation-{args.seed}-{args.episodes}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Report: {output}")


def play(args):
    from stable_baselines3 import PPO
    from .viewer import BrowserViewer

    if args.controller == "ppo":
        path = experiment_checkpoint(args.checkpoint, history=64)
        task, config = load_config(path), read_experiment(path)
        model = PPO.load(str(path), device="cpu")
        validate_policy(model, config.history_steps)
    else:
        task, config, model = TaskConfig(), ReacquisitionConfig(history_steps=1), None
    env = ReacquisitionEnv(task, config)
    viewer = None
    try:
        viewer = BrowserViewer(env.unwrapped, args.port) if not args.headless else None
        for index in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + index)
            while True:
                start = time.monotonic()
                action = model.predict(obs, deterministic=True)[0] if model else proportional_action(env.unwrapped, obs[-12:])
                obs, _, terminated, truncated, info = env.step(action)
                if viewer:
                    viewer.update(info)
                    viewer.target.visible = not bool(info["occluded"])
                    report = info["reacquisition"]
                    viewer.status.content = (f"**Reacquisition experiment — simulation only**\n\n"
                                              f"Occluded: {bool(info['occluded'])}\n\n"
                                              f"Detected: {bool(info['visible'])}\n\n"
                                              f"Camera-plane error: {info['tracking_error']:.3f}\n\n"
                                              f"Recovery: {report['status']}\n\nLatency: {report['latency_s']}")
                    time.sleep(max(0, task.control_dt - (time.monotonic() - start)))
                if terminated or truncated:
                    print(f"Episode {index + 1}: {json.dumps(info['reacquisition'])}")
                    break
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        if viewer:
            viewer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "smoke"):
        p = commands.add_parser(name)
        p.add_argument("--history", type=int, default=64, help="Past observation frames; 64=2.56s, 1=no history")
        p.add_argument("--timesteps", type=int, default=2048 if name == "smoke" else 500000)
        p.add_argument("--num-envs", type=int, default=2 if name == "smoke" else 8)
        p.add_argument("--n-steps", type=int, default=64 if name == "smoke" else 256)
        p.add_argument("--batch-size", type=int, default=64 if name == "smoke" else 256)
        p.add_argument("--epochs", type=int, default=2 if name == "smoke" else 10)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--device", default="cpu")
        p.add_argument("--torch-threads", type=int, default=1)
        p.add_argument("--checkpoint-every", type=int, default=50000)
        p.add_argument("--vectorization", choices=("dummy", "subproc"), default="dummy" if name == "smoke" else "subproc")
        p.add_argument("--resume", help="Reacquisition checkpoint path or latest; restores saved experiment settings")
        p.set_defaults(func=train)
    p = commands.add_parser("evaluate")
    p.add_argument("--memory", default="latest", help="History policy path; latest selects the newest h64 final.zip")
    p.add_argument("--feedforward", help="Optional history=1 policy trained on occlusions; path or latest")
    p.add_argument("--tracking", help="Optional original tracking PPO checkpoint; path or latest")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--output")
    p.set_defaults(func=evaluate)
    p = commands.add_parser("play")
    p.add_argument("--checkpoint", default="latest")
    p.add_argument("--controller", choices=("ppo", "proportional"), default="ppo")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=10000)
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
