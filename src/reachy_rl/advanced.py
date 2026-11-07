"""Independent residual-tracking and synthetic photo-framing RL workflows."""
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
from PIL import Image

from .assets import ROOT, asset_dir
from .cli import checkpoint_path, load_config
from .env import TaskConfig
from .occlusion import ReacquisitionConfig, ReacquisitionEnv
from .photo import PhotoConfig, PhotoFramingEnv
from .reacquisition import experiment_checkpoint
from .residual import ResidualConfig, ResidualTrackingEnv


def default_spec(kind, args=None):
    task = TaskConfig(max_delay_steps=getattr(args, "max_delay_steps", 2))
    spec = {"kind": kind, "version": 1, "task": asdict(task)}
    scale = getattr(args, "scale", 0.25)
    if kind == "residual":
        spec.update(occlusion=asdict(ReacquisitionConfig(history_steps=getattr(args, "history", 64))),
                    residual=asdict(ResidualConfig(scale=scale)))
    else:
        spec["photo"] = asdict(PhotoConfig(residual_scale=scale))
    return spec


def make_environment(spec, *, direct_history=None):
    task = TaskConfig(**spec["task"])
    if spec["kind"] == "residual":
        occlusion = ReacquisitionConfig(**spec["occlusion"])
        if direct_history is not None:
            return ReacquisitionEnv(task, replace(occlusion, history_steps=direct_history))
        return ResidualTrackingEnv(task, occlusion, ResidualConfig(**spec["residual"]))
    return PhotoFramingEnv(task, PhotoConfig(**spec["photo"]))


def find_checkpoint(kind, value):
    if value == "latest":
        candidates = list((ROOT / "runs" / kind).glob("*/final.zip"))
        if not candidates:
            raise FileNotFoundError(f"No {kind} checkpoint; train first or provide a checkpoint path")
        path = max(candidates, key=lambda item: item.stat().st_mtime)
    else:
        path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    spec_path = path.parent / "settings.json"
    if not spec_path.is_file():
        raise FileNotFoundError(f"Missing {spec_path}; keep run metadata beside its checkpoint")
    spec = json.loads(spec_path.read_text())
    if spec.get("kind") != kind or spec.get("version") != 1:
        raise ValueError(f"Checkpoint is not a supported {kind} experiment")
    return path, spec


def check_interface(model, env):
    if model.observation_space != env.observation_space or model.action_space != env.action_space:
        raise ValueError("Checkpoint observation/action spaces do not match this experiment")


def train(args):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv

    if min(args.timesteps, args.num_envs, args.n_steps, args.torch_threads, args.epochs, args.checkpoint_every) < 1:
        raise ValueError("Training counts must be positive")
    if args.max_delay_steps < 0 or args.batch_size < 2 or args.num_envs * args.n_steps % args.batch_size:
        raise ValueError("Delay must be nonnegative; batch-size must be at least two and divide the rollout")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; check the Spark PyTorch installation or use --device cpu")
    torch.set_num_threads(args.torch_threads)
    resume, spec = find_checkpoint(args.kind, args.resume) if args.resume else (None, default_spec(args.kind, args))
    # Validate the environment configuration before creating any output run directory.
    probe = make_environment(spec)
    shapes = {"observations": list(probe.observation_space.shape), "actions": list(probe.action_space.shape),
              "observation_space": str(probe.observation_space), "action_space": str(probe.action_space)}
    probe.close()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = ROOT / "runs" / args.kind / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "settings.json").write_text(json.dumps(spec, indent=2) + "\n")
    (run_dir / "env.json").write_text(json.dumps(spec["task"], indent=2) + "\n")
    manifest = {"arguments": {key: value for key, value in vars(args).items() if key != "func"},
                "model": json.loads((asset_dir() / "source.json").read_text()), "interfaces": shapes,
                "resumed_from": str(resume) if resume else None,
                "packages": {key: version(key) for key in ("mujoco", "numpy", "torch", "stable-baselines3", "pillow")}}
    (run_dir / "training.json").write_text(json.dumps(manifest, indent=2) + "\n")

    class Metrics(BaseCallback):
        def __init__(self):
            super().__init__()
            self.values = defaultdict(lambda: deque(maxlen=1000))
            self.outcomes = defaultdict(lambda: deque(maxlen=100))
        def _on_step(self):
            for done, info in zip(self.locals["dones"], self.locals["infos"]):
                for key in ("tracking_error", "visible", "centered", "action_change", "numerical_failure",
                            "correction_norm", "residual_clipped", "captured", "image_speed"):
                    if key in info:
                        self.values[key].append(info[key])
                if done:
                    report = info.get("reacquisition", info.get("photo"))
                    self.outcomes["success_rate"].append(float(report["success"]))
                    if "reacquisition" in info:
                        self.outcomes["capped_latency_s"].append(report["latency_s"] if report["success"] else report["timeout_s"])
                    else:
                        self.outcomes["first_capture_valid_rate"].append(float(report["first_capture_valid"]))
                        self.outcomes["attempts"].append(report["attempts"])
            for collection in (self.values, self.outcomes):
                for key, values in collection.items():
                    self.logger.record(f"{args.kind}/{key}", float(np.mean(values)))
            return True

    options = {}
    if args.vectorization == "subproc" and args.num_envs > 1:
        options = {"vec_env_cls": SubprocVecEnv, "vec_env_kwargs": {"start_method": "spawn"}}
    env = make_vec_env(make_environment, n_envs=args.num_envs, seed=args.seed,
                       env_kwargs={"spec": spec}, monitor_dir=str(run_dir / "monitor"), **options)
    try:
        if resume:
            model = PPO.load(str(resume), env=env, device=args.device)
            model.tensorboard_log = str(run_dir / "tensorboard")
        else:
            model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=args.n_steps,
                        batch_size=args.batch_size, n_epochs=args.epochs, gamma=0.99,
                        policy_kwargs={"net_arch": [128, 128]}, seed=args.seed, device=args.device,
                        tensorboard_log=str(run_dir / "tensorboard"), verbose=1)
            # Residual means start at zero. Photo categorical logits prefer neutral
            # corrections and waiting, but can learn a crisp capture/wait decision.
            with torch.no_grad():
                model.policy.action_net.weight.zero_()
                model.policy.action_net.bias.zero_()
                if args.kind == "photo":
                    for middle in (1, 4, 7):
                        model.policy.action_net.bias[middle] = 4.0
                    model.policy.action_net.bias[9] = 3.5
                else:
                    model.policy.log_std.fill_(np.log(0.5))
        callbacks = [Metrics(), CheckpointCallback(save_freq=max(1, args.checkpoint_every // args.num_envs),
                                                   save_path=str(run_dir), name_prefix="checkpoint")]
        print(f"Run: {run_dir}\nAdditional transitions: {args.timesteps}", flush=True)
        try:
            model.learn(total_timesteps=args.timesteps, callback=callbacks, reset_num_timesteps=resume is None)
        except KeyboardInterrupt:
            print("Interrupted; saving the current policy.")
        model.save(str(run_dir / "final.zip"))
        print(f"Saved {run_dir / 'final.zip'}")
    finally:
        env.close()


def rollout(env, model, seed, capture_dir=None):
    obs, _ = env.reset(seed=seed)
    values = defaultdict(list)
    total, captured = 0.0, 0
    while True:
        action = model.predict(obs, deterministic=True)[0] if model else env.baseline_action()
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        for key in ("tracking_error", "action_change", "numerical_failure", "correction_norm", "residual_clipped"):
            if key in info:
                values[key].append(info[key])
        if "photo" in info and len(info["photo"]["shots"]) > captured:
            captured = len(info["photo"]["shots"])
            if capture_dir is not None:
                capture_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(env.preview()).save(capture_dir / f"seed-{seed}-attempt-{captured}.png")
        if terminated or truncated:
            break
    report = info.get("reacquisition", info.get("photo"))
    return {"seed": seed, "return": total, "outcome": report,
            **{key: float(np.mean(data)) for key, data in values.items()}}


def summarize(kind, episodes, episode_seconds):
    outcomes = [episode["outcome"] for episode in episodes]
    result = {"success_rate": float(np.mean([row["success"] for row in outcomes])),
              "mean_return": float(np.mean([row["return"] for row in episodes])),
              "mean_tracking_error": float(np.mean([row["tracking_error"] for row in episodes])),
              "mean_action_change": float(np.mean([row["action_change"] for row in episodes])),
              "mean_numerical_failure_fraction": float(np.mean([row["numerical_failure"] for row in episodes])),
              "numerical_failure_rate": float(np.mean([row["numerical_failure"] > 0 for row in episodes]))}
    if kind == "residual":
        successes = [row["latency_s"] for row in outcomes if row["success"]]
        capped = [row["latency_s"] if row["success"] else row["timeout_s"] for row in outcomes]
        result.update(mean_successful_latency_s=float(np.mean(successes)) if successes else None,
                      mean_capped_latency_s=float(np.mean(capped)), p95_capped_latency_s=float(np.quantile(capped, 0.95)))
    else:
        successes = [row["capture_time_s"] for row in outcomes if row["success"]]
        result.update(first_capture_valid_rate=float(np.mean([row["first_capture_valid"] for row in outcomes])),
                      mean_attempts=float(np.mean([row["attempts"] for row in outcomes])),
                      mean_successful_capture_time_s=float(np.mean(successes)) if successes else None,
                      mean_capped_capture_time_s=float(np.mean([row["capture_time_s"] if row["success"] else episode_seconds for row in outcomes])))
    return result


def evaluate(args):
    from stable_baselines3 import PPO

    if args.episodes < 1 or (args.max_delay_steps is not None and args.max_delay_steps < 0):
        raise ValueError("episodes must be positive and max-delay-steps nonnegative")
    path, training_spec = find_checkpoint(args.kind, args.checkpoint)
    spec = json.loads(json.dumps(training_spec))
    if args.max_delay_steps is not None:
        spec["task"]["max_delay_steps"] = args.max_delay_steps
    output = Path(args.output).expanduser() if args.output else path.parent / f"evaluation-{args.seed}-{args.episodes}-delay{spec['task']['max_delay_steps']}.json"
    baseline_name = "proportional_baseline" if args.kind == "residual" else "framing_capture_baseline"
    controllers = [("learned", path, None), (baseline_name, None, None)]
    if args.kind == "residual":
        if args.compare_reacquisition:
            controllers.append(("direct_reacquisition_ppo", experiment_checkpoint(args.compare_reacquisition, history=64), "direct"))
        if args.compare_tracking:
            controllers.append(("direct_tracking_ppo", checkpoint_path(args.compare_tracking), "direct"))
    report = {"kind": args.kind, "episodes": args.episodes, "seed": args.seed,
              "training_settings": training_spec, "evaluation_settings": spec, "controllers": {}}
    for name, model_path, variant in controllers:
        model = PPO.load(str(model_path), device="cpu") if model_path else None
        if variant == "direct":
            compare_task = asdict(load_config(model_path))
            expected_task = dict(training_spec["task"])
            if args.max_delay_steps is not None:
                compare_task.pop("max_delay_steps")
                expected_task.pop("max_delay_steps")
            if compare_task != expected_task:
                raise ValueError(f"Base task differs for comparison checkpoint {model_path}")
            shape = model.observation_space.shape
            if len(shape) != 1 or shape[0] % 12:
                raise ValueError("Comparison must use the original tracking/reacquisition observation interface")
            env = make_environment(spec, direct_history=shape[0] // 12)
        else:
            env = make_environment(spec)
        episodes = []
        try:
            if model:
                check_interface(model, env)
            for index in range(args.episodes):
                captures = output.parent / (output.stem + "-captures") / name if args.save_captures and args.kind == "photo" else None
                episodes.append(rollout(env, model, args.seed + index, captures))
        finally:
            env.close()
        summary = summarize(args.kind, episodes, spec["task"]["episode_seconds"])
        report["controllers"][name] = {"checkpoint": str(model_path) if model_path else None,
                                       "summary": summary, "episodes": episodes}
        print(f"{name:27s} success={summary['success_rate']:.1%} error={summary['mean_tracking_error']:.4f} "
              f"action_change={summary['mean_action_change']:.4f}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Report: {output}")


def playback_environment(args):
    from stable_baselines3 import PPO
    try:
        path, spec = find_checkpoint(args.kind, args.checkpoint)
    except FileNotFoundError:
        if args.controller != "baseline" or args.checkpoint != "latest":
            raise
        path, spec = None, default_spec(args.kind)
    env = make_environment(spec)
    try:
        model = PPO.load(str(path), device="cpu") if args.controller == "ppo" else None
        if model:
            check_interface(model, env)
        return env, model
    except BaseException:
        env.close()
        raise


def play(args):
    from .viewer import BrowserViewer
    if args.episodes < 1:
        raise ValueError("episodes must be positive")
    env, model = playback_environment(args)
    viewer, preview = None, None
    try:
        if not args.headless:
            viewer = BrowserViewer(env.unwrapped, args.port)
            if args.kind == "photo":
                env.reset(seed=args.seed)
                preview = viewer.server.gui.add_image(env.preview(), label="Synthetic framing preview", format="jpeg")
        for index in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + index)
            shots = 0
            while True:
                start = time.monotonic()
                action = model.predict(obs, deterministic=True)[0] if model else env.baseline_action()
                obs, _, terminated, truncated, info = env.step(action)
                if args.kind == "photo" and len(info["photo"]["shots"]) > shots:
                    shots = len(info["photo"]["shots"])
                    if args.save_captures:
                        directory = Path(args.save_captures).expanduser()
                        directory.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(env.preview()).save(directory / f"seed-{args.seed+index}-attempt-{shots}.png")
                if viewer:
                    viewer.update(info)
                    viewer.target.visible = not bool(info.get("occluded", False))
                    if preview:
                        preview.image = env.preview()
                    if args.kind == "residual":
                        outcome = info["reacquisition"]
                        details = (f"Target hidden: {bool(info.get('occluded', False))}\n\n"
                                   f"Recovery: {outcome['status']}\n\n"
                                   f"Recovery time: {outcome['latency_s']}")
                    else:
                        outcome = info["photo"]
                        goal = "Center" if abs(outcome["goal"][0]) < 0.01 else ("Left third" if outcome["goal"][0] < 0 else "Right third")
                        last = outcome["shots"][-1] if outcome["shots"] else None
                        shot = "None yet" if last is None else ("Valid" if last["valid"] else "Invalid")
                        details = (f"Composition: {goal}\n\nAttempts: {outcome['attempts']}/{env.photo.max_attempts}\n\n"
                                   f"Last capture: {shot}\n\nSuccessful capture: {outcome['success']}")
                    viewer.status.content = (f"**{args.kind.title()} experiment — simulation only**\n\n"
                                              f"Framing error: {info['tracking_error']:.3f}\n\n{details}")
                    time.sleep(max(0, env.unwrapped.config.control_dt - (time.monotonic() - start)))
                if terminated or truncated:
                    print(json.dumps(info.get("reacquisition", info.get("photo"))))
                    break
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        if viewer:
            viewer.close()


def record(args):
    from .recording import record as record_video
    env, model = playback_environment(args)
    try:
        record_video(args, simulation=(env, model))
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    kinds = parser.add_subparsers(dest="kind", required=True)
    for kind in ("residual", "photo"):
        commands = kinds.add_parser(kind).add_subparsers(dest="command", required=True)
        for name in ("train", "smoke"):
            p = commands.add_parser(name)
            p.add_argument("--timesteps", type=int, default=2048 if name == "smoke" else 500000)
            p.add_argument("--num-envs", type=int, default=2 if name == "smoke" else 8)
            p.add_argument("--n-steps", type=int, default=64 if name == "smoke" else 256)
            p.add_argument("--batch-size", type=int, default=64 if name == "smoke" else 256)
            p.add_argument("--epochs", type=int, default=2 if name == "smoke" else 10)
            p.add_argument("--scale", type=float, default=0.25)
            p.add_argument("--max-delay-steps", type=int, default=2)
            if kind == "residual":
                p.add_argument("--history", type=int, default=64)
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--device", default="cpu")
            p.add_argument("--torch-threads", type=int, default=1)
            p.add_argument("--checkpoint-every", type=int, default=50000)
            p.add_argument("--vectorization", choices=("dummy", "subproc"), default="dummy" if name == "smoke" else "subproc")
            p.add_argument("--resume")
            p.set_defaults(func=train)
        p = commands.add_parser("evaluate")
        p.add_argument("--checkpoint", default="latest")
        p.add_argument("--episodes", type=int, default=50)
        p.add_argument("--seed", type=int, default=10000)
        p.add_argument("--max-delay-steps", type=int, help="Explicit evaluation-only delay stress override")
        p.add_argument("--output")
        p.add_argument("--save-captures", action="store_true", help="Save labelled synthetic PNG previews for photo attempts")
        if kind == "residual":
            p.add_argument("--compare-reacquisition")
            p.add_argument("--compare-tracking")
        p.set_defaults(func=evaluate)
        p = commands.add_parser("play")
        p.add_argument("--checkpoint", default="latest")
        p.add_argument("--controller", choices=("ppo", "baseline"), default="ppo")
        p.add_argument("--episodes", type=int, default=100)
        p.add_argument("--seed", type=int, default=10000)
        p.add_argument("--port", type=int, default=8081)
        p.add_argument("--headless", action="store_true")
        p.add_argument("--save-captures", help="Directory for synthetic photo-attempt PNG previews")
        p.set_defaults(func=play)
        p = commands.add_parser("record")
        p.add_argument("--checkpoint", default="latest")
        p.add_argument("--controller", choices=("ppo", "baseline"), default="ppo")
        p.add_argument("--seconds", type=float, default=10)
        p.add_argument("--fps", type=int, default=25)
        p.add_argument("--width", type=int, default=1280)
        p.add_argument("--height", type=int, default=720)
        p.add_argument("--distance", type=float, default=2.6)
        p.add_argument("--azimuth", type=float, default=135)
        p.add_argument("--elevation", type=float, default=-20)
        p.add_argument("--output", default=f"videos/{kind}.mp4")
        p.add_argument("--seed", type=int, default=10000)
        p.add_argument("--overwrite", action="store_true")
        p.set_defaults(func=record)
    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
