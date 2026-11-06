# Reachy Mini RL on DGX Spark

A working simulation-first starting point for learning smooth gaze tracking:
the official Reachy Mini MuJoCo model, its analytical inverse kinematics, a
Gymnasium task, Stable-Baselines3 PPO, evaluation, TensorBoard, and browser playback.

This is a custom RL task for **Reachy Mini / Mini Lite**, not an RL feature of
NVIDIA's photo-booth playbook. Targets are synthetic moving 3D points, projected
through the robot model's eye camera. No person detector, camera imagery, speech,
LLM, real robot connection, or hardware deployment is included. A trained model
from this task is an experiment, not a validated physical robot controller.

## Quick start on the Spark

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then copy
this repository from the Mac if it is not already on the Spark:

```bash
# On the Mac:
ssh saiseeker@spark-c64c 'mkdir -p ~/repos'
scp -r ~/repos/reachy-rl saiseeker@spark-c64c:~/repos/

# On the Spark:
cd ~/repos/reachy-rl
./scripts/setup.sh
./scripts/check.sh
./scripts/smoke.sh
./scripts/evaluate.sh --episodes 5
```

Setup creates an isolated Python 3.12 environment. It downloads the official
Reachy Mini wheel pinned in `model.lock.json`, verifies its SHA256, and extracts
only model/kinematics data and license files into ignored `vendor/`. The full SDK
is not installed or imported, so simulation does not require its media stack,
USB permissions, GStreamer, or a robot daemon. Downloaded assets retain upstream
licensing. Install/setup do not alter your Microduck/Isaac Lab environment.

Linux ARM64 selects PyTorch from the CUDA 13.0 wheel index. Package resolution
and actual CUDA support on your Spark still need to be checked there. MuJoCo
simulation in this project runs on the CPU; CUDA only applies to neural-network
training. The small MLP defaults to CPU, which should be benchmarked against GPU
before choosing `--device cuda`. The default 8 CPU environments do not imply
Microduck's thousands of GPU simulations or its throughput.

The smoke command trains for 2,048 transitions to verify the pipeline. It does
not establish that the policy learned to track. Evaluation includes the hand-tuned
proportional baseline and a no-motion baseline so weak policies are visible.

## Train

```bash
tmux new -s reachy-rl
./scripts/train.sh --timesteps 500000 --num-envs 8
```

Detach with Ctrl+B, then D. Reattach with `tmux attach -t reachy-rl`.
Start with fewer environments if CPU or memory is constrained:

```bash
./scripts/train.sh --timesteps 500000 --num-envs 2 --vectorization dummy
# Optional GPU comparison (requires working CUDA PyTorch):
./scripts/train.sh --timesteps 500000 --num-envs 8 --device cuda
```

Every run gets a separate directory under `runs/`, including `env.json`,
`training.json` (arguments, package versions and model hash), TensorBoard logs,
Monitor logs, periodic checkpoints, and `final.zip`. Ctrl+C saves the current
model. PPO collects complete rollouts, so the achieved transition count may
slightly exceed the requested budget. Timesteps count environment transitions,
not PPO iterations or elapsed seconds.

Resume a chosen checkpoint with an **additional** transition budget:

```bash
./scripts/train.sh --resume /absolute/path/to/run/final.zip --timesteps 250000
```

Keep `env.json` next to the model. Resume restores the saved optimizer and PPO
hyperparameters; rollout, minibatch, epoch, and seed CLI changes do not replace
those stored settings. Use a new training run to change hyperparameters.
`--num-envs`, `--device`, and the additional `--timesteps` budget apply on resume.

## Task definition

The policy has 12 observations: target camera-plane error (2), detection flag
(1), requested head yaw/pitch/body yaw (3), measured head yaw/pitch/body yaw (3),
and previous action (3). Values are normalized/clipped; absent detections hold
the last detected error and set the flag to zero. There is no privileged target
velocity or future trajectory in the observation.

Its 3 actions are bounded pose rates for head yaw, head pitch, and body yaw at
25 Hz. Desired poses go through the analytical IK for the parallel neck, then
the model's position actuators. Conservative requested-pose bounds are +/-40
degrees head yaw, +/-20 degrees pitch, +/-30 degrees body yaw, with at most a
65-degree head/body yaw difference. Unreachable IK targets hold the previous
command and incur a penalty. These are training-task constraints, not a complete
physical robot safety implementation.

The reward is an exponential centering score, minus action effort, action-change,
target-loss, and invalid-pose penalties. Success for a step means a visible target
within 0.15 normalized camera-plane distance of image center. This is not the
photo booth's image quality or user satisfaction score.

Each episode lasts 10 seconds. Target paths use seeded sinusoidal motion with
varying offset, amplitude, speed, phase, and distance. Additional randomization
varies actuator gain (+/-15%), damping (+/-20%), command delay (0-2 policy steps),
observation noise, and detection dropout. This is a limited dynamics model, not
an identified servo/backlash or camera/perception model. MuJoCo numerical warnings
terminate an episode and receive a penalty.

## Evaluate against baselines

```bash
./scripts/evaluate.sh --checkpoint latest --episodes 20 --seed 10000
./scripts/evaluate.sh --checkpoint /absolute/path/to/run/final.zip --episodes 50
```

PPO, proportional, and zero-action controllers receive identical episode seeds,
target paths, and randomized conditions. Reports contain per-episode values and
mean/std for return, tracking error, centered fraction, visibility, action-change,
invalid poses, numerical failures, and episode length. Use several unseen seed
ranges to assess generalization. Lower error and action-change are better; higher
centered fraction is better. PPO may underperform the baseline, especially early.

`latest` means the most recently modified `runs/*/final.zip`, including smoke
runs. Use an explicit checkpoint to avoid evaluating the wrong experiment.

## Monitor and view

```bash
# Separate Spark terminal:
./scripts/tensorboard.sh
# Another Spark terminal:
./scripts/play.sh --checkpoint latest
# Inspect the proportional controller before training:
./scripts/play.sh --controller proportional
```

On your Mac, keep `./scripts/tunnel.sh` running. Open
[Reachy viewer](http://localhost:8081/) and [TensorBoard](http://localhost:6007/).
Ports are separate from the earlier Microduck experiment. Set `SPARK_HOST` when
needed: `SPARK_HOST=user@host ./scripts/tunnel.sh`. Both servers bind to loopback.
The corresponding Spark services must be running. Stop an existing service/tunnel
before trying to reuse its port.

The browser viewer shows the model geometry and a red synthetic target; it does
not stream camera pixels. Scroll over the 3D scene to zoom and drag to orbit.
Playback runs at approximately real time for up to 100 episodes; Ctrl+C exits.
It can also run without the browser server:

```bash
./scripts/play.sh --checkpoint latest --headless --episodes 2
```

TensorBoard includes `rollout/ep_rew_mean`, `rollout/ep_len_mean`, and
`tracking/{tracking_error,visible,centered,action_change,invalid_pose,numerical_failure}`.
Tracking scalars use a rolling buffer; use evaluation reports for controlled
comparisons. Training or viewing can also run on macOS using CPU; Spark-specific
GPU results require separate verification.

## Development and next experiments

```bash
uv run --extra dev pytest -q
uv run reachy-rl --help
```

Keep `uv.lock` and `model.lock.json` for reproducibility. To intentionally change
the official model release, run `uv run reachy-rl fetch-assets --version VERSION`,
review the changed model lock, and rerun all checks before training. Existing
policies require evaluation after any model, reward, observation, or action change.

Next steps: compare residual RL to the baseline, use recorded detections with
occlusion and delay, integrate the photo-booth detector as an observation adapter,
and implement a separately validated SDK pose-command bridge. The scripts here
never open robot hardware or activate a physical policy.

## Sources

- [NVIDIA Reachy photo-booth overview](https://github.com/NVIDIA/spark-reachy-photo-booth/blob/main/docs/overview.md)
- [Official Reachy Mini SDK and model](https://github.com/pollen-robotics/reachy_mini)
- [Reachy simulation setup](https://github.com/pollen-robotics/reachy_mini/blob/main/docs/source/platforms/simulation/get_started.md)
- [Reachy kinematics implementation](https://github.com/pollen-robotics/reachy_mini/blob/main/src/reachy_mini/kinematics/analytical_kinematics.py)
- [Stable-Baselines3 documentation](https://stable-baselines3.readthedocs.io/)
- [MuJoCo Python documentation](https://mujoco.readthedocs.io/en/stable/python.html)
