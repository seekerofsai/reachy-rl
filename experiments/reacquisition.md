# Target reacquisition after temporary occlusion

This experiment tests whether observation history helps Reachy anticipate target
motion through a 1–2-second detection outage and recenter it afterward. It extends
the synthetic gaze task, not camera-based person tracking or hardware control.

## Run on the Spark

After copying the updated repository, use the existing setup environment. There
are no new Python dependencies. Run the checks and a short training run first:

```bash
cd ~/repos/reachy-rl
./scripts/check.sh
./scripts/reacquisition.sh smoke
./scripts/reacquisition.sh evaluate --memory latest --episodes 5
```

The smoke run checks model save/reload and the larger observation interface. Its
2,048 transitions are not enough to establish learned recovery performance.

Train the history policy and, optionally, an ablation with no history:

```bash
./scripts/reacquisition.sh train --history 64 --timesteps 500000 --num-envs 8
./scripts/reacquisition.sh train --history 1 --timesteps 500000 --num-envs 8
```

Run these inside tmux if disconnecting SSH. Each command creates its own run;
save the printed checkpoint paths. CPU PPO remains the default. You can reduce
`--num-envs` or select `--vectorization dummy` for a smaller CPU workload.
Training counts are environment transitions and may round up to full PPO rollouts.

## Compare controllers on the same episodes

```bash
./scripts/reacquisition.sh evaluate \
  --memory latest \
  --feedforward latest \
  --tracking /absolute/path/to/original/tracking/run/final.zip \
  --episodes 50 --seed 10000
```

The proportional controller is always included. Omit `--feedforward` if the
history=1 ablation has not been trained; omit `--tracking` if no original tracking
model is available. `--tracking latest` searches the original tracking run folder.
The base `env.json` must match for all compared learned controllers; if it does
not, the comparison stops rather than silently changing action or pose semantics.

Experiment `latest` lookup is separate from the original task. `--memory latest`
and `play --checkpoint latest` select the newest **history=64** experiment model.
`--feedforward latest` selects the newest **history=1** experiment model. Supply an
explicit path for other history lengths or reproducible comparisons. Smoke runs
are eligible for lookup, so an explicit final checkpoint is preferable for reports.

The JSON report is saved beside the history checkpoint by default. It records
the benchmark configuration, checkpoint paths, per-episode seeds and outage times,
outcomes, and summaries. Use `--output PATH` for a different destination.

## Metrics and interpretation

- **Success rate:** fraction of all episodes that recenter and hold the target
  before the recovery deadline. Failures remain in the denominator.
- **Successful latency:** time from the end of the forced occlusion until the
  target has been detected and within 0.15 normalized camera-plane distance for
  at least 0.2 consecutive seconds. This is reported only for successes; it is
  `null` when there are none.
- **Capped latency:** successful latency, or the full timeout for a failed episode.
  This prevents an agent with rare fast successes from appearing best by ignoring
  its failures. Lower is better; compare it together with success rate.
- **Post-occlusion error:** mean true tracking error during the fixed two-second
  recovery window. The report also includes whole-episode error, action-change,
  numerical failures, and whether the target was centered just before the outage.

Each controller receives the same target trajectories, actuator randomization,
outage schedule, and independently seeded sensor noise/dropout sequence. Actions
can change whether the target lies in the camera view, which is part of the task.
Before-outage centering is reported because a controller that never established
tracking is not a strong demonstration of *reacquisition*, even if it later centers.

Repeat the benchmark with multiple unseen seed ranges, for example 10000, 20000,
and 30000, and several independent training seeds. The scripts do not assume that
PPO or the history policy beats the proportional controller.

## What changes in the environment

- One forced detection outage begins between 2 and 5 seconds into each 10-second
  episode and lasts 1–2 seconds, quantized to the 25 Hz policy timestep.
- The synthetic target keeps moving. During the outage, the detection flag is
  zero and camera error holds its last detected value. It exposes no new target
  position, velocity, or future trajectory to the policy. This is a detector mask,
  not a physically rendered occluding object.
- Head/body pose and previous actions remain observable, as they would during a
  camera outage. Additional ordinary detector noise and dropouts still apply.
- `--history 64` concatenates the last 64 causal 12-value observations: 768 inputs
  covering 2.56 seconds. Reset padding repeats the first sample and does not reuse
  observations from the preceding episode. This is finite observation history
  with an MLP, not a recurrent/LSTM policy. `--history 1` has the original 12 inputs.
- The original true-target tracking reward remains available during occlusion as
  training feedback. This is privileged *reward* information, not a policy input.
  It encourages tracking through the outage without adding hidden-state features.
- A recovered episode remains successful if tracking is lost again later; the
  metric measures the first stable recovery within the fixed recovery window.

The original tracking environment and existing 12-input checkpoints are unchanged.
The task is specifically about recentering after sensor loss; it does not establish
general visual search, identity tracking, or transfer to the physical robot.

## Monitor and play

```bash
./scripts/tensorboard.sh
./scripts/reacquisition.sh play --checkpoint latest
# Inspect the baseline without training:
./scripts/reacquisition.sh play --controller proportional
# Run an episode without the browser:
./scripts/reacquisition.sh play --checkpoint latest --headless --episodes 1
```

Use the existing Mac `./scripts/tunnel.sh`; open Viser at port 8081 and TensorBoard
at 6007. Stop any other playback using 8081 first. The red target disappears during
the scheduled detector outage, and the viewer reports the recovery state. The
displayed true error is a diagnostic; the policy does not receive it while hidden.

TensorBoard records `reacquisition/success_rate`, `reacquisition/capped_latency_s`,
tracking error, occluded fraction, detection availability, centered fraction,
action-change, and numerical failures. Success and latency use a rolling buffer
of completed episodes; use the evaluation report for controlled comparisons.

The original `record.sh` currently supports original tracking policies; history
checkpoints must use the experiment playback entrypoint above.

## Checkpoints and resume

Files are stored under `runs/reacquisition/<timestamp>-h64/` (or `-h1/`), including
`final.zip`, periodic checkpoints, `env.json`, `experiment.json`, `training.json`,
Monitor CSVs, and TensorBoard events. Keep all metadata beside checkpoints.

```bash
./scripts/reacquisition.sh train \
  --resume /absolute/path/to/reacquisition/run/final.zip \
  --timesteps 250000
```

The budget is additional. Resume restores the saved history configuration, task,
optimizer, and PPO hyperparameters; CLI history/rollout/epoch changes do not replace
those saved values. An explicit path avoids selecting the wrong ablation.
