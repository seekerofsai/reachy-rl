# Local validation — November 6, 2026

Validated on macOS, Python 3.12.4, CPU, using the hash-verified Reachy Mini
1.14.0 model, MuJoCo 3.3.0, and Stable-Baselines3 2.9.2.

- All shell entrypoints passed `bash -n`.
- The Gymnasium / SB3 environment checker passed: 12 observations, 3 actions.
- Four physics tests passed: seeded resets and transitions, random actions and
  pose bounds, malformed actions / invalid relative yaw, and tracking improvement
  from the proportional baseline against a stationary robot.
- A PPO smoke run completed 2,048 transitions and wrote TensorBoard logs plus a
  reloadable `final.zip`.
- Spawn-based subprocess vectorization completed a 512-transition run with
  periodic checkpoints; resuming for 128 additional transitions succeeded.
- Evaluation and headless playback reloaded the trained checkpoint successfully.
- The Viser server rendered the Reachy geometry and updated tracking metrics in
  the browser. The final camera/material adjustments were checked separately.

In the three-episode smoke evaluation (seeds 10000–10002), mean camera-plane
tracking error was 0.2934 for the short-trained PPO model, 0.0908 for the
proportional baseline, and 0.1827 for no motion. This is a pipeline check: the
smoke policy did not outperform the baseline. Larger budgets and multiple unseen
seed ranges are needed to evaluate learning.

Linux ARM64 / CUDA dependency resolution is included in `uv.lock`, but execution
on the DGX Spark has not been validated. No physical robot was connected or
controlled. Test models, virtual environments, logs, and downloaded assets are
excluded from the source repository.

## MP4 recording

The recording extension passed seven tests in total, including FFmpeg output
dimensions, encoded duration/frame count, existing-file protection, and cleanup
after an incomplete recording. A real MuJoCo replay using the proportional
controller produced a 640x360 H.264 clip with 50 frames at 25 fps (2.00 seconds).
A second clip verified the final overview framing and visible target marker.
Offscreen rendering was tested on macOS; the Linux EGL path still requires
verification on the DGX Spark. Recording does not capture the Viser browser UI.

## Target-reacquisition experiment

The additive experiment passed 13 tests in total, including the original task and
recording checks. New checks cover the 768-value observation-history interface,
seeded resets, matching current observations/target paths across history lengths,
forced outage timing and held detector values while the target continues moving,
stable recovery timing, explicit failure records, and invalid configurations.

Both history=64 and history=1 PPO smoke runs completed 2,048 transitions. A
comparison against the original tracking smoke policy and proportional baseline
produced reports with identical episode seeds and occlusion schedules. Failed
recoveries retained null latency rather than being omitted. In the three-episode
smoke comparison, the history policy recovered in 0/3 cases and the proportional
baseline in 3/3; the short run does not demonstrate a benefit from observation
history. Larger budgets and multiple training/evaluation seeds remain necessary.

Subprocess-vectorized experiment training, periodic checkpoints, resumed learning,
and headless experiment playback were also checked on local CPU. Spark execution
and hardware transfer remain unverified. No trained model or evaluation result is
included in the source repository.

## Residual RL and photo framing/capture timing

The extension passed 22 tests in total on local macOS CPU. New checks cover exact
zero-residual equivalence to proportional control, correction bounds and applied
action history, photo Gymnasium interfaces and seeded resets, continuous target
pauses, baseline captures versus no-capture timeout, shutter rising edges/cooldown/
attempt budgets, stability-dependent shot validity, and labelled PNG previews.

Both experiments completed 2,048-transition PPO smoke runs, save/reload, evaluation,
and headless playback. Residual comparisons against the original tracking and
reacquisition policies and a five-step delay stress cohort completed. No advantage
over proportional tracking was established by the short residual run.

Photo training uses categorical `MultiDiscrete([3,3,3,2])` actions so deterministic
playback uses the same explicit capture/wait decision learned during training. A
100,000-transition photo validation run (rounded to 100,352 transitions by PPO)
achieved valid first captures in 10/10 episodes at seeds 10000–10009. The heuristic
baseline also succeeded in 10/10. Mean successful capture time was 2.344 seconds
for the learned policy and 2.740 seconds for the baseline. These are small local
validation results, not a robust performance claim; further seeds and training
replicates are required. Photographs and real detections were not used.

Spawn-based subprocess training and resumed learning were checked for both
experiments. The photo browser preview showed the simulated robot, framing diagram,
and live outcome updates. MP4 tests covered residual/photo workflows and regression
of the original tracking recorder; a 640x360 photo clip encoded 75 frames at 25 fps
for exactly 3.00 seconds, including the synthetic framing inset.

The new sources retain simulation-only operation. Execution on DGX Spark/CUDA/EGL,
real-camera framing, and physical hardware transfer remain unverified. Generated
models, videos, capture previews and evaluation reports are not part of the source
changes.
