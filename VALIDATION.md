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
