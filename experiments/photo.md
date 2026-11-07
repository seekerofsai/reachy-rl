# Synthetic photo framing and capture timing

Reachy adjusts its head/body orientation to place a synthetic face box at a desired
image composition and learns when to press a virtual shutter. This is a simulation
experiment using the real Reachy geometry and eye-camera projection. There is no
face detector, real camera connection, identity recognition, or image generation.
Saved PNGs are labelled **synthetic face-box diagrams**, not camera photographs.

## Run on the Spark

```bash
cd ~/repos/reachy-rl
uv sync --extra dev
./scripts/check.sh
./scripts/photo.sh smoke
./scripts/photo.sh train --timesteps 500000 --num-envs 8
./scripts/photo.sh evaluate --checkpoint latest --episodes 50 --save-captures
```

Use tmux for long runs. CPU PPO is the default. A smoke run checks the pipeline;
its policy may never trigger a valid capture. More training is needed to evaluate
framing and shutter behavior against the heuristic baseline.

## Task and observations

Each episode lasts up to 10 seconds. One synthetic face moves along a seeded
trajectory, with a continuous-position pause beginning between 2 and 5 seconds
and lasting 1.5–2.5 seconds. Face dimensions vary by +/-15%. The composition goal
is chosen from center, left third, or right third. The policy sees the desired
composition, not the future pause timing or target trajectory.

The 24 normalized/clipped inputs are:

| Indices | Features |
|---|---|
| 0–3 | Last detected face-box center x/y and width/height |
| 4 | Detection flag |
| 5–6 | Desired composition center |
| 7–9 | Commanded head yaw/pitch/body yaw |
| 10–12 | Measured head yaw/pitch/body yaw |
| 13–15 | Previous applied motion action |
| 16 | Previous shutter gate |
| 17–18 | Filtered image velocity estimated from past detections |
| 19 | Observed stable-duration fraction |
| 20 | Episode time remaining |
| 21 | Capture attempts remaining |
| 22 | Shutter cooldown remaining |
| 23 | Head angular speed estimated from measured pose changes |

Detector noise/dropout applies as in the tracking task. A missing detection holds
the last box and sets the detection flag to zero. Image velocity is causal and
filtered; it is not the true target velocity. The true geometry is used to score
captures and training rewards, not exposed as additional policy features.

Control is yaw/pitch/body rotation with the existing conservative rate and IK
constraints. Face size changes with projection and subject distance; there is no
zoom or camera-translation action in this version.

## Four actions and virtual shutter

The action space is `MultiDiscrete([3, 3, 3, 2])`. Each of the first three indices
selects a negative, zero, or positive correction (`0 -> -1`, `1 -> 0`, `2 -> +1`),
scaled by 0.25 before adding it to proportional framing. The baseline aims at the
requested composition, rather than always centering the face. The fourth index
explicitly chooses wait (0) or capture (1), mapped internally to a shutter gate.

- A rising gate at or above 0.5 requests a capture.
- Captures become eligible after 0.5 simulated seconds.
- A one-second cooldown separates accepted attempts.
- Holding the gate high does not repeatedly fire; lower it before another request.
- An episode allows at most three attempts and ends on a valid shot, exhaustion of
  that budget, a numerical failure, or the time limit.

Applied pose rates remain bounded to [-1, 1]. A fresh categorical policy prefers
zero correction and waiting but explores other choices. The learned categorical
mode is also used for deterministic playback, avoiding the mismatch of thresholding
a continuous shutter mean. Premature captures can still occur during training.

## Shot validity and reward

A shot is valid only when the face is fully visible, its normalized center is
within 0.12 of the desired composition, image speed is below 0.20 normalized units
per second, and head yaw/pitch speed is below 0.45 radians per second. Those true
conditions must hold for at least 0.30 consecutive seconds before capture.

Dense shaping rewards improvement in composition; small penalties apply to time,
motion effort, action changes, and invalid poses. A valid capture receives +10
plus a small bonus for remaining episode time. An invalid attempt receives -2;
ending without a valid shot adds -5. This balances framing, waiting for stability,
and avoiding wasted shots. It is not a human image-quality or aesthetic rating.

The comparison baseline uses proportional framing plus a rule-based shutter that
waits for 0.40 seconds of observed stable framing. Observed and true stability can
differ under detection noise, which is why the baseline is evaluated rather than
assumed perfect.

## Evaluate

```bash
./scripts/photo.sh evaluate --episodes 50 --seed 10000 --save-captures
./scripts/photo.sh evaluate --episodes 50 --seed 20000
# Optional command-delay stress cohort:
./scripts/photo.sh evaluate --episodes 50 --max-delay-steps 5
```

Learned and heuristic controllers receive identical seeded subject motion, pause
times, face dimensions, composition goals, and randomized conditions. Reports include
every episode and shot attempt, success rate, first-shot validity, mean attempts,
successful-only capture time, and capped capture time (10 seconds for a failed
episode). A controller that never fires has zero success and a full failure time;
it does not win by avoiding bad shots. Tracking error, action-change and numerical
failure metrics are also included. Use multiple training seeds and held-out episodes.

The JSON report is saved beside the checkpoint. `--save-captures` writes PNG diagrams
for **all** accepted shutter attempts, including invalid ones, in a matching
`evaluation-...-captures/<controller>/` directory. Shot validity and measured quality
are recorded in the JSON; a saved diagram alone does not prove a shot is valid.

## Preview and record

```bash
./scripts/photo.sh play --checkpoint latest --save-captures runs/photo-preview-captures
./scripts/photo.sh play --controller baseline
./scripts/photo.sh play --checkpoint latest --headless --episodes 1
./scripts/photo.sh record --checkpoint latest --seconds 10 --output videos/photo.mp4
```

The browser at port 8081 shows the robot, a synthetic framing diagram with a
composition crosshair, and capture outcomes. The existing SSH tunnel and TensorBoard
script work unchanged. Choose a different port or stop existing playback first.
MP4 recording adds the synthetic diagram as an inset; FFmpeg and working EGL are
required on headless Spark. Successful captures can end episodes early, so a clip
may include several episode resets to fill its requested simulated duration.

Runs are isolated under `runs/photo/<timestamp>/`. Keep `settings.json`, `env.json`
and `training.json` beside the model. `latest` includes smoke runs; use an explicit
checkpoint for reproducible reports. Resume with an additional budget:

```bash
./scripts/photo.sh train --resume /absolute/path/to/run/final.zip --timesteps 250000
```

Resume restores saved task settings, reward parameters, optimizer and PPO settings.
The next real-world integration would replace synthetic boxes with detector output
and add a separately validated SDK/camera capture adapter. These scripts operate
only on simulation and never activate physical robot motors or a real shutter.
