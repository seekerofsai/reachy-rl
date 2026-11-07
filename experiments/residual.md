# Residual RL for occlusion recovery

PPO learns corrections to the existing proportional controller under exactly the
reacquisition task and reward. This is a separate experiment; the original tracking
and direct-PPO reacquisition models remain usable through their existing scripts.

## Run on the Spark

After syncing the updated source repository:

```bash
cd ~/repos/reachy-rl
uv sync --extra dev
./scripts/check.sh
./scripts/residual.sh smoke
./scripts/residual.sh train --timesteps 500000 --num-envs 8
```

The new direct Pillow dependency supports the photo previews; it is already present
transitively in the previous environment. The lockfile includes the new declaration.
Smoke uses 2,048 transitions to check the pipeline, not establish improvement.
Training defaults to CPU, 64 history frames, the same [occlusion schedule and
recovery criteria](reacquisition.md), and a small 128x128 MLP.

## Action composition

```text
applied_action = clip(proportional_action + scale * clip(PPO_action, -1, 1), -1, 1)
```

`scale` defaults to 0.25 and is restricted to [0, 1]. Actions are normalized head
yaw, pitch, and body-yaw pose rates. The original pose/rate/IK constraints still
apply downstream. The history includes the previous **applied** action, not merely
the residual. The controller does not receive privileged target motion while hidden.

Zero residual exactly reproduces proportional tracking, including observations
and reward on the same seeds. A fresh policy's actor mean is initialized to zero;
training still explores with a Gaussian distribution. This initialization is not
a guarantee that a learned or exploratory correction improves the baseline.

The original reward is unchanged, including motion-effort, smoothness, and invalid
pose penalties on applied control. There is no additional residual-specific reward
that could make comparison to direct PPO misleading.

```bash
# Optional different correction scale or history length for a fresh run:
./scripts/residual.sh train --scale 0.15 --history 64 --timesteps 500000
```

## Evaluate against baselines

```bash
./scripts/residual.sh evaluate --checkpoint latest --episodes 50 \
  --compare-reacquisition latest --compare-tracking latest
```

The proportional baseline is always evaluated. Optional direct-PPO comparison
checkpoints use the plain reacquisition environment, so their outputs are applied
directly rather than accidentally added to proportional control. Their observation
history length is inferred from the checkpoint interface. Base task configurations
must match; explicit paths make experiment selection reproducible.

All compared controllers use the same episode seeds, target paths, randomized
conditions, and detector outages within each evaluation cohort. Reports retain all
failures and include success rate, mean/p95 capped recovery latency, successful-only
latency, tracking error, action-change, and numerical failures. The learned agent
also logs correction magnitude and action-clipping frequency during training.

Test three unseen seed ranges and multiple independent training seeds. Compare
controllers trained for equal transition budgets. A useful result is lower capped
latency/error without more failures or abrupt motion; do not infer an advantage
from a short smoke run.

## Delay robustness

The default command delay is sampled per episode from 0–2 steps (0–80 ms at 25 Hz).
Use a deliberate evaluation override to test a wider range, such as 0–200 ms:

```bash
./scripts/residual.sh evaluate --episodes 50 --seed 20000 --max-delay-steps 5
```

The report records training and evaluation settings separately. This changes the
delay distribution for every controller in that cohort; it does not claim a measured
latency for a physical Reachy. Other base task settings must continue to match when
comparing older direct-PPO policies.

## Monitor, view, record, resume

```bash
./scripts/tensorboard.sh
./scripts/residual.sh play --checkpoint latest
./scripts/residual.sh play --controller baseline
./scripts/residual.sh play --checkpoint latest --headless --episodes 1
./scripts/residual.sh record --checkpoint latest --seconds 10 --output videos/residual.mp4
./scripts/residual.sh train --resume /absolute/path/to/run/final.zip --timesteps 250000
```

Use the existing Mac tunnel and ports 8081/6007. Recording requires FFmpeg and
working MuJoCo EGL on the Spark; the Linux wrapper selects EGL for `record`.
The target marker is hidden during detector occlusion. Recording is simulation-only.

Runs live under `runs/residual/<timestamp>/` with `final.zip`, `settings.json`,
`env.json`, `training.json`, periodic checkpoints, Monitor logs, and TensorBoard
events. Preserve metadata beside the model. `latest` searches this experiment only
and includes smoke runs. Resume restores the saved environment, scale, history,
PPO hyperparameters and optimizer; the transition budget is additional. CLI changes
to those saved settings do not replace them on resume.

## Reference

The algorithmic idea is based on [Residual Reinforcement Learning for Robot
Control](https://arxiv.org/abs/1812.03201). This repository's implementation and
Reachy task are custom adaptations, not a reproduction of that paper's experiments.
