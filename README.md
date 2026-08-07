# Behavior Cloning versus Reinforcement Learning for Elevator Group Control under Varying Traffic Density

Reproducible artifact for the manuscript submitted to *Engineering Applications of
Artificial Intelligence*. This repository contains everything needed to reproduce the
training, evaluation, statistics, and figures end-to-end. The manuscript itself is not
distributed with this repository (double-anonymized review).

**Protocol in one line:** compare a 30-minute supervised clone (BC) and a 3-hour
from-scratch PPO policy against classical rule-based dispatch across a calibrated
density spectrum (4x / 8x / 12x the Al-Sharif base arrival rate), on held-out 12-hour
episodes with real passenger group sizes (1–10 pax per call, mean 3.71, calibrated
from a public elevator traffic dataset).

---

## 1. Headline Results

**Transition density 8x** (nine held-out seeds, mean ± SD of 12-hour episode reward;
paired t-test vs. the SD-ETA rule on the same episodes):

| Agent | Reward | Mean wait (s) | p95 wait (s) | Paired t (vs. rule) |
|---|---|---|---|---|
| SD-nearest (distance) | −51.5 ± 22.8 | 34.3 | — | — |
| SD-ETA (teacher rule) | −38.4 ± 30.0 | 32.0 | 77.9 | — |
| **BC** (clone, e20, 30 min) | **−8.5 ± 33.0** | 27.2 | 72.1 | t = 3.52, **p = 0.008** |
| **PPO from scratch** (3 h) | **−3.4 ± 45.2** | 26.1 | 68.3 | t = 2.63, **p = 0.030** |
| PPO fine-tuned (lr 3e-5) | −9.4 ± 30.9 | — | — | — |

BC and PPO are statistically indistinguishable from each other (p = 0.64): a 30-minute
supervised clone matches a 3-hour RL policy at the transition density.

**Density–reward profile** (rule / BC / PPO; 9 seeds at 8x, 3 seeds at 4x and 12x):

| Rates | SD-ETA | BC | PPO |
|---|---|---|---|
| 4x (comfort) | +54.3 ± 1.2 | +54.0 ± 1.2 | +49.0 ± 2.9 (over-intervenes, p = 0.021) |
| 8x (transition) | −38.4 ± 30.0 | **−8.5 ± 33.0** | **−3.4 ± 45.2** |
| 12x (overload) | −1188 ± 154 | −1180 ± 72 | −1231 ± 50 (indistinguishable) |

The common claim that RL's value concentrates at extreme overload is **not** reproduced:
with a correctly calibrated demand model, RL's value sits in the transition band where
the rule is stressed but the return signal is still learnable.

**Policy distillation** (BC of PPO, 30 min): −5.8 ± 40.9 vs. PPO teacher −3.4 ± 45.2,
p = 0.80 — 3 hours of PPO distilled into 30 minutes of supervised cloning without loss.

**Key findings reproduced by this repo:**
1. **Density regime determines everything** — at 4x all policies are near-optimal; at
   8x both learned policies beat the rule (p = 0.008 / 0.030); at 12x all methods
   collapse to indistinguishable large-negative rewards.
2. **Match-reward nonlinearity** — raising the clone's teacher-match from 93.6% to
   99.3% (20 supervised epochs) improves held-out reward ~7x; below ~99% match the
   clone's reward degrades steeply.
3. **Dual critical carriers** — the exact-ETA features and the car-call distribution
   are both necessary (removing either degrades reward ~6x: no-ETA −50.4, p = 0.044;
   no-car-call −47.5, p = 0.042).
4. **Fine-tuning instability** — PPO fine-tuning of an under-trained clone (93.6%
   match) diverges monotonically (value function fits the near-optimal policy,
   advantage degenerates to noise); fine-tuning a well-trained clone is stable but
   offers no gain.
5. **Cross-density robustness** — BC trained at 8x stays robust at 4x and 12x, while
   PPO over-intervenes at 4x (trained at 8x, it keeps "busy" behavior at low density).
6. **20-floor transfer** — the BC pipeline transfers with positive reward at 4x
   (+15.0 ± 14.1).
7. **Teacher choice matters less than protocol** — cloning the SD-ETA rule or cloning
   PPO yields indistinguishable reward (p = 0.61); the bottleneck is clonability, not
   the teacher.

---

## 2. Protocol

**Training — BC** (`scripts/bc_boost.py`):
- 40 teacher episodes (seeds 2000 + i·13, disjoint from all evaluation seeds),
  group event stream at 8x (SCALE = 3.2; 1–10 pax per call, mean 3.71), 12 h each
  (112,664 decision states).
- Chunked cross-entropy over 64-step sequences, Adam lr 3e-4, 50 epochs;
  checkpoint selected at epoch 20 (generalization sweet spot). ~30 minutes on one GPU.

**Training — PPO** (`src/train.py`, `configs/config_gru_shared_event_d8x_noaux.yaml`):
- Shared GRU (2×256), actor + critic heads, no auxiliary objectives.
- Single-passenger decomposition at 8x (dense sequence; BC vs PPO protocol comparison
  shows no significant effect of the decomposition, −10.6 vs −8.5, p = 0.61), 60 epochs,
  8 parallel envs, rollout 32768, reward scale 0.01. ~3 hours on one GPU.
- `configs/config_gru_shared_event_d8x_noaux_group.yaml` is the group-arrival variant
  used to verify the training-distribution question.

**Independent evaluation** (nine held-out seeds 9999, 10001, ..., 10015):
- 12-hour episodes, real group arrivals at 8x (SCALE = 3.2), deterministic greedy,
  GRU hidden state reset per episode.
- Rule baselines: SD-nearest, SD-ETA; zoning baseline: static sectoring (3 zones).
- Service metrics: mean wait, p95 wait, energy (kinematic model).

**Density scan:** SCALE 1.6 = 4x, 3.2 = 8x, 4.8 = 12x.

**Robustness probes (reviewer-driven):**
- `results/wcorr0_9seed.json` — evaluation with the assignment-agreement bonus
  disabled (w_corr = 0): conclusions unchanged (BC −9.8, PPO −4.2, rule −39.7).
- `results/eta_noise_9seed.json` — ±10/20% multiplicative noise on exact-ETA features:
  BC −8.5 → −16.8, PPO −3.4 → +2.1, still far above the noise-free rule (−38.4).
- `results/sectoring_eval.json` — static sectoring collapses (−3,283 at 8x) from
  lobby-zone saturation.
- `results/wait_percentiles_8x.json` — mean/median/p90/p95 wait over 95,566 passengers.

---

## 3. Quick Start

```bash
# 1. Environment (Python >= 3.11)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # or: uv sync

# 2. Train the BC clone (~30 min on an RTX 4060-class GPU)
TRAIN_MODE=group SCALE=3.2 N_DEMO=40 EPOCHS=50 OUT_TAG=bc_boost SEED=42 \
  .venv/bin/python scripts/bc_boost.py

# 3. Train PPO from scratch (~3 h)
.venv/bin/python src/train.py --config configs/config_gru_shared_event_d8x_noaux.yaml \
  --seed 42 --epochs 60 --checkpoint-dir checkpoints/ppo_noaux_s42

# 4. Independent evaluation under real group arrivals (9 held-out seeds)
.venv/bin/python scripts/eval_matrix.py

# 5. Statistics (paired t-tests, exactly the paper numbers)
.venv/bin/python scripts/eval_significance.py

# 6. Figures (IEEE style, PDF vector)
.venv/bin/python scripts/make_figures.py
```

---

## 4. Repository Layout

```
configs/        YAML configs: final PPO recipe (noaux), group variant, ablations, 20F
src/            train.py (PPO loop, _adapt_gen_events 3-mode), env/ (elevator_env,
                n_pax & capacity, exact-kinematics ETA), traffic/ (generator, OD matrix,
                passenger_profile), models/ (gru_ppo GRUSharedActorCritic), zone_map.py
scripts/        bc_boost.py (BC training/eval), eval_matrix.py, eval_significance.py,
                eval_multi_metric.py, eval_finetune_window.py, cmp_actions.py,
                eval_20f_group.py, pool_stats.py, make_figures.py
results/        9-seed JSONs for every experiment (rule, bc, ppo, distillation,
                ablations, density scans, reviewer robustness probes), raw npy legacy
                (marked superseded), rtt_validation.json
figs_ieee/      IEEE-style figure sources (PDF vector + PNG)
LICENSE, README.md, README.zh-CN.md, requirements.txt
```

---

## 5. Key Result Files (`results/`)

| File | Contents |
|---|---|
| `rule_9seed.json` | SD-ETA rule, 9 seeds @ 8x |
| `bc_boost_e20_9seed.json` | BC clone e20, 9 seeds @ 8x (−8.5 ± 33.0) |
| `ppo_noaux_9seed.json` | pure PPO (−3.4 ± 45.2) and PPO+aux (+3.2 ± 46.1) |
| `bc_ppot_9seed.json` | BC-of-PPO distillation (−5.8 ± 40.9) |
| `bc_single_9seed.json` | single-trained BC (−10.6 ± 30.7) — protocol check |
| `bc_density_eval.json` | BC + rule across 4x/8x/12x |
| `ppo_noaux_density3.json` | PPO across 4x/8x/12x |
| `bc_density_train_9seed.json` | training-density sensitivity |
| `bc_nocar_9seed.json` / no-ETA curves | feature ablations |
| `multi_metric_8x.json` / `multi_metric_noaux_8x.json` | wait/energy trade-off |
| `significance_8x.json` | paired t-tests (BC t=3.517, PPO t=2.630) |
| `wcorr0_9seed.json`, `eta_noise_9seed.json` | reviewer robustness probes |
| `sectoring_eval.json`, `wait_percentiles_8x.json` | zoning baseline + service metrics |
| `rtt_validation.json` | flow-level throughput validation vs classical RTT theory |
| `raw_*.npy` | legacy zone-aux era per-episode arrays — **superseded** (see §6) |

---

## 6. Experiment Lineage (what was tried and rejected — all reproducible)

1. **Zone-aware auxiliary head (previous line of work, 2026-07)** — next-event zone
   prediction aux on top of the shared GRU. Superseded: the auxiliary objective
   provides no benefit over pure supervised cloning (BC+aux −30.7 vs pure BC −8.5 at
   epoch 20, n = 9) and the auxiliary head was removed from the manuscript.
   Configs `config_gru_shared_daux_*.yaml` and `raw_zoneaux_*.npy` document this line.
2. **Group training (aligned/unaligned)** — training directly on group calls at
   pax-aligned rates: sparse auxiliary signal; superseded by the current protocol.
3. **Burst expansion** — decomposing group calls into 0.1 s-offset single calls:
   capacity-conservative policy, worse reward.
4. **Old split architecture** (`type: gru`) — separate actor/critic encoders +
   skip-connected dest head reading time features instead of destinations: invalid
   (obs has no destination info); replaced by the shared GRU (`type: gru_shared`).
5. **reward_pred + value_replay auxiliaries** — no benefit, disabled.

All rejected designs remain reproducible from the archived configs and raw arrays.

---

## 7. Citation

```bibtex
@article{song2026bcvpprl,
  title={Behavior cloning versus reinforcement learning for elevator group control
         under varying traffic density},
  author={Song, Chenle},
  journal={Engineering Applications of Artificial Intelligence},
  year={2026},
  note={submitted}
}
```

## 8. License

MIT (see `LICENSE`).
