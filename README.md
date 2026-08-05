# Learning to Dispatch Where Rules Collapse: Zone-Aware GRU–RL Elevator Group Control via Single-Call Decomposition Training

Reproducible artifact for the paper. This repository contains everything needed to
reproduce the training, evaluation, statistics, and figures end-to-end. The paper
manuscripts themselves are not distributed with this repository.

**Protocol in one line:** train PPO on *single-passenger hall calls* (dense sequence,
learnable auxiliary signal), evaluate on *real passenger group sizes* (1–10 pax per
call, mean 3.71, calibrated from a real elevator traffic dataset).

---

## 1. Headline Results

At **2.5× peak arrival rates, passenger-count-aligned** (the regime where rule-based
dispatch collapses), 12-hour episodes, real group sizes, 3 seeds × 12 held-out episodes
per agent (pooled n = 36):

| Agent | Reward (mean ± SE) | Pax served | Wait (s) |
|---|---|---|---|
| SD-nearest (rule) | −3,099.8 ± 444.1 | 445 | 17.1 |
| SD-ETA (rule) | −2,417.7 ± 325.6 | 831 | 17.0 |
| GRU+PPO-92 (no aux) | −1,429.5 ± 219.7 | 641 | 17.9 |
| GRU+PPO-122 (no aux) | −1,389.9 ± 295.1 | 505 | 17.9 |
| Zone-Aux-92 (aux w/o car-call dist) | −2,153.5 ± 173.8 | 870 | 20.0 |
| MIX | −1,056.7 ± 173.2 | 580 | 17.7 |
| **Zone-Aux (proposed)** | **−792.9 ± 59.8** | ~495 | 17.0 |

**Pooled statistics vs. baselines** (Welch t-test, Cohen's d as in `scripts/pool_stats.py`):

| Comparison | t | p | Cohen's d |
|---|---|---|---|
| vs SD-ETA | 4.908 | 3.84×10⁻⁴ | 2.02 |
| vs SD-nearest | 5.148 | 2.85×10⁻⁴ | 2.15 |
| vs GRU+PPO-92 | 2.80 | 0.016 | 1.11 |

**20-floor generalization** (trained on 10F, evaluated on 20F, n = 36):

| Agent | Reward (mean ± SE) | vs SD-ETA |
|---|---|---|
| **Zone-Aux (20F)** | **−1,322.9 ± 122.0** | t=3.54, p=0.0037, d=1.40 |
| SD-ETA (20F) | −2,895.4 ± 427.1 | — |
| SD-nearest (20F) | −3,494.2 ± 403.2 | t=5.15, p=1.8×10⁻⁴, d=2.15 |

**Key findings reproduced by this repo:**
1. **RL value lives in the rule-collapse regime** — rules are competitive at 1.5–2.0×,
   collapse at 2.5×+ (the density region where the paper's evaluation is run).
2. **Single-call decomposition training** — decomposing each real group call into
   single-passenger calls keeps the training sequence dense (auxiliary signal learnable,
   event accuracy 0.58→0.73) while being exactly passenger-count-equivalent to the
   deployment distribution. Training on real group calls directly is infeasible
   (sparse auxiliary signal ≈ 0.5 accuracy; misaligned rates overload the system 8.2×).
3. **Feature–auxiliary synergy** — the car-call distribution encoding is the *necessary
   carrier* for the zone next-event auxiliary head: with 92-dim obs the aux head is
   harmful (−2,154), with 122-dim obs (car-call distribution) both components synergize
   (−793).
4. **20-floor transfer** — the zone OD prior stays predictable (0.545 vs 0.333 chance),
   and the trained agent generalizes with a significant margin over rules.
5. **Negative results (documented, not paper claims)** — burst-expansion training
   (0.1 s offsets), group training (aligned & unaligned), mixed-density curricula,
   and the old split-architecture (type: `gru` with skip-connected dest head) all
   underperform the final recipe. See §7.

---

## 2. The Final Protocol (why these numbers)

**Training** (`train_train_mode: single`):
- 12 h daily schedule, 7 segments (up_peak/interfloor/lunch/interfloor/down_peak/off_peak/interfloor),
  arrival rates ×2.0 → ~219 calls/h.
- Each call carries 1 passenger (`_adapt_gen_events` col 3 = 1.0). This is the *single-call
  decomposition* of the real group distribution: 219 pax/h is exactly the pax rate of the
  deployment scenario (rate ÷ mean group size 3.71, then re-multiplied by group members).
- PPO (GRU shared encoder, 256×2, seq_len 32), 8 parallel envs, rollout 32 768,
  60 epochs, early stop patience 6, reward scale 0.01 (12 h × ~1400 events),
  `normalize_rewards: false`, `normalize_advantage: true`.
- Auxiliary head: **next-event zone prediction** (3 zones), λ = 0.3, label from
  `info["next_event_zone"]` — *not* destination prediction (hall-call-only setting:
  destinations are physically unavailable at decision time).

**Validation** (`val_train_mode: single` by default):
- Same single-call decomposition, 1 400-event cap, fixed held-out seeds, every 5 epochs.

**Independent evaluation** (`scripts/eval_group_independent.py`):
- Real group sizes 1–10 (weights `[0.157, 0.130, 0.163, 0.233, 0.143, 0.110, 0.027, 0.023, 0.010, 0.003]`,
  mean 3.71 — calibrated from `elevator_traffic_dataset.csv`), arrival rates ÷3.71 then ×2.5
  → 292 pax/h.
- 12 held-out episodes × 3 seeds, deterministic greedy policy, GRU hidden state reset per episode.
- Rule baselines: SD-nearest (shortest-distance) and SD-ETA (shortest expected arrival time).

**Density check:** x2.5 aligned = 292 pax/h vs. physical capacity ~1 440 pax/12 h (3 cars)
— a hard stress test, but *not* the 8.2× overload of unaligned group training.

---

## 3. Quick Start

```bash
# 1. Environment (Python ≥ 3.12)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # or: uv sync

# 2. Train the proposed agent (3 seeds, ~1.2 h each on an RTX 4060-class GPU)
python src/train.py --config configs/config_gru_shared_daux_s360.yaml --seed 42 \
  --epochs 60 --checkpoint-dir checkpoints/zoneaux_s42 --swanlab-project eet-paper
python src/train.py --config configs/config_gru_shared_daux_s360.yaml --seed 360 \
  --epochs 60 --checkpoint-dir checkpoints/zoneaux_s360 --swanlab-project eet-paper
python src/train.py --config configs/config_gru_shared_daux_s360.yaml --seed 712 \
  --epochs 60 --checkpoint-dir checkpoints/zoneaux_s712 --swanlab-project eet-paper
# (config_gru_shared_daux_s360.yaml and _s712.yaml differ only by seed-dependent
#  augmentation seeds; either reproduces the same architecture & protocol)

# 3. Independent evaluation under real group sizes
python scripts/eval_group_independent.py --ckpt checkpoints/zoneaux_s42 --tag zoneaux_s42
python scripts/eval_group_independent.py --ckpt checkpoints/zoneaux_s360 --tag zoneaux_s360
python scripts/eval_group_independent.py --ckpt checkpoints/zoneaux_s712 --tag zoneaux_s712
python scripts/eval_group_independent.py --sd-eta
python scripts/eval_group_independent.py --sd-nearest

# 4. Statistics (Welch t + Cohen's d — exactly the paper numbers)
python scripts/pool_stats.py --npy results --proposed zoneaux_s42_main zoneaux_s360 zoneaux_s712 --sd raw_sd_eta.npy
python scripts/pool_stats.py --npy results --proposed zoneaux_s42_main zoneaux_s360 zoneaux_s712 --sd raw_sd_nearest.npy

# 5. Figures (IEEE style, white background, PDF vector)
python scripts/make_figures.py

```

---

## 4. Full Reproduction Steps (paper numbers, step by step)

### 4.1 Generate the training set
`src/train.py` builds episodes on the fly from `src/traffic/` (DAILY_SCHEDULE_12H,
arrival rates, OD matrix, `_adapt_gen_events` with `train_mode`). No external data
files are required for training.

### 4.2 Train the 5 agents in the comparison table
| Agent | Config | Notes |
|---|---|---|
| Zone-Aux (proposed) | `configs/config_gru_shared_daux_s360.yaml` | 122-dim obs + zone aux, λ=0.3 |
| GRU+PPO-92 | `configs/config_gru_noaux.yaml` | 92-dim, no aux head |
| GRU+PPO-122 | `configs/config_gru_shared_event_d.yaml` | 122-dim, no aux head |
| Zone-Aux-92 | remove car-call distribution from obs (see §4.4) | ablation |
| MIX | train with aux λ=0.3 + reward-pred/val-replay off, obs 122 | ablated recipe |

Run each with seeds 42(The Universe Final Answer), 360(Circle), 712(Luo Tianyi's birthday), each 60 epochs.

### 4.3 Evaluate everything under group sizes
`scripts/eval_group_independent.py` runs the fixed protocol (12 ep/seed, seeds 9999+i,
GRU reset, deterministic). Outputs per-episode rewards; save them as
`results/raw_<agent>.npy` (12 values per agent).

### 4.4 Regenerate all result tables
```bash
python scripts/pool_stats.py          # per-seed + pooled Welch t, p, Cohen's d
python scripts/od_prior.py            # zone OD priors (10F & 20F)
```
Tables 1–3 in the papers are produced by these scripts from `results/raw_*.npy`.
The synergy ablation removes the car-call distribution channels from the observation
(obs 122→92) while keeping the aux head.

### 4.5 Reproduce every figure
`scripts/make_figures.py` produces the IEEE-style figures (white background, Okabe-Ito
palette, serif, PDF vector): training curves, density profile, main results, synergy,
OD prior. The figure files under `paper_access/figs/` and `paper_csmag/figs/` are the
paper-embedded versions.

### 4.6 (paper manuscripts)
The paper manuscripts (IEEE conference / IEEE Access / IEEE Intelligent Systems
versions) are not distributed with this repository; they compile with **0 errors,
0 undefined references** from their own LaTeX sources.

---

## 5. Repository Layout

```
configs/        YAML configs: final recipe, ablations, 20F, rejected designs
src/            train.py (PPO loop, _adapt_gen_events 3-mode), env/ (elevator_env with
                n_pax & capacity), traffic/ (generator, OD matrix, passenger_profile),
                models/ (gru_ppo GRUActorCritic), runner.py, zone_map.py
scripts/        eval_group_independent.py (final protocol), eval_group_matrix.py,
                eval_20f_group.py, eval_independent.py, eval_per_mode.py,
                train_all.sh, od_prior.py, pool_stats.py, make_figures.py
results/        raw per-episode npy (16 agents/scenarios), main_results.csv,
                main_results_perseed.csv, synergy_ablation.csv, group_eval_matrix.csv,
                adaptive_raw.csv, coverage/density/od-prior tables
figs_ieee/      IEEE-style figure sources (PDF vector)
lib/            shared plotting/persistence helpers (no hard-coded paths)
LICENSE, README.md, README.zh-CN.md, requirements.txt
```

---

## 6. Raw Data Reference (`results/`)

| File | Contents |
|---|---|
| `raw_zoneaux_s42_main.npy` | proposed agent, seed 42 (12 eps) |
| `raw_zoneaux_s360.npy` / `raw_zoneaux_s712.npy` | proposed agent, seeds 360/712 |
| `raw_noaux92_s42.npy` / `raw_noaux122_s42.npy` | no-aux baselines, seed 42 |
| `raw_eventaux92_s42.npy` | zone aux on 92-dim obs (harmful ablation) |
| `raw_mixaux122_s42.npy` | MIX recipe |
| `raw_sd_eta.npy` / `raw_sd_nearest.npy` | rule baselines |
| `raw_zone20f_s{42,360,712}.npy` | 20F generalization |
| `raw_20f_sd_eta.npy` / `raw_20f_sd_nearest.npy` | 20F rules |
| `raw_group_aligned_s42.npy` / `raw_group_unaligned_s42.npy` | rejected group training |
| `raw_burst2_s42.npy` | rejected burst-expansion training |
| `raw_zoneaux_s42_legacy.npy` | old split-architecture (invalidated) |
| `group_eval_matrix.csv` | every model's reward/pax/wait in one table |
| `adaptive_raw.csv` | MIX ×3.0 per-episode (negative result) |

---

## 7. Experiment Lineage (what was tried and rejected — all reproducible)

1. **Group training (aligned)** — train directly on real group calls at pax-aligned
   rates: auxiliary signal too sparse (event_acc ≈ 0.51 ≈ prior) → −1,363, worse than
   single-call training (−793).
2. **Group training (unaligned)** — rates not divided by group size: 866 pax/h = 8.2×
   overload → do-nothing local optimum → −4,285.
3. **Burst expansion** — expand each group call into same-floor same-destination single
   calls at 0.1 s offsets: keeps sequence density (event_acc 0.64, learnable) but the
   policy learns capacity-conservative behavior (pax 145 vs 581 served) → −1,543.
4. **Mixed-density curriculum** — in-envelope match, worse extrapolation at 3.0×.
5. **Old split architecture** (`type: gru` in older configs) — separate actor/critic
   encoders + skip-connected dest head reading time features instead of destinations
   (obs 89-dim has no destination info) → invalidated; replaced by the shared GRU +
   event-head (`type: gru_shared`).
6. **reward_pred + value_replay auxiliaries** — turned OFF (λ=0.0) in the final recipe
   (no benefit, extra cost).

All of these can be reproduced with the configs in `configs/`
(`config_gru_shared_daux_group.yaml`, `config_gru_shared_daux_burst.yaml`,
`config_gru_shared_zone_hd.yaml`, ...) and the raw results are archived above.

---

## 8. Citation

```bibtex
@article{song2026zoneaware,
  title={Learning to Dispatch Where Rules Collapse: Zone-Aware GRU--RL Elevator Group
         Control via Single-Call Decomposition Training},
  author={Song, Chenle},
  journal={IEEE Access},
  year={2026}
}
```

## 9. License

MIT (see `LICENSE`).
