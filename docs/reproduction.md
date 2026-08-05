# Reproduction Manual — commands, math, and caveats

This file complements `README.md` with the exact command sequences, the density/alignment
math, the statistics conventions, and known environment sensitivities that affect
byte-for-byte reproduction.

## 1. Full command sequence (one shot)

```bash
# ---- setup ----
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# ---- train proposed agent, 3 seeds (each ~1.2 h on RTX 4060 8 GB) ----
for s in 42 360 712; do
  python src/train.py --config configs/config_gru_shared_daux_s360.yaml \
    --seed $s --epochs 60 --checkpoint-dir checkpoints/zoneaux_s$s \
    > logs/zoneaux_s$s.log 2>&1 &
done; wait

# ---- train comparison agents (seed 42 each; 3 seeds for Zone-Aux-92 per paper) ----
python src/train.py --config configs/config_gru_noaux.yaml --seed 42 \
  --epochs 60 --checkpoint-dir checkpoints/noaux92_s42 > logs/noaux92.log 2>&1
python src/train.py --config configs/config_gru_shared_event_d.yaml --seed 42 \
  --epochs 60 --checkpoint-dir checkpoints/noaux122_s42 > logs/noaux122.log 2>&1
# Zone-Aux-92 ablation: remove car-call distribution channels (obs 122→92), keep aux head
python src/train.py --config configs/config_gru_shared_daux_s360.yaml \
  --obs-carcall-dist 0 --seed 42 --epochs 60 \
  --checkpoint-dir checkpoints/eventaux92_s42 > logs/eventaux92.log 2>&1

# ---- independent evaluation (12 ep/seed, fixed held-out seeds) ----
for s in 42 360 712; do
  python scripts/eval_group_independent.py --ckpt checkpoints/zoneaux_s$s \
    --tag zoneaux_s$s --out results/raw_zoneaux_s$s.npy
done
python scripts/eval_group_independent.py --sd-eta --out results/raw_sd_eta.npy
python scripts/eval_group_independent.py --sd-nearest --out results/raw_sd_nearest.npy
# ... same for noaux92 / noaux122 / eventaux92 / mix / 20F checkpoints

# ---- statistics & figures ----
python scripts/pool_stats.py
python scripts/od_prior.py
python scripts/make_figures.py
```

## 2. Density / alignment math (why 2.5× is the paper's regime)

- Real dataset (12-floor, one day): ~21 hall calls/h × 3.6 pax = ~77 pax/h average.
- Our generator at 1×: ~109 calls/h ≈ 109 pax/h (single-person simplification:
  1 call = 1 pax). So **pax density at 1× ≈ 1.4× real; call density ≈ 5× real**
  (each real group call is split into ~3.6 single calls).
- Training at 2×: 219 pax/h (dense, aux learnable).
- Evaluation: rates ÷ 3.71 (mean group size) then × 2.5 → 292 pax/h with real groups.
  This keeps the *pax* load at 2.5× while restoring realistic call packaging.
- Physical capacity: 3 cars × ~40 pax/h × 12 h ≈ 1,440 pax — 2.5× group evaluation
  (3,504 pax demand) is a genuine stress test, not a 8.2× overload (that only happens
  in the rejected unaligned-group training).

## 3. Statistics conventions (must match the paper)

- **Unit**: one per-episode reward; 12 episodes per seed; 3 seeds pooled → n = 36.
- **Test**: Welch's unequal-variance two-sample t-test (`scipy.stats.ttest_ind`,
  `equal_var=False`), proposed vs. each baseline.
- **Effect size** (`scripts/pool_stats.py`):
  `d = (m1 − m2) / sqrt((var1 + var2) / 2)` with population variance (`ddof=0`) —
  this matches the paper's d = 2.02 (vs SD-ETA) and d = 2.15 (vs SD-nearest).
- **Pseudo-replication pitfall**: never concatenate a single SD run 3× to "match" the
  n = 36 pooled RL runs — that inflates t to ~13 and p to 1e-20. The SD rows in the
  tables are n = 12 (one run × 12 episodes); the pooled test is RL n = 36 vs SD n = 12.

## 4. `_adapt_gen_events` three modes (the protocol's core)

`src/train.py::_adapt_gen_events(gen_events, train_mode)`:

| mode | col 3 (n_pax) | effect | used for |
|---|---|---|---|
| `single` | 1.0 | every call = 1 pax; dense sequence | training & validation |
| `group` | gen_events col 5 (real group size) | every call carries its group | deployment evaluation |
| `burst` | expand call into group_size events at 0.1 s offsets | dense + capacity pressure | rejected (§7 README) |

Config keys (both must be set consistently):
`traffic.train_train_mode` (default `single`) and `traffic.val_train_mode`
(default `single`; the rejected group/burst experiments used `group`/`burst`).

## 5. Known environment sensitivities

- **GPU**: 3 parallel trainings run at ~69 s/epoch on RTX 4060 8 GB; **4 parallel
  crushes to 8–10 min/epoch** (memory swap) — run 3 at a time.
- **Seeds**: augmentation seeds differ between `config_gru_shared_daux_s360.yaml` and
  `_s712.yaml` (both reproduce the same architecture and protocol; seed 42 uses the
  `--seed 42` CLI override).
- **Resume**: `CosineAnnealingLR` restarts from peak LR on resume — any run that was
  resumed carries artifacts; the paper numbers come from fresh from-scratch runs only.
- **Determinism**: eval uses greedy deterministic action selection; GRU hidden state is
  reset at episode start; held-out episode seeds are fixed (9999 + i).
- **Reward scale**: env applies `reward_scale: 0.01`; paper rewards are in scaled units.
