# Learning to Dispatch Where Rules Collapse: Feature–Auxiliary Synergy for Hall-Call-Only Elevator Group Control

Reproducible artifact for the paper. All training, evaluation, statistics, and figures for the 10-floor main results and the 20-floor generalization analysis.

## Results Summary

At **2.5× peak arrival rates** (the regime where rule-based dispatch collapses), the proposed **Zone-Aux** agent (PPO + shared GRU + car-call distribution encoding + zone destination-prediction auxiliary head):

| Agent | Reward (mean±SE) | Pax served | Wait (s) |
|---|---|---|---|
| SD-nearest | −25,971 ± 3,827 | 1,154 | 52.2 |
| SD-ETA | −32,085 ± 3,014 | 693 | 48.8 |
| GRU+PPO-92 | −5,931 ± 2,178 | — | — |
| GRU+PPO-122 | −6,607 ± 2,151 | — | — |
| Zone-Aux-92 | −14,861 ± 2,097 | — | — |
| **Zone-Aux (proposed)** | **−5,080 ± 1,090** | **2,760** | **46.7** |

**6.3× over the best rule, t(35)=13.05, p<2×10⁻²⁰, Cohen's d=3.12** (3 seeds × 12 held-out episodes). The learned policy Pareto-dominates both heuristics on passengers served and waiting time.

Key findings reproduced by this repo:
1. **RL value is in the rule-collapse regime** — rules win at 1.5–2.0×, collapse at 2.5×+.
2. **Deployment-aware training** — front-6h peak-focused training beats uniform all-day training 5× (idle cost ≈ free vs empty travel −0.1/floor).
3. **Feature–auxiliary synergy** — the car-call distribution encoding is the *necessary carrier* for the zone aux head: without it aux is harmful (−14,861), with it both components synergize (−4,922).
4. **Mixed-density curriculum (negative result)** — matches in-envelope, extrapolates worse at 3.0×.
5. **20-floor transfer** — OD zone prior equally predictable (0.545 vs 0.512 chance 0.333).

## Requirements

- Python ≥ 3.12, PyTorch ≥ 2.12 (CUDA recommended; CPU works, slower)
- `pip install -r requirements.txt` (or `uv sync`)

## Repository Layout

```
configs/     Golden hyperparameter files (10F and 20F, aux/no-aux)
src/         Training, env, traffic generator, models (no hardcoded paths)
scripts/     Reproduce everything: train, evaluate, statistics, figures
results/     All numbers reported in the paper (CSV)
paper/       LaTeX source + compiled PDF
figures/     Generated figure PNGs
```

## Reproduce

```bash
# 0. Install
pip install -r requirements.txt

# 1. Train the proposed agent (3 seeds, ~60 epochs each)
./scripts/train_all.sh configs/config_gru_shared_event_d.yaml 3
#    baseline (no aux): ./scripts/train_all.sh configs/config_gru_noaux.yaml 3

# 2. Evaluate (independent protocol, 12 held-out episodes per seed)
python scripts/eval_independent.py --sd-eta --n-episodes 12
python scripts/eval_independent.py --sd-nearest --n-episodes 12
for s in 42 360 712; do
  python scripts/eval_independent.py \
    --config configs/config_gru_shared_event_d.yaml \
    --ckpt-dir checkpoints/config_gru_shared_event_d_s$s \
    --seed $s --n-episodes 12
done
#    Pool the 36 RL episodes (or run scripts/pool_stats.py) for the t-test.

# 3. OD zone-prior statistics (10F vs 20F; Table VIII)
python scripts/od_prior.py --n-episodes 30

# 4. Figures (matplotlib, from results/*.csv)
python scripts/make_figures.py --out figures
```

## Evaluation Protocol (must match for comparability)

- **Greedy deterministic actions**; per-episode GRU hidden reset (no cross-episode leakage).
- Held-out episode seeds `9999..9999+n-1`; traffic seed-shift `10000+i`; never used for model selection.
- Full 12-hour daily schedule; rates scaled by `--rates-scale` (default 2.5×).
- Models selected by best validation reward with early stopping (patience 6).
- Training-internal validation rewards are *not* comparable across runs (truncated episodes, high variance ±300%) — always use the independent protocol above.

## Configurations

| File | Building | Obs | Aux head | Paper agent |
|---|---|---|---|---|
| `configs/config_gru_shared_event_d.yaml` | 10F | 122-dim (car-call dist) | zone (λ=0.3) | **Zone-Aux (proposed)** |
| `configs/config_gru_noaux.yaml` | 10F | 122-dim | none | GRU+PPO-122 |
| `configs/config_gru_zone_20f.yaml` | 20F | 202-dim | zone height | 20F height |
| `configs/config_gru_fzone_20f.yaml` | 20F | 202-dim | zone functional | 20F functional |
| `configs/config_gru_noaux_20f.yaml` | 20F | 202-dim | none | 20F no-aux |

92-dim ablations: set `obs_car_calls_dist: false` in the env section (env asserts the resulting dimension).

## Notes on GPU Variance

Per-seed rewards vary substantially (−1,868 to −8,451 across seeds 42/360/712), so **single-seed comparisons are unreliable**; use the 3-seed pooled protocol. All evaluations are deterministic given the fixed held-out seeds, so numbers are exactly reproducible on the same checkpoint.

## License

MIT (see LICENSE).
