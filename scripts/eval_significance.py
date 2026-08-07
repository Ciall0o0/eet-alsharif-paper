# eval_significance.py — 9 held-out seeds, BC-e20 vs rule vs PPO best (8x core claim)
import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "src"); sys.path.insert(0, "."); sys.path.insert(0, "/tmp")
import numpy as np, torch
from bc_boost import build_model, gen_events, make_env, run_policy, _eta

dev = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = (9999, 10001, 10003, 10005, 10007, 10009, 10011, 10013, 10015)  # disjoint from BC train seeds (2000+i*13)

def load_ckpt(path):
    ck = torch.load(path, map_location=dev, weights_only=False)
    sd = ck.get("policy_state", ck)
    m = build_model().to(dev); m.load_state_dict(sd, strict=False); m.eval()
    return m

def rule_run(env, events):
    obs, _ = env.reset(options={"events": events})
    env._ep_cap = 300000
    done, steps, tot = False, 0, 0.0
    while not done and steps < 20000:
        if env.pending_calls:
            c = env.pending_calls[0]
            a = min(range(3), key=lambda k: _eta(env, k, c))
        else:
            a = 0
        obs, r, done, trunc, _ = env.step(a)
        tot += r; steps += 1
        if trunc: break
    return tot

def eval_agent(m, scale, label):
    rs = []
    for s in SEEDS:
        ev = gen_events(s, scale); env = make_env()
        rs.append(run_policy(env, ev, m, dev))
    rs = np.array(rs)
    print(f"{label}: {rs.mean():.1f} ± {rs.std(ddof=1):.1f}  per-seed: {[round(x,1) for x in rs]}")
    return rs

def welch_t(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va/len(a) + vb/len(b))
    t = (a.mean() - b.mean()) / se
    # Welch-Satterthwaite df
    df = (va/len(a) + vb/len(b))**2 / ((va/len(a))**2/(len(a)-1) + (vb/len(b))**2/(len(b)-1))
    return t, df

OUT = {}
bc = eval_agent(load_ckpt("checkpoints/bc_boost_e20.pt"), 3.2, "BC-e20@8x")
OUT["bc_e20"] = [float(bc.mean()), float(bc.std(ddof=1))]

rule = eval_agent(None, 3.2, "rule@8x") if False else None
rule_rs = []
for s in SEEDS:
    ev = gen_events(s, 3.2); env = make_env()
    rule_rs.append(rule_run(env, ev))
rule_rs = np.array(rule_rs)
print(f"rule@8x: {rule_rs.mean():.1f} ± {rule_rs.std(ddof=1):.1f}  {[round(x,1) for x in rule_rs]}")
OUT["rule"] = [float(rule_rs.mean()), float(rule_rs.std(ddof=1))]

if os.path.exists("checkpoints/probe_8x_pposcratch_s42/ppo_elevator_best.pt"):
    ppo = eval_agent(load_ckpt("checkpoints/probe_8x_pposcratch_s42/ppo_elevator_best.pt"), 3.2, "PPOscratch-best@8x")
    OUT["ppo_scratch"] = [float(ppo.mean()), float(ppo.std(ddof=1))]
    t, df = welch_t(bc, ppo); print(f"BC vs PPO: t={t:.3f} df={df:.1f} (n={len(bc)})")
    OUT["bc_vs_ppo"] = {"t": t, "df": df}
    t, df = welch_t(bc, rule_rs); print(f"BC vs rule: t={t:.3f} df={df:.1f}")
    OUT["bc_vs_rule"] = {"t": t, "df": df}
    t, df = welch_t(ppo, rule_rs); print(f"PPO vs rule: t={t:.3f} df={df:.1f}")
    OUT["ppo_vs_rule"] = {"t": t, "df": df}

json.dump(OUT, open("results/significance_8x.json", "w"), indent=1)
print("done -> results/significance_8x.json")
