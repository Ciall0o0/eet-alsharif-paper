# bc_boost.py — BC strengthening (longer training + more data), per-epoch acc + final eval
# Mirrors bc_diagnostic.py model/data/eval protocol exactly; 50 epochs instead of 15.
# Env-overridable: N_DEMO, EPOCHS, SEED, N_FLOORS, OBS_ETA, N_WORKERS, OUT_TAG, SAVE_EVERY
import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "src"); sys.path.insert(0, ".")
import numpy as np, torch
import torch.nn.functional as F
from src.train import DAILY_SCHEDULE_12H, _adapt_gen_events
from src.traffic.generator import TrafficGenerator
from src.env.elevator_env import ElevatorEnv
from src.models.gru_ppo import GRUSharedActorCritic

RATES_25 = {"up_peak": 7.5/3.71, "down_peak": 7.5/3.71, "lunch_peak": 5.0/3.71,
            "interfloor": 4.0/3.71, "off_peak": 1.5/3.71}
MAX_EVENTS = 20000
SEQ = 64
NF = int(os.environ.get("N_FLOORS", "10"))
OBS_ETA = os.environ.get("OBS_ETA", "1") == "1"
CAR_DIST = True

def make_env(n_floors=NF, obs_eta=OBS_ETA):
    return ElevatorEnv(config={"num_floors": n_floors, "num_elevators": 3, "max_load_kg": 900,
                               "max_total_time": 43200.0, "max_dt": 30.0, "reward_scale": 0.01,
                               "obs_car_calls_dist": CAR_DIST, "obs_eta": obs_eta,
                               "door_open_time": 3.0, "door_close_time": 3.0,
                               "boarding_time_per_pax": 0.8, "alighting_time_per_pax": 0.6,
                               "passenger_delivered": 2.0, "wait_time_per_sec": -0.05,
                               "empty_distance_per_floor": -0.1, "energy_per_start_stop": -0.05,
                               "idle_penalty_per_sec": -0.005,
                               "assignment_proximity": -0.05, "assignment_direction_align": 0.02,
                               "assignment_load_balance": -0.03,
                               "assignment_estimated_wait": -0.01, "assignment_correct": 0.3})

def _eta(env, k, call):
    el = env.elevators[k]
    tt = el.travel_time_for_distance(abs(el.current_floor - call["floor"]))
    if el.state == "moving":
        tt += (el.travel_time_for_distance(abs(el.target_floor - el.current_floor))
               + el.door_open_time + el.door_close_time)
    return tt

def gen_events(seed, scale=3.2, n_floors=NF):
    rates = {k: v * scale for k, v in RATES_25.items()}
    g = TrafficGenerator(n_floors=n_floors, seed=seed, schedule=DAILY_SCHEDULE_12H, arrival_rates=rates)
    raw = g.generate_episode_multi_segment(n_segments=1, seed_shift=seed + 7,
                                           schedule=DAILY_SCHEDULE_12H, max_events=MAX_EVENTS)[0]
    return _adapt_gen_events(np.array(raw))

def collect_demo(env, events, max_steps=40000):
    obs, _ = env.reset(options={"events": events})
    env._ep_cap = 300000
    obs_list, act_list = [], []
    done, steps = False, 0
    while not done and steps < max_steps:
        if env.pending_calls:
            c = env.pending_calls[0]
            a = min(range(env.num_elevators), key=lambda k: _eta(env, k, c))
            obs_list.append(obs.copy()); act_list.append(a)
        else:
            a = 0
        obs, r, done, trunc, _ = env.step(a)
        steps += 1
        if trunc:
            break
    return np.array(obs_list), np.array(act_list)

def run_policy(env, events, model, dev="cpu"):
    obs, _ = env.reset(options={"events": events})
    env._ep_cap = 300000
    h = model.get_initial_hidden(1, dev)
    done, steps, tot = False, 0, 0.0
    while not done and steps < 20000:
        if env.pending_calls:
            ot = torch.as_tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                a, _, _, h, _ = model.get_action(ot, h, deterministic=True)
            a = int(a.item())
        else:
            a = 0
        obs, r, done, trunc, _ = env.step(a)
        tot += r; steps += 1
        if trunc:
            break
    return tot

def build_model(state_dim=128):
    return GRUSharedActorCritic(state_dim=state_dim, action_dim=3, aux_prediction=True,
                                num_dest_classes=10, use_layer_norm=True,
                                gru_hidden=256, gru_layers=2, gru_dropout=0.1,
                                actor_hidden=64, critic_hidden=64,
                                dest_head_on=False, event_head_on=True,
                                reward_change_on=False)

def eval_seeds(model, dev="cpu", seeds=(9999, 10001, 10003), scale=3.2, n_floors=NF):
    rs = []
    for s in seeds:
        ev = gen_events(s, scale, n_floors)
        env = make_env(n_floors)
        tot = run_policy(env, ev, model, dev)
        rs.append(tot)
        print(f"  eval seed {s}: {tot:.1f}", flush=True)
    return float(np.mean(rs)), float(np.std(rs))

if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = int(os.environ.get("SEED", "42"))
    torch.manual_seed(SEED); np.random.seed(SEED)
    OUT_TAG = os.environ.get("OUT_TAG", "bc_boost")

    # --- collect demos in parallel (pure-CPU env simulation) ---
    n_demo = int(os.environ.get("N_DEMO", "40"))
    import multiprocessing as mp
    n_workers = min(int(os.environ.get("N_WORKERS", "8")), mp.cpu_count())

    def _collect_one(seed):
        ev = gen_events(seed, 3.2)
        env = make_env()
        return collect_demo(env, ev)

    demo_seeds = [2000 + i * 13 for i in range(n_demo)]  # disjoint from eval seeds 9999+
    demos = []
    with mp.Pool(n_workers) as pool:
        for i, (obs_arr, act_arr) in enumerate(pool.imap_unordered(_collect_one, demo_seeds)):
            demos.append((obs_arr, act_arr))
            if (i + 1) % 10 == 0:
                print(f"collected {i+1}/{n_demo} demos (last {len(obs_arr)} decisions)", flush=True)
    print(f"total demos: {n_demo}, decisions: {sum(len(o) for o, _ in demos)}, obs_dim={demos[0][0].shape[1]}", flush=True)

    # state_dim inferred from actual obs (10F-eta=128, 10F-noeta=122, 20F=202, ...)
    state_dim = demos[0][0].shape[1]
    model = build_model(state_dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    # --- BC training: seq-64 chunked CE, 50 epochs ---
    n_epochs = int(os.environ.get("EPOCHS", "50"))
    curve = []
    for epoch in range(n_epochs):
        model.train()
        total_loss, total_n = 0.0, 0
        for obs_arr, act_arr in demos:
            n = len(obs_arr) - (len(obs_arr) % SEQ)
            for s in range(0, n, SEQ):
                obs_t = torch.as_tensor(obs_arr[s:s+SEQ], dtype=torch.float32, device=dev).unsqueeze(0)
                act_t = torch.as_tensor(act_arr[s:s+SEQ], dtype=torch.int64, device=dev)
                hidden = model.get_initial_hidden(1, dev)
                out, _, _, _, _, _ = model.forward(obs_t, hidden)
                logits = out.squeeze(0)
                loss = F.cross_entropy(logits, act_t)
                opt.zero_grad(); loss.backward(); opt.step()
                total_loss += loss.item(); total_n += SEQ
        model.eval()
        acc_ok, acc_n = 0, 0
        with torch.no_grad():
            for obs_arr, act_arr in demos:
                n = len(obs_arr) - (len(obs_arr) % SEQ)
                obs_t = torch.as_tensor(obs_arr[:n], dtype=torch.float32, device=dev).unsqueeze(0)
                out, _, _, _, _, _ = model.forward(obs_t, model.get_initial_hidden(1, dev))
                pred = out.squeeze(0).argmax(-1)
                acc_ok += (pred == torch.as_tensor(act_arr[:n], device=dev)).sum().item()
                acc_n += n
        acc = acc_ok / acc_n
        curve.append(round(acc, 4))
        if (epoch + 1) % 5 == 0 or epoch in (0, n_epochs - 1):
            print(f"BC ep{epoch+1}: loss={total_loss/max(total_n,1):.4f} acc={acc:.4f}", flush=True)
        if (epoch + 1) % 10 == 0:
            torch.save({"policy_state": model.state_dict(), "aux_prediction": True,
                        "event_head_on": True}, f"checkpoints/{OUT_TAG}_e{epoch+1}.pt")
            print(f"  saved {OUT_TAG}_e{epoch+1}.pt", flush=True)

    print(f"=== held-out eval (seed 9999/10001/10003) ===", flush=True)
    me, se = eval_seeds(model, dev)
    print(f"{OUT_TAG}-{n_epochs}ep mean: {me:.1f} ± {se:.1f}", flush=True)

    json.dump({"acc_curve": curve, "eval3": [me, se], "obs_dim": state_dim, "n_demo": n_demo,
               "seed": SEED, "n_floors": NF},
              open(f"results/{OUT_TAG}_curve.json", "w"), indent=1)
    print(f"done -> results/{OUT_TAG}_curve.json", flush=True)
