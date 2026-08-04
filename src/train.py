"""PPO Training script with on-the-fly traffic generation.

Replaces static .eet NPZ dataset loading with a Poisson-process
TrafficGenerator that generates fresh episodes every epoch for training
and a fixed validation set covering all five traffic modes.

Usage
-----
    python -m src.train                     # default config/config.yaml
    python -m src.train --config config/cloudmax.yaml
    python -m src.train --config config/smoke.yaml --epochs 1
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

_PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJ_ROOT))

from src.env.elevator_env import ElevatorEnv
from src.models.lstm_ppo import PPOTrainer
from src.runner import MultiEnvRunner
from src.traffic.generator import TrafficGenerator, DAILY_SCHEDULE_12H
from src.utils import get_device, set_seed

try:
    import swanlab
    HAS_SWANLAB = True
except ImportError:
    HAS_SWANLAB = False


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _adapt_gen_events(gen_events: np.ndarray) -> np.ndarray:
    """Adapt TrafficGenerator output for ElevatorEnv event consumption.

    ElevatorEnv._process_events expects:
        [floor(0), dest_floor(1), n_pax(2), start_time(3), ...]

    TrafficGenerator._pack produces:
        [origin(0), dest(1), event_time(2), patience(3), mass(4),
         group_size(5), jitter(6), routing(7), delta(8), 10.0(9)]

    Remap: col 2 = 1 (one passenger per row), col 3 = gen col 2 (event_time).
    """
    if gen_events.size == 0:
        return gen_events
    n = gen_events.shape[0]
    ncols = max(10, gen_events.shape[1])
    out = np.zeros((n, ncols), dtype=np.float32)
    out[:, 0] = gen_events[:, 0]
    out[:, 1] = gen_events[:, 1]
    out[:, 2] = gen_events[:, 2]  # event_time -> col 2 (ElevatorEnv convention)
    out[:, 3] = 1.0               # n_pax
    for c in range(4, min(gen_events.shape[1], ncols)):
        out[:, c] = gen_events[:, c]
    return out


# ---------------------------------------------------------------------------
# entropy annealing
# ---------------------------------------------------------------------------

def cosine_anneal(epoch: int, total: int, start: float, end: float,
                  floor: float | None = None) -> float:
    """Cosine schedule: start -> end over total epochs."""
    if total <= 0:
        return end
    progress = min(epoch / max(total, 1), 1.0)
    coef = 0.5 * (1.0 + math.cos(math.pi * progress))
    value = end + (start - end) * coef
    if floor is not None:
        value = max(value, floor)
    return value


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def validate(
    trainer: PPOTrainer,
    runner: MultiEnvRunner,
    val_items: list,
) -> dict:
    """Run validation pass over fixed validation episodes.

    Parameters
    ----------
    val_items : list[tuple[str, np.ndarray]]
        Each tuple is (mode_name, events_array) with events_array
        a (N, 10) float32 array from the traffic generator.
    """
    if not val_items:
        return {"val/avg_reward": 0.0, "val/n_episodes": 0}

    # NaN guard: skip validation if model parameters are corrupted
    for name, p in trainer.policy.named_parameters():
        if torch.isnan(p).any():
            print(f"\n[NAN DETECTED in {name}] Skipping validation — model corrupted", flush=True)
            return {"val/avg_reward": float("-inf"), "val/n_episodes": 0, "nan_detected": True}

    total_reward = 0.0
    total_steps = 0
    n_episodes = 0

    # Only use env 0 for sequential validation
    val_env_idx = 0

    for mode_name, events_arr in val_items:
        if events_arr.size == 0 or events_arr.shape[0] == 0:
            continue

        adapted = _adapt_gen_events(events_arr)
        runner.reset_env(val_env_idx, adapted, trainer.policy)

        episode_reward = 0.0
        episode_steps = 0
        max_val_steps = 8000  # full 12 h day ~4400 steps; was 4000 (truncated tail)

        for _ in range(max_val_steps):
            if runner.done[val_env_idx]:
                break
            rew, steps, ndone = runner.step_all(
                trainer.policy, deterministic=True,
            )
            episode_reward += rew
            episode_steps += steps
            if ndone > 0:
                break

        total_reward += episode_reward
        total_steps += episode_steps
        n_episodes += 1

    avg_reward = total_reward / max(n_episodes, 1)
    avg_steps = total_steps / max(n_episodes, 1)

    return {
        "val/avg_reward": avg_reward,
        "val/avg_steps": avg_steps,
        "val/n_episodes": n_episodes,
    }


# ---------------------------------------------------------------------------
# auxiliary-prediction metrics
# ---------------------------------------------------------------------------

def _floor_to_zone(floor: int, mode: str = "height", n_floors: int = 10) -> int:
    """Map destination floor to zone label (delegates to src.zone_map)."""
    from src.zone_map import zone_label
    return zone_label(floor, mode=mode, n_floors=n_floors)


def _compute_aux_metrics(
    trainer: PPOTrainer,
    val_items: list,
    device: torch.device,
    env_cfg: dict | None = None,
) -> dict:
    """Compute dest_pred accuracy@1 and CE loss using real env rollouts."""
    if not trainer.aux_prediction:
        return {}

    policy = trainer.policy
    policy.eval()

    # Create env for real rollouts
    env = ElevatorEnv(config=env_cfg)
    _zm = getattr(env, "zone_mode", "height")
    _nf = int(getattr(env, "max_floor", 10))

    total_correct_1 = 0
    total_samples = 0
    total_ce = 0.0

    with torch.no_grad():
        for _mode_name, events_arr in val_items:
            adapted = _adapt_gen_events(events_arr)
            if adapted.shape[0] == 0:
                continue

            # Reset env with real events
            obs, info = env.reset(options={"events": adapted})

            obs_buf: list[np.ndarray] = []
            dest_buf: list[int] = []

            max_steps = 2000
            for _ in range(max_steps):
                dest_label = info.get("active_dest", -1)
                if dest_label > 0:
                    dest_label = _floor_to_zone(dest_label, mode=_zm, n_floors=_nf)

                obs_buf.append(obs)
                dest_buf.append(dest_label)

                # Deterministic action
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                obs_t = obs_t.unsqueeze(0).unsqueeze(0)
                action, _lp, _val, _hid, _dest = policy.get_action(obs_t, deterministic=True)
                a = action.item()

                obs, _reward, done, _trunc, info = env.step(a)

                if done or _trunc:
                    dest_label = info.get("active_dest", -1)
                    if dest_label > 0:
                        dest_label = _floor_to_zone(dest_label, mode=_zm, n_floors=_nf)
                    obs_buf.append(obs)
                    dest_buf.append(dest_label)
                    break

            if len(obs_buf) < 2:
                continue

            # Stack into (1, T, dim)
            obs_arr = np.stack(obs_buf)
            obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=device).unsqueeze(0)

            dest_labels = torch.as_tensor(dest_buf, dtype=torch.int64, device=device)

            # Run through policy
            hidden = policy.get_initial_hidden(1, device)
            if hasattr(policy, "event_head"):
                _, _, _, dest_logits, _, _ = policy.forward(obs_t, hidden)
            else:
                _, _, _, dest_logits = policy.forward(obs_t, hidden)

            if dest_logits is None:
                continue

            num_cls = dest_logits.size(-1)
            preds = dest_logits.squeeze(0)

            valid = (dest_labels >= 0) & (dest_labels < num_cls)
            if not valid.any():
                continue

            # Accuracy@1
            _unused, top1 = preds.topk(1, dim=-1)
            acc1 = (top1[valid].squeeze(-1) == dest_labels[valid]).float().sum().item()
            total_correct_1 += acc1
            total_samples += valid.sum().item()

            # CE loss
            ce = torch.nn.functional.cross_entropy(preds, dest_labels, ignore_index=-1)
            total_ce += ce.item() * len(obs_buf)

    policy.train()

    if total_samples == 0:
        return {}

    return {
        "aux/dest_pred_acc1": total_correct_1 / total_samples,
        "aux/dest_ce_loss": total_ce / total_samples,
    }
# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def _save_training_plot(trainer: PPOTrainer, out_dir: Path, epoch: int):
    """Save a training-progress text summary."""
    stats = trainer.stats
    if not stats:
        return
    summary_path = out_dir / f"stats_e{epoch:04d}.txt"
    lines = [f"epoch={epoch}"] + [
        f"  {k}={v}" for k, v in sorted(stats.items())
    ]
    summary_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PPO Elevator Training")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no-swanlab", action="store_true")
    parser.add_argument("--swanlab-cloud", action="store_true")
    parser.add_argument("--swanlab-project", default="elevator-ppo")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    # ---- Config ------------------------------------------------------------
    config_path = args.config or str(_PROJ_ROOT / "config" / "config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    env_cfg = cfg.get("env", {})
    model_cfg = cfg.get("model", {})
    ppo_cfg = cfg.get("ppo", {})
    training_cfg = cfg.get("training", {})
    traffic_cfg = cfg.get("traffic", {})
    aux_cfg = cfg.get("aux_prediction", {})

    num_floors = env_cfg.get("num_floors", 10)
    num_elevators = env_cfg.get("num_elevators", 2)
    num_envs = training_cfg.get("num_envs", 4)
    total_epochs = args.epochs or training_cfg.get("total_epochs", 100)
    device_str = get_device(args.device)
    device = torch.device(device_str)

    set_seed(args.seed)

    # ---- SwanLab -----------------------------------------------------------
    use_swanlab = HAS_SWANLAB and not args.no_swanlab
    use_cloud = args.swanlab_cloud
    project_name = args.swanlab_project
    if use_swanlab:
        swanlab.init(
            experiment_name=f"{model_cfg.get('type', 'lstm')}_f{num_floors}_e{num_elevators}_s{args.seed}",
            mode="cloud" if use_cloud else None, 
            project=project_name,
            config={
                "num_floors": num_floors,
                "num_elevators": num_elevators,
                "num_envs": num_envs,
                "total_epochs": total_epochs,
                "model_type": model_cfg.get("type", "lstm"),
                "seed": args.seed,
                "aux_enabled": aux_cfg.get("enabled", False),
                "aux_lambda": aux_cfg.get("lambda", 0.0),
                "config_file": args.config,
            },
        )

    # ---- Environment template ----------------------------------------------
    env_cfg = dict(cfg.get("env", {}))
    env_cfg["traffic"] = cfg.get("traffic", {})
    env_template = ElevatorEnv(config=env_cfg)
    state_dim = env_template.observation_space.shape[0]
    action_dim = env_template.action_space.n

    # ---- Traffic generator -------------------------------------------------
    traffic_seed = traffic_cfg.get("traffic_seed", traffic_cfg.get("seed", 42))
    gen = TrafficGenerator(
        n_floors=num_floors,
        entrance_floor=traffic_cfg.get("entrance_floor", 1),
        seed=traffic_seed,
        arrival_rates=traffic_cfg.get("arrival_rates"),
    )

    # Fixed validation set: list of (mode_name, events_array) tuples
    # NOTE: max_events must match the TRAINING episode density (500), otherwise
    # train/val distributions diverge and the comparison is polluted by
    # distribution mismatch (observed: noaux wins on low-density val, unreal
    # wins on high-density train distribution).
    val_per_mode = training_cfg.get("val_per_mode", 3)
    val_max_events = traffic_cfg.get("val_max_events", 700)
    # B2: validation uses the SAME 12h mixed schedule as training (fixed
    # seeds) so train/val distributions agree structurally. Per-mode pure
    # evaluation lives in scripts/eval_per_mode.py (auxiliary report).
    val_items: list = []
    for i in range(val_per_mode * 5):
        ep = gen.generate_episode_multi_segment(
            n_segments=1, seed_shift=10000 + i, schedule=DAILY_SCHEDULE_12H,
            max_events=val_max_events,
        )[0]
        if ep.size > 0 and ep.shape[0] > 0:
            val_items.append(("daily", ep))

    # ---- Trainer -----------------------------------------------------------
    trainer = PPOTrainer.from_config(
        state_dim=state_dim,
        action_dim=action_dim,
        cfg=cfg,
        device=device_str,
    )
    if args.compile:
        try:
            trainer.policy = torch.compile(trainer.policy)
        except Exception:
            pass

    # ---- LR scheduler ------------------------------------------------------
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.actor_optimizer,
        T_max=total_epochs,
        eta_min=ppo_cfg.get("lr_min", 1e-6),
    )
    trainer.scheduler = scheduler

    # ---- Runner ------------------------------------------------------------
    aug = ppo_cfg.get("use_augmentation", False)
    runner = MultiEnvRunner(
        env_template=env_template,
        num_envs=num_envs,
        device=device,
        augment=aug,
        seed=args.seed,
        hidden_is_tuple=trainer.hidden_is_tuple,
        # aux_prediction is now in policy.dest_head (removed from runner)
    )

    # ---- Resume ------------------------------------------------------------
    start_epoch = 0
    best_val_reward = -float("inf")
    best_val_stats = {}
    patience_counter = 0
    early_stop_patience = training_cfg.get("early_stop_patience", 20)

    if args.resume:
        start_epoch = trainer.load(args.resume) + 1
        best_val_reward = trainer.stats.get("best_val_reward", -float("inf"))
        print(f"[resume] Restarting from epoch {start_epoch}")

    # ---- Checkpoint dir ----------------------------------------------------
    cfg_save = training_cfg.get("save_dir", "")
    checkpoint_dir = Path(args.checkpoint_dir or cfg_save or _PROJ_ROOT / "checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ---- Entropy annealing -------------------------------------------------
    ent_start = ppo_cfg.get("entropy_coef_start", 0.05)
    ent_end = ppo_cfg.get("entropy_coef_end", 0.001)
    ent_floor = ppo_cfg.get("entropy_floor", 0.0005)
    ent_total = training_cfg.get("entropy_anneal_epochs", total_epochs)

    # ===================================================================
    # Training loop
    # ===================================================================
    epoch_bar = tqdm(
        range(start_epoch, total_epochs), desc="epochs",
        initial=start_epoch, total=total_epochs,
    )

    for epoch in epoch_bar:
        epoch_start_time = time.time()

        # -- Entropy annealing -----------------------------------------------
        ent_coef = cosine_anneal(
            epoch, ent_total, ent_start, ent_end, ent_floor,
        )
        trainer.set_entropy_coef(ent_coef)

        # -- Generate fresh training episodes --------------------------------
        # B2: full 12 h working day (incl. off_peak) so training/validation/
        # deployment distributions agree. The model learns real mode
        # transitions instead of static pure-mode scenarios.
        max_ev_train = traffic_cfg.get("max_events_per_episode", 1400)
        mix = training_cfg.get("density_mix", False)
        if mix:
            # Domain randomization: each episode samples a random density scale
            # (relative to configured arrival_rates) -> single model adapts to
            # multiple densities. e.g. density_range [0.75, 1.5] with x2.0 cfg
            # -> actual x1.5..x3.0.
            dlo, dhi = training_cfg.get("density_range", [0.75, 1.5])
            base_rates = dict(traffic_cfg.get("arrival_rates", {}))
            import numpy as _np
            _mix_rng = _np.random.default_rng(epoch * 1000 + 7)
            train_episodes = []
            for _j in range(num_envs * 5):
                sc = float(_mix_rng.uniform(dlo, dhi))
                rates = {k: v * sc for k, v in base_rates.items()}
                g2 = TrafficGenerator(
                    n_floors=gen.n_floors if hasattr(gen, "n_floors") else 10,
                    seed=int(_mix_rng.integers(0, 2**31)),
                    schedule=DAILY_SCHEDULE_12H,
                    arrival_rates=rates,
                )
                ep2 = g2.generate_episode_multi_segment(
                    n_segments=1, seed_shift=epoch * 1000 + _j,
                    schedule=DAILY_SCHEDULE_12H, max_events=max_ev_train,
                )[0]
                if ep2.size > 0 and ep2.shape[0] > 0:
                    train_episodes.append(ep2)
        else:
            train_episodes = gen.generate_episode_multi_segment(
                n_segments=num_envs * 5,
                seed_shift=epoch * 1000,
                schedule=DAILY_SCHEDULE_12H,
                max_events=max_ev_train,
            )
        train_ptr = 0

        # Feed initial episodes
        for i in range(num_envs):
            if train_ptr < len(train_episodes):
                adapted = _adapt_gen_events(train_episodes[train_ptr])
                runner.reset_env(i, adapted, trainer.policy)
                train_ptr += 1
            else:
                runner.done[i] = True

        # -- Episode rollout -------------------------------------------------
        epoch_total_reward = 0.0
        epoch_total_steps = 0
        rollout_steps = ppo_cfg.get("rollout_steps", 2048)
        steps_collected = 0

        while steps_collected < rollout_steps:
            # Re-feed finished envs
            for i in range(num_envs):
                if runner.done[i] and train_ptr < len(train_episodes):
                    adapted = _adapt_gen_events(train_episodes[train_ptr])
                    runner.reset_env(i, adapted, trainer.policy)
                    train_ptr += 1

            if runner.all_done:
                break

            reward, steps, ndone = runner.step_all(
                trainer.policy, buffer=trainer.buffer,
            )
            epoch_total_reward += reward
            epoch_total_steps += steps
            steps_collected += steps

        # -- PPO update ------------------------------------------------------
        # Populate UNREAL replay buffer from rollout data
        if trainer.aux_prediction and trainer.replay_buffer is not None:
            total_steps = trainer.buffer.size()
            for i in range(total_steps):
                trainer.replay_buffer.add(
                    obs=trainer.buffer.obs[i],
                    reward=trainer.buffer.rewards[i].item(),
                )
        
        last_obs_list = runner.get_last_obs_per_env()
        update_stats = trainer.update(last_obs_per_env=last_obs_list)

        # -- LR scheduler ----------------------------------------------------
        scheduler.step()

        # -- Validation ------------------------------------------------------
        val_stats = {}
        val_freq = training_cfg.get("val_freq", 5)
        if val_freq > 0 and (
            epoch % val_freq == 0 or epoch == total_epochs - 1
        ):
            val_stats = validate(trainer, runner, val_items)

        # -- Aux prediction metrics ------------------------------------------
        aux_stats = {}
        if trainer.aux_prediction:
            aux_stats = _compute_aux_metrics(trainer, val_items, device, env_cfg)

        # -- Logging ---------------------------------------------------------
        epoch_time = time.time() - epoch_start_time
        current_lr = scheduler.get_last_lr()[0]

        log_dict = {
            "epoch": epoch,
            "entropy_coef": ent_coef,
            "learning_rate": current_lr,
            "epoch_time_s": epoch_time,
            "train/reward": (
                epoch_total_reward / max(epoch_total_steps, 1)
            ),
            "train/steps": epoch_total_steps,
        }
        if update_stats:
            for k, v in update_stats.items():
                log_dict[f"train/{k}"] = v
        if val_stats:
            log_dict.update(val_stats)
        if aux_stats:
            log_dict.update(aux_stats)

        # Console progress
        desc_parts = [f"e{epoch}"]
        if update_stats:
            desc_parts.append(
                f"pl={update_stats.get('policy_loss', 0):.3f}"
            )
            desc_parts.append(
                f"vl={update_stats.get('value_loss', 0):.3f}"
            )
            desc_parts.append(f"ent={ent_coef:.4f}")
        if val_stats:
            desc_parts.append(
                f"vr={val_stats.get('val/avg_reward', 0):.1f}"
            )
        epoch_bar.set_description(" ".join(desc_parts))

        # SwanLab
        if use_swanlab:
            swanlab.log(log_dict, step=epoch)
        trainer.current_epoch = epoch


        # -- Best model tracking ---------------------------------------------
        val_metric = val_stats.get("val/avg_reward", -float("inf"))
        if val_metric > best_val_reward:
            best_val_reward = val_metric
            best_val_stats = dict(val_stats)
            patience_counter = 0
            best_path = checkpoint_dir / "ppo_elevator_best.pt"
            trainer.save(str(best_path))
            if use_swanlab:
                swanlab.log({"best_val_reward": best_val_reward}, step=epoch)
        elif val_stats:
            patience_counter += 1

        # -- Early stopping --------------------------------------------------
        if patience_counter >= early_stop_patience:
            print(
                f"\n[early_stop] No improvement for {early_stop_patience} "
                f"validations (best={best_val_reward:.2f}), stopping."
            )
            break

        # -- Plotting --------------------------------------------------------
        plot_freq = training_cfg.get("plot_freq", 50)
        if plot_freq > 0 and epoch % plot_freq == 0:
            _save_training_plot(trainer, checkpoint_dir, epoch)

    # ---- Cleanup -----------------------------------------------------------
    runner.close()

    # Save ablation-friendly summary JSON
    summary_path = checkpoint_dir / "ablation_summary.json"
    import json as _json
    summary = {
        "best_val_reward": best_val_reward,
        "total_epochs": epoch + 1,
        "aux_enabled": aux_cfg.get("enabled", False),
        "aux_lambda": aux_cfg.get("lambda", 0.0),
        "seed": args.seed,
        "config_file": args.config,
        "num_floors": num_floors,
        "num_elevators": num_elevators,
        "best_val_stats": best_val_stats,
    }
    try:
        summary_path.write_text(_json.dumps(summary, indent=2))
        print(f"[summary] Saved to {summary_path}")
    except Exception as e:
        print(f"[summary] Failed to save: {e}")

    if use_swanlab:
        swanlab.finish()

    print(f"\nTraining complete. Best val reward: {best_val_reward:.2f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()
