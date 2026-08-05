# 在规则失效处学会调度：基于单呼梯分解训练的 Zone-Aware GRU–RL 电梯群控

本仓库是论文《Learning to Dispatch Where Rules Collapse: Zone-Aware GRU–RL Elevator
Group Control via Single-Call Decomposition Training》的完整可复现工件。包含全部代码、
配置、逐幕原始结果、统计检验与图件，可端到端复现。论文稿件本身不随本仓库分发。

**协议一句话总结：** 训练时用*单人呼梯*（序列密集、辅助信号可学），评估时用*真实乘客组*
（每组 1–10 人、均值 3.71、由真实电梯客流数据集校准）。

---

## 1. 核心结果

评估条件：**2.5× 峰值到达率、按人次对齐**（规则基线崩溃的密度区间）、12 小时 episode、
真实组规模、每 agent 3 种子 × 12 个独立 held-out episode（pooled n = 36）：

| Agent | Reward (均值 ± SE) | 服务乘客数 | 等待 (s) |
|---|---|---|---|
| SD-nearest（规则） | −3,099.8 ± 444.1 | 445 | 17.1 |
| SD-ETA（规则） | −2,417.7 ± 325.6 | 831 | 17.0 |
| GRU+PPO-92（无 aux） | −1,429.5 ± 219.7 | 641 | 17.9 |
| GRU+PPO-122（无 aux） | −1,389.9 ± 295.1 | 505 | 17.9 |
| Zone-Aux-92（aux 无轿厢分布） | −2,153.5 ± 173.8 | 870 | 20.0 |
| MIX | −1,056.7 ± 173.2 | 580 | 17.7 |
| **Zone-Aux（本文方法）** | **−792.9 ± 59.8** | ~495 | 17.0 |

**Pooled 统计检验**（Welch t 检验，Cohen's d 口径同 `scripts/pool_stats.py`）：

| 对比 | t | p | Cohen's d |
|---|---|---|---|
| vs SD-ETA | 4.908 | 3.84×10⁻⁴ | 2.02 |
| vs SD-nearest | 5.148 | 2.85×10⁻⁴ | 2.15 |
| vs GRU+PPO-92 | 2.80 | 0.016 | 1.11 |

**20 层泛化**（10F 训练、20F 评估，n = 36）：

| Agent | Reward (均值 ± SE) | vs SD-ETA |
|---|---|---|
| **Zone-Aux（20F）** | **−1,322.9 ± 122.0** | t=3.54, p=0.0037, d=1.40 |
| SD-ETA（20F） | −2,895.4 ± 427.1 | — |
| SD-nearest（20F） | −3,494.2 ± 403.2 | t=5.15, p=1.8×10⁻⁴, d=2.15 |

**本仓库可复现的关键发现：**
1. **RL 的价值在规则崩溃区**——1.5–2.0× 密度下规则尚有竞争力，2.5×+ 全面崩溃
   （本文评估即在此区间）。
2. **单呼梯分解训练**——把真实组呼梯分解为单人呼梯，保持训练序列密集（辅助信号可学，
   event_acc 0.58→0.73），且在人次上与部署分布严格等价。直接按真实组训练不可行
   （辅助信号稀疏 ≈ 0.5 先验；率不对齐则系统 8.2× 超载）。
3. **特征–辅助协同**——轿厢呼叫分布编码是 zone 辅助头的*必要载体*：92 维观测下
   aux 有害（−2,154），122 维（含轿厢分布）下两者协同（−793）。
4. **20 层迁移**——zone OD 先验仍可预测（0.545 vs 0.333 随机），训练好的 agent
   以显著优势胜过规则。
5. **已记录的负结果**（论文不作为主张）——burst 展开训练、group 训练（对齐/未对齐）、
   混合密度课程、旧分离架构（type: `gru` 带 skip dest head）均不如最终配方。见 §7。

---

## 2. 最终协议（数字为什么是这样）

**训练**（`train_train_mode: single`）：
- 12 小时日运行时刻表，7 个时段（早高峰/层间/午高峰/层间/晚高峰/闲时/层间），
  到达率 ×2.0 → 约 219 呼梯/小时。
- 每个呼梯载 1 人（`_adapt_gen_events` 第 3 列 = 1.0）——即真实组分布的*单呼梯分解*：
  219 pax/h 与部署场景的人次速率严格相等（率 ÷ 平均组规模 3.71 后再乘回组员数）。
- PPO（共享 GRU 编码器 256×2、seq_len 32）、8 并行环境、rollout 32,768、60 epochs、
  early stop patience 6、reward scale 0.01（12 h × ~1400 事件）、
  `normalize_rewards: false`、`normalize_advantage: true`。
- 辅助头：**下一事件 zone 预测**（3 类），λ = 0.3，标签来自 `info["next_event_zone"]`
  ——*不是*目的地预测（纯呼梯设定：决策时刻物理上拿不到目的地信息）。

**验证**（`val_train_mode: single`，默认）：
- 同样单呼梯分解，1400 事件上限，固定 held-out 种子，每 5 epochs 一次。

**独立评估**（`scripts/eval_group_independent.py`）：
- 真实组规模 1–10（权重 `[0.157, 0.130, 0.163, 0.233, 0.143, 0.110, 0.027, 0.023, 0.010, 0.003]`，
  均值 3.71——由 `elevator_traffic_dataset.csv` 校准），到达率 ÷3.71 后 ×2.5 → 292 pax/h。
- 每种子 12 个 held-out episode、确定性 greedy 策略、每 episode 重置 GRU 隐状态。
- 规则基线：SD-nearest（最短距离）、SD-ETA（最短期望到达时间）。

**密度核对**：x2.5 对齐 = 292 pax/h vs 物理容量 ~1,440 pax/12 h（3 车）——是硬压测，
但*不是*未对齐 group 训练的 8.2× 超载。

---

## 3. 快速开始

```bash
# 1. 环境（Python ≥ 3.12）
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # 或 uv sync

# 2. 训练本文方法（3 种子，RTX 4060 级 GPU 每种子约 1.2 小时）
python src/train.py --config configs/config_gru_shared_daux_s360.yaml --seed 42 \
  --epochs 60 --checkpoint-dir checkpoints/zoneaux_s42 --swanlab-project eet-paper
python src/train.py --config configs/config_gru_shared_daux_s360.yaml --seed 360 \
  --epochs 60 --checkpoint-dir checkpoints/zoneaux_s360 --swanlab-project eet-paper
python src/train.py --config configs/config_gru_shared_daux_s360.yaml --seed 712 \
  --epochs 60 --checkpoint-dir checkpoints/zoneaux_s712 --swanlab-project eet-paper
# （config_gru_shared_daux_s360.yaml 与 _s712.yaml 仅增广种子不同，架构与协议一致）

# 3. 真实组规模独立评估
python scripts/eval_group_independent.py --ckpt checkpoints/zoneaux_s42 --tag zoneaux_s42
python scripts/eval_group_independent.py --ckpt checkpoints/zoneaux_s360 --tag zoneaux_s360
python scripts/eval_group_independent.py --ckpt checkpoints/zoneaux_s712 --tag zoneaux_s712
python scripts/eval_group_independent.py --sd-eta
python scripts/eval_group_independent.py --sd-nearest

# 4. 统计检验（Welch t + Cohen's d——与论文数字完全一致）
python scripts/pool_stats.py --npy results --proposed zoneaux_s42_main zoneaux_s360 zoneaux_s712 --sd raw_sd_eta.npy
python scripts/pool_stats.py --npy results --proposed zoneaux_s42_main zoneaux_s360 zoneaux_s712 --sd raw_sd_nearest.npy

# 5. 出图（IEEE 风格、白底、PDF 矢量）
python scripts/make_figures.py

```

---

## 4. 完整复现步骤（逐步复现论文数字）

### 4.1 生成训练集
`src/train.py` 实时从 `src/traffic/`（DAILY_SCHEDULE_12H、到达率、OD 矩阵、
`_adapt_gen_events` 三模式）构建 episode——训练无需任何外部数据文件。

### 4.2 训练对比表中的 5 个 agent
| Agent | 配置 | 说明 |
|---|---|---|
| Zone-Aux（本文方法） | `configs/config_gru_shared_daux_s360.yaml` | 122 维观测 + zone aux, λ=0.3 |
| GRU+PPO-92 | `configs/config_gru_noaux.yaml` | 92 维，无辅助头 |
| GRU+PPO-122 | `configs/config_gru_shared_event_d.yaml` | 122 维，无辅助头 |
| Zone-Aux-92 | 从观测中移除轿厢分布通道（见 §4.4） | 消融 |
| MIX | 122 维 obs + aux λ=0.3、关闭 reward-pred/val-replay | 消融配方 |

每个用种子 42/360/712 各跑一次（60 epochs，early stop patience 6）。

### 4.3 统一在组规模下评估
`scripts/eval_group_independent.py` 执行固定协议（每种子 12 ep、种子 9999+i、
GRU 重置、确定性策略）。输出逐幕 reward；保存为 `results/raw_<agent>.npy`
（每 agent 12 个值）。

### 4.4 重新生成全部结果表
```bash
python scripts/pool_stats.py          # per-seed + pooled Welch t、p、Cohen's d
python scripts/od_prior.py            # zone OD 先验（10F 与 20F）
```
论文表 1–3 均由这些脚本从 `results/raw_*.npy` 生成。协同消融 = 从观测中移除
轿厢呼叫分布通道（obs 122→92）但保留辅助头。

### 4.5 复现每张图
`scripts/make_figures.py` 产出 IEEE 风格图（白底、Okabe-Ito 配色、衬线字体、
PDF 矢量）：训练曲线、密度剖面、主结果、协同、OD 先验。
`paper_access/figs/` 与 `paper_csmag/figs/` 下为论文嵌入版。

### 4.6 论文稿件
论文稿件（IEEE 会议版 / IEEE Access / IEEE Intelligent Systems 三版）不随本仓库分发；
其 LaTeX 源码各自编译均为 **0 错误、0 undefined 引用**。

---

## 5. 仓库结构

```
configs/        YAML 配置：最终配方、消融、20F、被否决的设计
src/            train.py（PPO 主循环、_adapt_gen_events 三模式）、env/（elevator_env
                含 n_pax 与容量）、traffic/（生成器、OD 矩阵、passenger_profile）、
                models/（gru_ppo GRUActorCritic）、runner.py、zone_map.py
scripts/        eval_group_independent.py（最终协议）、eval_group_matrix.py、
                eval_20f_group.py、eval_independent.py、eval_per_mode.py、
                train_all.sh、od_prior.py、pool_stats.py、make_figures.py
results/        逐幕原始 npy（16 个 agent/场景）、main_results.csv、
                main_results_perseed.csv、synergy_ablation.csv、group_eval_matrix.csv、
                adaptive_raw.csv、coverage/density/od-prior 表
figs_ieee/      IEEE 风格图源文件（PDF 矢量）
lib/            共享绘图/持久化辅助（零硬编码路径）
LICENSE, README.md, README.zh-CN.md, requirements.txt
```

---

## 6. 原始数据参考（`results/`）

| 文件 | 内容 |
|---|---|
| `raw_zoneaux_s42_main.npy` | 本文方法，种子 42（12 ep） |
| `raw_zoneaux_s360.npy` / `raw_zoneaux_s712.npy` | 本文方法，种子 360/712 |
| `raw_noaux92_s42.npy` / `raw_noaux122_s42.npy` | 无 aux 基线，种子 42 |
| `raw_eventaux92_s42.npy` | 92 维观测上的 zone aux（有害消融） |
| `raw_mixaux122_s42.npy` | MIX 配方 |
| `raw_sd_eta.npy` / `raw_sd_nearest.npy` | 规则基线 |
| `raw_zone20f_s{42,360,712}.npy` | 20F 泛化 |
| `raw_20f_sd_eta.npy` / `raw_20f_sd_nearest.npy` | 20F 规则 |
| `raw_group_aligned_s42.npy` / `raw_group_unaligned_s42.npy` | 被否决的 group 训练 |
| `raw_burst2_s42.npy` | 被否决的 burst 展开训练 |
| `raw_zoneaux_s42_legacy.npy` | 旧分离架构（作废） |
| `group_eval_matrix.csv` | 全部模型的 reward/pax/wait 汇总表 |
| `adaptive_raw.csv` | MIX ×3.0 逐幕（负结果） |

---

## 7. 实验谱系（尝试过并被否决的方案——全部可复现）

1. **Group 训练（对齐）**——按人次对齐率直接训练真实组呼梯：辅助信号太稀疏
   （event_acc ≈ 0.51 ≈ 先验）→ −1,363，不如单呼梯训练（−793）。
2. **Group 训练（未对齐）**——率未除组规模：866 pax/h = 8.2× 超载 → do-nothing
   局部最优 → −4,285。
3. **Burst 展开**——组呼梯展开为 0.1 s 偏移的同层同目的单人事件：序列密度保持
   （event_acc 0.64 可学），但策略学到容量保守行为（pax 145 vs 581）→ −1,543。
4. **混合密度课程**——包络内匹配，3.0× 外推更差。
5. **旧分离架构**（旧配置 `type: gru`）——actor/critic 分离编码器 + skip dest head
   读的是时间特征而非目的地（89 维观测无目的地信息）→ 作废；由共享 GRU + event_head
   （`type: gru_shared`）取代。
6. **reward_pred + value_replay 辅助**——最终配方已关闭（λ=0.0，无收益只有开销）。

以上均可由 `configs/` 中的配置复现（`config_gru_shared_daux_group.yaml`、
`config_gru_shared_daux_burst.yaml`、`config_gru_shared_zone_hd.yaml` 等），
原始结果已归档于 `results/`。

---

## 8. 引用

```bibtex
@article{song2026zoneaware,
  title={Learning to Dispatch Where Rules Collapse: Zone-Aware GRU--RL Elevator Group
         Control via Single-Call Decomposition Training},
  author={Song, Chenle},
  journal={IEEE Access},
  year={2026}
}
```

## 9. 许可证

MIT（见 `LICENSE`）。
