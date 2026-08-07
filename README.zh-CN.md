# 变密度客流下的电梯群控：行为克隆 vs 强化学习

本仓库是投稿至 *Engineering Applications of Artificial Intelligence*（EAAI）的论文
《Behavior cloning versus reinforcement learning for elevator group control under
varying traffic density》的完整可复现工件。包含全部代码、配置、逐幕结果、统计检验与
图件，可端到端复现。论文稿件本身不随本仓库分发（双盲审稿）。

**协议一句话总结：** 在标定密度谱（Al-Sharif 基础到达率的 4x / 8x / 12x）上，对比
30 分钟监督克隆（BC）、3 小时从头训练的 PPO 与经典规则调度，全部在 held-out
12 小时 episode、真实乘客组规模（每组 1–10 人、均值 3.71、由公开电梯客流数据集校准）
下评估。

---

## 1. 核心结果

**过渡密度 8x**（九个 held-out 种子，12 小时 episode 奖励均值 ± SD；与 SD-ETA 规则
在同一组 episode 上做配对 t 检验）：

| Agent | Reward | 平均等待 (s) | p95 等待 (s) | 配对 t（vs 规则）|
|---|---|---|---|---|
| SD-nearest（距离） | −51.5 ± 22.8 | 34.3 | — | — |
| SD-ETA（教师规则） | −38.4 ± 30.0 | 32.0 | 77.9 | — |
| **BC**（克隆，e20，30 分钟） | **−8.5 ± 33.0** | 27.2 | 72.1 | t = 3.52，**p = 0.008** |
| **PPO 从头训练**（3 小时） | **−3.4 ± 45.2** | 26.1 | 68.3 | t = 2.63，**p = 0.030** |
| PPO 微调（lr 3e-5） | −9.4 ± 30.9 | — | — | — |

BC 与 PPO 统计不可区分（p = 0.64）：在过渡密度，30 分钟监督克隆即可匹配 3 小时
强化学习策略。

**密度–奖励剖面**（规则 / BC / PPO；8x 用 9 种子，4x/12x 用 3 种子）：

| 密度 | SD-ETA | BC | PPO |
|---|---|---|---|
| 4x（舒适区） | +54.3 ± 1.2 | +54.0 ± 1.2 | +49.0 ± 2.9（过度干预，p = 0.021）|
| 8x（过渡区） | −38.4 ± 30.0 | **−8.5 ± 33.0** | **−3.4 ± 45.2** |
| 12x（过载区） | −1188 ± 154 | −1180 ± 72 | −1231 ± 50（不可区分）|

"RL 的价值集中在极端过载"这一常见说法**未被复现**：在正确标定的需求模型下，
RL 的价值位于规则被压垮但回报信号仍可学习的过渡带。

**策略蒸馏**（PPO 克隆，30 分钟）：−5.8 ± 40.9 vs PPO 教师 −3.4 ± 45.2，p = 0.80——
3 小时 PPO 无损蒸馏进 30 分钟监督克隆。

**本仓库复现的关键发现：**
1. **密度机制决定一切** —— 4x 所有策略近最优；8x 两个学习策略显著优于规则
   （p = 0.008 / 0.030）；12x 全部方法塌缩为统计不可区分的大负奖励。
2. **匹配率–奖励非线性** —— 教师匹配率从 93.6% 提到 99.3%（20 个监督 epoch）
   使 held-out 奖励提升约 7 倍；低于 ~99% 匹配时克隆奖励急剧下降。
3. **双关键特征载体** —— exact-ETA 特征与轿厢呼叫分布缺一不可（移除任一奖励退化
   ~6 倍：无 ETA −50.4，p = 0.044；无轿厢呼叫 −47.5，p = 0.042）。
4. **微调不稳定性** —— 对欠训练克隆（93.6% 匹配）做 PPO 微调单调发散（价值函数
   拟合近最优策略，优势信号退化为噪声）；对良好克隆（99.3%）微调稳定但无增益。
5. **跨密度鲁棒性** —— 8x 训练的 BC 在 4x/12x 保持稳健；PPO 在 4x 过度干预。
6. **20 层楼迁移** —— BC 流水线在 4x 正奖励迁移（+15.0 ± 14.1）。
7. **教师选择不如协议重要** —— 克隆 SD-ETA 规则与克隆 PPO 奖励不可区分（p = 0.61）；
   瓶颈在可克隆性而非教师。

---

## 2. 协议

**BC 训练**（`scripts/bc_boost.py`）：
- 40 条教师 episode（种子 2000 + i·13，与所有评估种子不相交），8x 组事件流
  （SCALE = 3.2；每组 1–10 人，均值 3.71），每条 12 小时（112,664 个决策状态）。
- 64 步序列分块交叉熵，Adam lr 3e-4，50 epochs；选择 epoch 20 检查点（泛化甜点）。
  单卡约 30 分钟。

**PPO 训练**（`src/train.py`，`configs/config_gru_shared_event_d8x_noaux.yaml`）：
- 共享 GRU（2×256）+ actor/critic 双头，无辅助目标。
- 8x 单乘客分解（序列密集；BC 协议对照显示分解无显著影响，−10.6 vs −8.5，p = 0.61），
  60 epochs，8 并行环境，rollout 32768，reward scale 0.01。单卡约 3 小时。
- `configs/config_gru_shared_event_d8x_noaux_group.yaml` 是组到达训练变体，
  用于验证训练分布问题。

**独立评估**（九个 held-out 种子 9999, 10001, ..., 10015）：
- 12 小时 episode，8x 真实组到达（SCALE = 3.2），确定性贪心，GRU 隐藏状态逐
  episode 重置。
- 规则基线：SD-nearest、SD-ETA；分区基线：静态 sectoring（3 区）。
- 服务指标：平均等待、p95 等待、能耗（运动学模型）。

**密度扫描：** SCALE 1.6 = 4x，3.2 = 8x，4.8 = 12x。

**鲁棒性探针（审稿驱动）：**
- `results/wcorr0_9seed.json` —— 关闭分配一致奖励（w_corr = 0）重评估：
  结论不变（BC −9.8，PPO −4.2，rule −39.7）。
- `results/eta_noise_9seed.json` —— exact-ETA 特征 ±10/20% 乘性噪声：
  BC −8.5 → −16.8，PPO −3.4 → +2.1，仍远优于无噪规则（−38.4）。
- `results/sectoring_eval.json` —— 静态分区塌缩（8x −3,283），源于大厅区电梯饱和。
- `results/wait_percentiles_8x.json` —— 95,566 名乘客的均值/中位/p90/p95 等待。

---

## 3. 快速开始

```bash
# 1. 环境（Python >= 3.11）
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # 或: uv sync

# 2. 训练 BC 克隆（RTX 4060 级单卡约 30 分钟）
TRAIN_MODE=group SCALE=3.2 N_DEMO=40 EPOCHS=50 OUT_TAG=bc_boost SEED=42 \
  .venv/bin/python scripts/bc_boost.py

# 3. 从头训练 PPO（约 3 小时）
.venv/bin/python src/train.py --config configs/config_gru_shared_event_d8x_noaux.yaml \
  --seed 42 --epochs 60 --checkpoint-dir checkpoints/ppo_noaux_s42

# 4. 真实组到达独立评估（9 个 held-out 种子）
.venv/bin/python scripts/eval_matrix.py

# 5. 统计检验（配对 t，与论文数字一致）
.venv/bin/python scripts/eval_significance.py

# 6. 图件（IEEE 风格，PDF 矢量）
.venv/bin/python scripts/make_figures.py
```

---

## 4. 仓库结构

```
configs/        YAML 配置：最终 PPO 配方（noaux）、组变体、消融、20F
src/            train.py（PPO 循环、_adapt_gen_events 三模式）、env/（elevator_env，
                n_pax 与容量、精确运动学 ETA）、traffic/（生成器、OD 矩阵、
                passenger_profile）、models/（gru_ppo GRUSharedActorCritic）、zone_map.py
scripts/        bc_boost.py（BC 训练/评估）、eval_matrix.py、eval_significance.py、
                eval_multi_metric.py、eval_finetune_window.py、cmp_actions.py、
                eval_20f_group.py、pool_stats.py、make_figures.py
results/        每个实验的 9-seed JSON（规则、BC、PPO、蒸馏、消融、密度扫描、
                审稿鲁棒性探针）、旧 raw npy（已标注过时）、rtt_validation.json
figs_ieee/      IEEE 风格图件源（PDF 矢量 + PNG）
LICENSE、README.md、README.zh-CN.md、requirements.txt
```

---

## 5. 关键结果文件（`results/`）

| 文件 | 内容 |
|---|---|
| `rule_9seed.json` | SD-ETA 规则，9 种子 @ 8x |
| `bc_boost_e20_9seed.json` | BC 克隆 e20，9 种子 @ 8x（−8.5 ± 33.0）|
| `ppo_noaux_9seed.json` | 纯 PPO（−3.4 ± 45.2）与 PPO+aux（+3.2 ± 46.1）|
| `bc_ppot_9seed.json` | PPO 克隆蒸馏（−5.8 ± 40.9）|
| `bc_single_9seed.json` | 单乘客训练 BC（−10.6 ± 30.7）— 协议对照 |
| `bc_density_eval.json` | BC + 规则跨 4x/8x/12x |
| `ppo_noaux_density3.json` | PPO 跨 4x/8x/12x |
| `bc_density_train_9seed.json` | 训练密度敏感性 |
| `bc_nocar_9seed.json` / no-ETA 曲线 | 特征消融 |
| `multi_metric_8x.json` / `multi_metric_noaux_8x.json` | 等待/能耗权衡 |
| `significance_8x.json` | 配对 t（BC t=3.517，PPO t=2.630）|
| `wcorr0_9seed.json`、`eta_noise_9seed.json` | 审稿鲁棒性探针 |
| `sectoring_eval.json`、`wait_percentiles_8x.json` | 分区基线与服务指标 |
| `rtt_validation.json` | 流量级吞吐验证 vs 经典 RTT 理论 |
| `raw_*.npy` | 旧 zone-aux 时代逐幕数组 — **已过时**（见 §6）|

---

## 6. 实验谱系（已尝试并否决的设计，全部可复现）

1. **Zone-Aware 辅助头（2026-07 前的工作线）** —— 共享 GRU 上的下一事件分区预测。
   已取代：辅助目标对纯监督克隆无增益（BC+aux −30.7 vs 纯 BC −8.5，e20，n = 9），
   辅助头已从稿件移除。`config_gru_shared_daux_*.yaml` 与 `raw_zoneaux_*.npy`
   记录该工作线。
2. **组训练（对齐/未对齐）** —— 直接按人次对齐率在组呼叫上训练：辅助信号稀疏；
   已被当前协议取代。
3. **突发展开** —— 将组呼叫展开为 0.1s 偏移的单呼叫：策略偏保守，奖励更差。
4. **旧分离架构**（`type: gru`）—— 独立 actor/critic 编码器 + 读取时间特征而非
   目的地的 skip 连接 dest head：无效（观测无目的地信息）；已由共享 GRU
   （`type: gru_shared`）取代。
5. **reward_pred + value_replay 辅助** —— 无增益，已禁用。

所有否决设计仍可由存档配置与原始数组复现。

---

## 7. 引用

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

## 8. 许可证

MIT（见 `LICENSE`）。
