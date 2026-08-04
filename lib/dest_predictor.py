"""
目的地预测模块。
统计各模式下每层呼梯的目的地概率分布。
"""
from pathlib import Path

import sys, numpy as np, json
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eet"))

from src.data.dataset import load_raw_data

SCENARIO_NAMES = {
    1: "MorningPeak", 2: "EveningPeak", 3: "NoonCross",
    4: "Interfloor", 5: "Idle", 6: "Meeting", 7: "Scatter",
}

def build_destination_table():
    """统计7种场景下每层上呼/下呼的目的地分布。"""
    data = load_raw_data(str(Path(__file__).resolve().parents[2] / "eet" / "datasets"))
    event_seqs = data["event_sequences"]["arr_0"]  # (70, 400, 10)
    event_lens = data["event_lengths"]["arr_0"]
    labels = np.squeeze(data["labels"]["arr_0"])
    
    # dest_table[scenario][src_floor][direction] = [dest_floor_counts]
    # direction: 0=up, 1=down
    dest_table = {}
    total_calls = 0
    
    for ep_idx in range(len(event_seqs)):
        label = int(labels[ep_idx])
        scenario_name = SCENARIO_NAMES.get(label, f"Label{label}")
        length = int(event_lens[ep_idx])
        seq = event_seqs[ep_idx][:length]
        
        for ev in seq:
            src = int(ev[0]) + 1    # 0-indexed → 1-indexed
            dst = int(ev[1]) + 1
            if src == dst:
                continue
            
            if dst > src:
                direction = 0  # up
            else:
                direction = 1  # down
            
            dest_table.setdefault(label, {})
            dest_table[label].setdefault(src, {})
            dest_table[label][src].setdefault(direction, {})
            dest_table[label][src][direction][dst] = \
                dest_table[label][src][direction].get(dst, 0) + 1
            total_calls += 1
    
    # 转为概率
    prob_table = {}
    for scenario, src_dict in dest_table.items():
        prob_table[scenario] = {}
        for src, dir_dict in src_dict.items():
            prob_table[scenario][src] = {}
            for direction, dest_counts in dir_dict.items():
                total = sum(dest_counts.values())
                probs = {k: v/total for k, v in dest_counts.items()}
                prob_table[scenario][src][direction] = probs
    
    print(f"Total calls in dataset: {total_calls}")
    print(f"Scenarios: {sorted(prob_table.keys())}")
    
    # 打印几个示例
    for scenario in [1, 2]:
        print(f"\n=== {SCENARIO_NAMES[scenario]} ===")
        for src in sorted(prob_table[scenario].keys())[:3]:
            for direction, probs in prob_table[scenario][src].items():
                dir_name = "上" if direction == 0 else "下"
                top3 = sorted(probs.items(), key=lambda x: -x[1])[:3]
                top3_str = ", ".join(f"F{d}({p:.0%})" for d, p in top3)
                print(f"  F{src}{dir_name}呼 → {top3_str}")
    
    return prob_table


class DestinationPredictor:
    """运行时目的地预测器。"""
    
    def __init__(self, prob_table=None):
        self.prob_table = prob_table or build_destination_table()
        # 合并模式1,3,6(高峰上行)和模式2(晚高峰)等
        self._build_merged()
    
    def _build_merged(self):
        """模式1→6映射: 1/3/6合并, 4/5合并, 2/7独立"""
        merged = {}
        for label, table in self.prob_table.items():
            merged[label] = table
        self.merged_table = merged
    
    def predict(self, scenario_id, src_floor, direction):
        """预测目的地概率分布。
        
        Args:
            scenario_id: 1-7
            src_floor: 1-10
            direction: 0=up, 1=down
        
        Returns:
            np.ndarray shape=(10,) 每层的概率
        """
        probs = np.zeros(10, dtype=np.float32)
        table = self.merged_table.get(scenario_id, {})
        floor_table = table.get(src_floor, {})
        dest_probs = floor_table.get(direction, {})
        
        for dest, prob in dest_probs.items():
            if 1 <= dest <= 10:
                probs[dest - 1] = prob
        
        # 如果没有数据，均匀分布
        if probs.sum() < 0.001:
            probs[:] = 1.0 / 10
        
        return probs / probs.sum()
    
    def augment_obs(self, obs_109, scenario_id):
        """将109维obs增强为129维（增加20维目的地概率）。"""
        up_calls = [bool(obs_109[60 + i] > 0.5) for i in range(10)]
        down_calls = [bool(obs_109[70 + i] > 0.5) for i in range(10)]
        
        dest_features = np.zeros(20, dtype=np.float32)
        for f in range(10):
            if up_calls[f]:
                dest = self.predict(scenario_id, f + 1, 0)
                dest_features[f * 2:f * 2 + 2] = [dest.sum(), 0]
            if down_calls[f]:
                dest = self.predict(scenario_id, f + 1, 1)
                dest_features[f * 2:f * 2 + 2] = [0, dest.sum()]
        
        return np.concatenate([obs_109, dest_features])


if __name__ == "__main__":
    dp = DestinationPredictor()
    
    # 测试
    obs = np.zeros(109, dtype=np.float32)
    obs[60] = 1.0  # F1上呼
    aug = dp.augment_obs(obs, 1)  # 早高峰
    print(f"\nAugmented obs dim: {aug.shape}")  # 129
    
    # 预测
    print(f"\n早高峰F1上呼目的地:")
    probs = dp.predict(1, 1, 0)
    top3 = np.argsort(probs)[-3:][::-1] + 1
    print(f"  最可能: F{top3[0]}({probs[top3[0]-1]:.1%}), F{top3[1]}({probs[top3[1]-1]:.1%}), F{top3[2]}({probs[top3[2]-1]:.1%})")
