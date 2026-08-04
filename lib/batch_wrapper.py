"""
多呼梯并行调度环境包装器。
将连续dt=0的呼梯视为一个"批次"，在批次内检测负载均衡。
"""
import numpy as np
from src.env.elevator_env import ElevatorEnv


class BatchDispatchWrapper:
    """包装ElevatorEnv: 检测多呼梯批次, 添加负载均衡惩罚"""

    def __init__(self, env, batch_penalty=-0.5):
        self.env = env
        self.batch_penalty = batch_penalty
        self.action_space = env.action_space
        self.num_elevators = env.num_elevators
        self.STATE_DIM = env.STATE_DIM
        self._batch_assignments = {0: 0, 1: 0, 2: 0}
        self._batch_id = ""
        self._total_batch_penalty = 0.0

    def reset(self, **kwargs):
        self._batch_assignments = {0: 0, 1: 0, 2: 0}
        self._batch_id = ""
        self._total_batch_penalty = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        n_calls_before = len(self.env.pending_calls)
        batch_key = f"{n_calls_before}_{id(self.env.pending_calls)}"

        # 检测新批次
        if batch_key != self._batch_id:
            self._batch_assignments = {0: 0, 1: 0, 2: 0}
            self._batch_id = batch_key

        # 执行分配
        obs, reward, done, truncated, info = self.env.step(action)

        # 跟踪本批次分配
        self._batch_assignments[action] += 1

        # 如果有多个呼梯且dt=0（批次内），加负载均衡惩罚
        extra_penalty = 0.0
        if len(self.env.pending_calls) > 1 and self.env.elapsed == 0:
            # 检查是否有其他空闲电梯
            n_assigned = self._batch_assignments[action]
            idle_others = sum(1 for i in range(3) if i != action
                              and self._batch_assignments[i] < n_assigned)
            if idle_others > 0:
                extra_penalty = self.batch_penalty
                self._total_batch_penalty += extra_penalty

        return obs, reward + extra_penalty, done, truncated, info

    def inject_batch_calls(self, n_extra=2):
        """注入额外呼梯, 模拟多呼梯同时到达"""
        for _ in range(n_extra):
            sf = np.random.randint(1, 11)
            direction = 1 if np.random.random() < 0.6 else -1
            dest = sf + direction
            if dest < 1: dest = 2
            if dest > 10: dest = 9
            self.env.pending_calls.append({
                "floor": sf, "dest": dest, "direction": direction,
                "passenger_id": 999 + _, "arrival_time": 0.0,
            })
            if direction == 1:
                self.env._obs_buffer[59 + sf] = 1.0
            else:
                self.env._obs_buffer[69 + sf] = 1.0

    @property
    def pending_calls(self):
        return self.env.pending_calls

    def close(self):
        self.env.close()
