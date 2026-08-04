"""
Dueling Double DQN agent for elevator dispatching.
"""

import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random


class DuelingDQN(nn.Module):
    """Dueling architecture: separate Value and Advantage streams."""
    
    def __init__(self, state_dim=109, action_dim=3, hidden=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value_stream = nn.Linear(hidden, 1)
        self.advantage_stream = nn.Linear(hidden, action_dim)
    
    def forward(self, x):
        feats = self.features(x)
        value = self.value_stream(feats)
        advantage = self.advantage_stream(feats)
        # Q(s,a) = V(s) + A(s,a) - mean(A)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(next_states), np.array(dones, dtype=np.float32))
    
    def __len__(self):
        return len(self.buffer)


class DDQNAgent:
    def __init__(self, state_dim=109, action_dim=3, lr=3e-4, gamma=0.99,
                 tau=0.05, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=5000):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps = 0
        
        self.policy_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer()
        self.criterion = nn.MSELoss()
    
    def act(self, state, eval_mode=False):
        if not eval_mode and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return int(q_values.argmax(dim=-1).item())
    
    def update(self, batch_size=64):
        if len(self.memory) < batch_size:
            return 0.0
        
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        s = torch.from_numpy(states).float().to(self.device)
        a = torch.from_numpy(actions).long().unsqueeze(1).to(self.device)
        r = torch.from_numpy(rewards).float().unsqueeze(1).to(self.device)
        ns = torch.from_numpy(next_states).float().to(self.device)
        d = torch.from_numpy(dones).float().unsqueeze(1).to(self.device)
        
        # Double DQN: policy net selects action, target net evaluates
        with torch.no_grad():
            next_actions = self.policy_net(ns).argmax(dim=-1, keepdim=True)
            next_q = self.target_net(ns).gather(1, next_actions)
            target_q = r + self.gamma * next_q * (1 - d)
        
        current_q = self.policy_net(s).gather(1, a)
        loss = self.criterion(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Poljak soft update
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1 - self.tau) * target_param.data)
        
        # Epsilon decay
        self.steps += 1
        self.epsilon = max(self.epsilon_end, self.epsilon * (1 - 1 / self.epsilon_decay))
        
        return loss.item()
    
    def save(self, path):
        torch.save({
            "policy": self.policy_net.state_dict(),
            "target": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
            "epsilon": self.epsilon,
        }, path)
    
    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy"])
        self.target_net.load_state_dict(ckpt["target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.steps = ckpt["steps"]
        self.epsilon = ckpt["epsilon"]
