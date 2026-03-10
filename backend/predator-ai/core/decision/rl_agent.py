import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Dict, List, Tuple
from loguru import logger

class PPONetwork(nn.Module):
    """PPO algoritması için policy network"""
    
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Actor (policy) head
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic (value) head
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        shared_out = self.shared(x)
        action_probs = self.actor(shared_out)
        value = self.critic(shared_out)
        return action_probs, value

class RLAgent:
    """Pekiştirmeli öğrenme ajanı - pozisyon büyüklüğü optimizasyonu"""
    
    def __init__(self, config):
        self.config = config
        self.state_dim = 128
        self.action_dim = 5  # [0%, 25%, 50%, 75%, 100%]
        
        # PPO network
        self.policy = PPONetwork(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.ai.rl_learning_rate)
        
        # Experience replay
        self.memory = deque(maxlen=config.ai.rl_buffer_size)
        
        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        
        # Epsilon for exploration
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        
    def get_state(self, market_data: Dict, portfolio_data: Dict) -> np.ndarray:
        """Mevcut durumu feature vektörüne dönüştür"""
        
        state = []
        
        # Market features
        state.extend([
            market_data.get('price', 0),
            market_data.get('volume', 0),
            market_data.get('volatility', 0),
            market_data.get('trend', 0),
            market_data.get('rsi', 50),
            market_data.get('macd', 0),
        ])
        
        # Portfolio features
        state.extend([
            portfolio_data.get('total_value', 0),
            portfolio_data.get('cash', 0),
            portfolio_data.get('position_size', 0),
            portfolio_data.get('unrealized_pnl', 0),
            portfolio_data.get('daily_pnl', 0),
        ])
        
        # Risk features
        state.extend([
            portfolio_data.get('current_drawdown', 0),
            portfolio_data.get('var_95', 0),
            portfolio_data.get('sharpe_ratio', 0),
        ])
        
        # Pad to state_dim
        if len(state) < self.state_dim:
            state.extend([0] * (self.state_dim - len(state)))
        elif len(state) > self.state_dim:
            state = state[:self.state_dim]
            
        return np.array(state, dtype=np.float32)
        
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, float]:
        """Aksiyon seç (pozisyon büyüklüğü yüzdesi)"""
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_probs, value = self.policy(state_tensor)
            
        if deterministic:
            # En iyi aksiyon
            action = torch.argmax(action_probs).item()
        else:
            # Exploration: epsilon-greedy
            if np.random.random() < self.epsilon:
                action = np.random.randint(self.action_dim)
            else:
                action = torch.multinomial(action_probs.squeeze(), 1).item()
                
        # Aksiyon değerini hesapla
        action_value = [0, 0.25, 0.5, 0.75, 1.0][action]
        
        return action, action_value
        
    def remember(self, state: np.ndarray, action: int, reward: float, 
                 next_state: np.ndarray, done: bool):
        """Deneyimi hafızaya kaydet"""
        
        self.memory.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })
        
    def update_epsilon(self):
        """Exploration oranını güncelle"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def compute_gae(self, rewards: List[float], values: List[float], dones: List[bool]) -> List[float]:
        """Generalized Advantage Estimation hesapla"""
        
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            
        return advantages
        
    def train(self, batch_size: int = 32):
        """PPO ile policy güncelle"""
        
        if len(self.memory) < batch_size:
            return
            
        # Mini-batch örnekle
        batch = random.sample(self.memory, batch_size)
        
        states = torch.FloatTensor([b['state'] for b in batch])
        actions = torch.LongTensor([b['action'] for b in batch])
        rewards = torch.FloatTensor([b['reward'] for b in batch])
        next_states = torch.FloatTensor([b['next_state'] for b in batch])
        dones = torch.FloatTensor([b['done'] for b in batch])
        
        # Eski policy ile log prob hesapla
        with torch.no_grad():
            old_action_probs, old_values = self.policy(states)
            old_log_probs = torch.log(old_action_probs.gather(1, actions.unsqueeze(1)) + 1e-10)
            
        # GAE hesapla
        advantages = self.compute_gae(
            rewards.tolist(),
            old_values.squeeze().tolist(),
            dones.tolist()
        )
        advantages = torch.FloatTensor(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Returns hesapla
        returns = advantages + old_values.squeeze()
        
        # PPO update
        for _ in range(10):  # Multiple epochs
            action_probs, values = self.policy(states)
            log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)) + 1e-10)
            
            # Ratio hesapla
            ratios = torch.exp(log_probs - old_log_probs.detach())
            
            # Surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic loss
            value_loss = nn.MSELoss()(values.squeeze(), returns.detach())
            
            # Entropy bonus
            entropy = -(action_probs * torch.log(action_probs + 1e-10)).sum(dim=1).mean()
            
            # Total loss
            total_loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Gradient descent
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            
        self.update_epsilon()
        
    def save(self, path: str):
        """Modeli kaydet"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        
    def load(self, path: str):
        """Modeli yükle"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']