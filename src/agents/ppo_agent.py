import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.
    
    Actor: Outputs mean and std for action distribution
    Critic: Outputs state value estimate
    """
    
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super().__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head (value function)
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, obs):
        """Forward pass through network."""
        features = self.shared(obs)
        
        # Actor outputs
        action_mean = torch.tanh(self.actor_mean(features))  # Bounded to [-1, 1]
        action_std = torch.exp(self.actor_logstd)
        
        # Critic output
        value = self.critic(features)
        
        return action_mean, action_std, value
    
    def get_action(self, obs, deterministic=False):
        """Sample action from policy."""
        action_mean, action_std, value = self.forward(obs)
        
        if deterministic:
            return action_mean, value
        
        # Sample from Gaussian distribution
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        action = torch.clamp(action, -1.0, 1.0)
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob, value


class PPOAgent:
    """
    Proximal Policy Optimization agent.
    
    Key features:
    - Clipped objective to prevent large policy updates
    - Generalized Advantage Estimation (GAE)
    - Value function learning
    """
    
    def __init__(
        self,
        obs_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        device='cpu'
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        # Initialize network
        self.network = ActorCritic(obs_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Storage for rollout data
        self.reset_storage()
    
    def reset_storage(self):
        """Clear rollout storage."""
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def select_action(self, obs, deterministic=False):
        """Select action given observation."""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if deterministic:
                action, value = self.network.get_action(obs_tensor, deterministic=True)
                return action.cpu().numpy()[0], None, value.cpu().item()
            else:
                action, log_prob, value = self.network.get_action(obs_tensor)
                return action.cpu().numpy()[0], log_prob.cpu().item(), value.cpu().item()
    
    def store_transition(self, obs, action, log_prob, reward, value, done):
        """Store transition in rollout buffer."""
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_advantages(self, next_value):
        """
        Compute advantages using Generalized Advantage Estimation (GAE).
        
        GAE reduces variance while maintaining low bias.
        """
        advantages = []
        gae = 0
        
        # Iterate backwards through episode
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_val = next_value
            else:
                next_val = self.values[t + 1]
            
            # TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            delta = self.rewards[t] + self.gamma * next_val * (1 - self.dones[t]) - self.values[t]
            
            # GAE: A_t = δ_t + (γλ) * δ_{t+1} + ...
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
        
        return advantages
    
    def update(self, next_obs, n_epochs=10, batch_size=64):
        """
        Update policy using PPO algorithm.
        
        Args:
            next_obs: Next observation (for bootstrapping)
            n_epochs: Number of optimization epochs
            batch_size: Minibatch size
        """
        # Get next value for advantage computation
        with torch.no_grad():
            next_obs_tensor = torch.FloatTensor(next_obs).unsqueeze(0).to(self.device)
            _, _, next_value = self.network(next_obs_tensor)
            next_value = next_value.item()
        
        # Compute advantages
        advantages = self.compute_advantages(next_value)
        returns = [adv + val for adv, val in zip(advantages, self.values)]
        
        # Convert to tensors
        obs_tensor = torch.FloatTensor(np.array(self.observations)).to(self.device)
        actions_tensor = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # PPO update for multiple epochs
        n_samples = len(self.observations)
        indices = np.arange(n_samples)
        
        for epoch in range(n_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                # Get current policy outputs
                action_mean, action_std, values = self.network(obs_tensor[batch_indices])
                dist = Normal(action_mean, action_std)
                
                # Calculate log probs and entropy
                log_probs = dist.log_prob(actions_tensor[batch_indices]).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                
                # Importance sampling ratio
                ratio = torch.exp(log_probs - old_log_probs_tensor[batch_indices])
                
                # Clipped surrogate objective
                advantages_batch = advantages_tensor[batch_indices]
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_batch
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value function loss
                value_loss = nn.MSELoss()(values.squeeze(), returns_tensor[batch_indices])
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
        
        # Clear storage
        self.reset_storage()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }
    
    def save(self, path):
        """Save model checkpoint."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])