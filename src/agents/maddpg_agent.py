import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class Actor(nn.Module):
    """
    Actor network for MADDPG.
    Outputs deterministic action given observation.
    """
    
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Actions bounded to [-1, 1]
        )
    
    def forward(self, obs):
        return self.network(obs)


class Critic(nn.Module):
    """
    Centralized critic for MADDPG.
    Takes all agents' observations and actions as input.
    """
    
    def __init__(self, total_obs_dim, total_action_dim, hidden_dim=128):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(total_obs_dim + total_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, obs, actions):
        """
        Args:
            obs: Concatenated observations from all agents
            actions: Concatenated actions from all agents
        """
        x = torch.cat([obs, actions], dim=-1)
        return self.network(x)


class ReplayBuffer:
    """Experience replay buffer for off-policy learning."""
    
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience):
        """
        Store experience tuple.
        experience = (obs, actions, rewards, next_obs, dones)
        where obs, actions, rewards are dicts with agent_id keys
        """
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """Sample random batch from buffer."""
        batch = random.sample(self.buffer, batch_size)
        return batch
    
    def __len__(self):
        return len(self.buffer)


class MADDPGAgent:
    """
    Multi-Agent Deep Deterministic Policy Gradient.
    
    Key features:
    - Centralized critics (see all observations and actions)
    - Decentralized actors (only see own observation)
    - Target networks for stability
    - Experience replay
    """
    
    def __init__(
        self,
        n_agents,
        obs_dim,
        action_dim,
        lr_actor=1e-3,
        lr_critic=1e-3,
        gamma=0.99,
        tau=0.01,  # Soft update parameter
        buffer_capacity=100000,
        device='cpu'
    ):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device
        
        # Create actor and critic for each agent
        self.actors = []
        self.critics = []
        self.target_actors = []
        self.target_critics = []
        self.actor_optimizers = []
        self.critic_optimizers = []
        
        total_obs_dim = obs_dim * n_agents
        total_action_dim = action_dim * n_agents
        
        for i in range(n_agents):
            # Actor (decentralized)
            actor = Actor(obs_dim, action_dim).to(device)
            target_actor = Actor(obs_dim, action_dim).to(device)
            target_actor.load_state_dict(actor.state_dict())
            
            # Critic (centralized)
            critic = Critic(total_obs_dim, total_action_dim).to(device)
            target_critic = Critic(total_obs_dim, total_action_dim).to(device)
            target_critic.load_state_dict(critic.state_dict())
            
            # Optimizers
            actor_optimizer = optim.Adam(actor.parameters(), lr=lr_actor)
            critic_optimizer = optim.Adam(critic.parameters(), lr=lr_critic)
            
            self.actors.append(actor)
            self.critics.append(critic)
            self.target_actors.append(target_actor)
            self.target_critics.append(target_critic)
            self.actor_optimizers.append(actor_optimizer)
            self.critic_optimizers.append(critic_optimizer)
        
        # Shared replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # Exploration noise
        self.noise_scale = 0.1
        self.noise_decay = 0.9995
        self.noise_min = 0.01
    
    def select_actions(self, observations, explore=True):
        """
        Select actions for all agents.
        
        Args:
            observations: Dict of {agent_id: observation}
            explore: Whether to add exploration noise
        
        Returns:
            Dict of {agent_id: action}
        """
        actions = {}
        
        for i in range(self.n_agents):
            obs_tensor = torch.FloatTensor(observations[i]).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action = self.actors[i](obs_tensor).cpu().numpy()[0]
            
            # Add exploration noise
            if explore:
                noise = np.random.normal(0, self.noise_scale, size=action.shape)
                action = np.clip(action + noise, -1.0, 1.0)
            
            actions[i] = action
        
        return actions
    
    def store_transition(self, obs, actions, rewards, next_obs, dones):
        """Store transition in replay buffer."""
        self.replay_buffer.push((obs, actions, rewards, next_obs, dones))
    
    def update(self, batch_size=64):
        """
        Update all agents using MADDPG algorithm.
        
        Returns:
            Dict of losses for logging
        """
        if len(self.replay_buffer) < batch_size:
            return None
        
        # Sample batch
        batch = self.replay_buffer.sample(batch_size)
        
        # Unpack batch
        obs_batch = []
        actions_batch = []
        rewards_batch = []
        next_obs_batch = []
        dones_batch = []
        
        for experience in batch:
            obs, actions, rewards, next_obs, dones = experience
            
            # Stack observations and actions from all agents
            obs_batch.append([obs[i] for i in range(self.n_agents)])
            actions_batch.append([actions[i] for i in range(self.n_agents)])
            rewards_batch.append([rewards[i] for i in range(self.n_agents)])
            next_obs_batch.append([next_obs[i] for i in range(self.n_agents)])
            dones_batch.append([dones[i] for i in range(self.n_agents)])
        
        # Convert to tensors
        # Shape: [batch_size, n_agents, dim]
        obs_batch = torch.FloatTensor(np.array(obs_batch)).to(self.device)
        actions_batch = torch.FloatTensor(np.array(actions_batch)).to(self.device)
        rewards_batch = torch.FloatTensor(np.array(rewards_batch)).to(self.device)
        next_obs_batch = torch.FloatTensor(np.array(next_obs_batch)).to(self.device)
        dones_batch = torch.FloatTensor(np.array(dones_batch)).to(self.device)
        
        total_actor_loss = 0
        total_critic_loss = 0
        
        # Update each agent
        for agent_id in range(self.n_agents):
            # === Update Critic ===
            
            # Get next actions from target actors
            next_actions = []
            for i in range(self.n_agents):
                next_action = self.target_actors[i](next_obs_batch[:, i])
                next_actions.append(next_action)
            next_actions = torch.stack(next_actions, dim=1)  # [batch, n_agents, action_dim]
            
            # Flatten observations and actions for critic
            next_obs_flat = next_obs_batch.reshape(batch_size, -1)
            next_actions_flat = next_actions.reshape(batch_size, -1)
            
            # Compute target Q-value
            with torch.no_grad():
                target_q = self.target_critics[agent_id](next_obs_flat, next_actions_flat)
                target_q = rewards_batch[:, agent_id].unsqueeze(1) + \
                           self.gamma * target_q * (1 - dones_batch[:, agent_id].unsqueeze(1))
            
            # Current Q-value
            obs_flat = obs_batch.reshape(batch_size, -1)
            actions_flat = actions_batch.reshape(batch_size, -1)
            current_q = self.critics[agent_id](obs_flat, actions_flat)
            
            # Critic loss
            critic_loss = nn.MSELoss()(current_q, target_q)
            
            # Update critic
            self.critic_optimizers[agent_id].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[agent_id].parameters(), 1.0)
            self.critic_optimizers[agent_id].step()
            
            # === Update Actor ===
            
            # Get current actions from all actors
            current_actions = []
            for i in range(self.n_agents):
                if i == agent_id:
                    # Use current actor being updated
                    current_action = self.actors[i](obs_batch[:, i])
                else:
                    # Use fixed actions from other agents
                    with torch.no_grad():
                        current_action = self.actors[i](obs_batch[:, i])
                current_actions.append(current_action)
            current_actions = torch.stack(current_actions, dim=1)
            current_actions_flat = current_actions.reshape(batch_size, -1)
            
            # Actor loss: maximize Q-value
            actor_loss = -self.critics[agent_id](obs_flat, current_actions_flat).mean()
            
            # Update actor
            self.actor_optimizers[agent_id].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), 1.0)
            self.actor_optimizers[agent_id].step()
            
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
        
        # Soft update target networks
        self._soft_update_targets()
        
        # Decay exploration noise
        self.noise_scale = max(self.noise_min, self.noise_scale * self.noise_decay)
        
        return {
            'actor_loss': total_actor_loss / self.n_agents,
            'critic_loss': total_critic_loss / self.n_agents,
            'noise_scale': self.noise_scale
        }
    
    def _soft_update_targets(self):
        """Soft update target networks."""
        for i in range(self.n_agents):
            # Update target actor
            for param, target_param in zip(self.actors[i].parameters(), 
                                          self.target_actors[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            # Update target critic
            for param, target_param in zip(self.critics[i].parameters(), 
                                          self.target_critics[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, path):
        """Save all agent models."""
        checkpoint = {
            'actors': [actor.state_dict() for actor in self.actors],
            'critics': [critic.state_dict() for critic in self.critics],
            'target_actors': [actor.state_dict() for actor in self.target_actors],
            'target_critics': [critic.state_dict() for critic in self.target_critics],
        }
        torch.save(checkpoint, path)
    
    def load(self, path):
        """Load all agent models."""
        checkpoint = torch.load(path)
        for i in range(self.n_agents):
            self.actors[i].load_state_dict(checkpoint['actors'][i])
            self.critics[i].load_state_dict(checkpoint['critics'][i])
            self.target_actors[i].load_state_dict(checkpoint['target_actors'][i])
            self.target_critics[i].load_state_dict(checkpoint['target_critics'][i])