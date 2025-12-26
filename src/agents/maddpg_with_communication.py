import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class CommunicationActor(nn.Module):
    """
    Actor with explicit communication capability.
    
    Process:
    1. Encode own observation
    2. Generate message to broadcast to other agents
    3. Receive and process messages from other agents
    4. Output action based on own state + received messages
    """
    
    def __init__(self, obs_dim, action_dim, comm_dim=32, n_agents=5, hidden_dim=128):
        super().__init__()
        self.comm_dim = comm_dim
        self.n_agents = n_agents
        
        # Encode own observation
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Generate communication message from encoded observation
        self.message_generator = nn.Sequential(
            nn.Linear(hidden_dim, comm_dim),
            nn.Tanh()  # Bound messages to [-1, 1]
        )
        
        # Process messages received from other agents
        # Input: concatenated messages from (n_agents - 1) other agents
        self.message_processor = nn.Sequential(
            nn.Linear(comm_dim * (n_agents - 1), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Final policy network: combines own encoding + processed messages
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
    
    def forward(self, obs, messages_from_others=None):
        """
        Args:
            obs: [batch_size, obs_dim] - agent's own observation
            messages_from_others: [batch_size, (n_agents-1) * comm_dim] - concatenated messages
        
        Returns:
            action: [batch_size, action_dim]
            my_message: [batch_size, comm_dim] - message to broadcast
        """
        # Encode own observation
        obs_encoded = self.obs_encoder(obs)
        
        # Generate message to send to others
        my_message = self.message_generator(obs_encoded)
        
        # Process messages from other agents
        if messages_from_others is not None:
            comm_features = self.message_processor(messages_from_others)
        else:
            # If no messages (e.g., first call), use zeros
            batch_size = obs.shape[0]
            comm_features = torch.zeros(batch_size, self.policy[0].in_features - obs_encoded.shape[1]).to(obs.device)
        
        # Combine own encoding with communication features
        combined = torch.cat([obs_encoded, comm_features], dim=-1)
        
        # Output action
        action = self.policy(combined)
        
        return action, my_message


class CommunicationCritic(nn.Module):
    """
    Centralized critic that sees:
    - All observations
    - All actions
    - All communication messages
    """
    
    def __init__(self, total_obs_dim, total_action_dim, total_comm_dim, hidden_dim=128):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(total_obs_dim + total_action_dim + total_comm_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, obs, actions, messages):
        """
        Args:
            obs: Concatenated observations from all agents
            actions: Concatenated actions from all agents
            messages: Concatenated messages from all agents
        """
        x = torch.cat([obs, actions, messages], dim=-1)
        return self.network(x)


class ReplayBuffer:
    """Experience replay buffer for off-policy learning."""
    
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience):
        """
        Store experience tuple.
        experience = (obs, actions, messages, rewards, next_obs, next_messages, dones)
        """
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """Sample random batch from buffer."""
        batch = random.sample(self.buffer, batch_size)
        return batch
    
    def __len__(self):
        return len(self.buffer)


class CommMADDPGAgent:
    """
    Multi-Agent DDPG with Explicit Communication.
    
    Key additions over standard MADDPG:
    - Actors generate and process messages
    - Critics consider communication in value estimation
    - Messages are learned end-to-end (no predefined protocol)
    """
    
    def __init__(
        self,
        n_agents,
        obs_dim,
        action_dim,
        comm_dim=32,
        lr_actor=1e-3,
        lr_critic=1e-3,
        gamma=0.99,
        tau=0.01,
        buffer_capacity=100000,
        device='cpu'
    ):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.comm_dim = comm_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device
        
        # Create communication-enabled actors and critics
        self.actors = []
        self.critics = []
        self.target_actors = []
        self.target_critics = []
        self.actor_optimizers = []
        self.critic_optimizers = []
        
        total_obs_dim = obs_dim * n_agents
        total_action_dim = action_dim * n_agents
        total_comm_dim = comm_dim * n_agents
        
        for i in range(n_agents):
            # Communication Actor
            actor = CommunicationActor(obs_dim, action_dim, comm_dim, n_agents).to(device)
            target_actor = CommunicationActor(obs_dim, action_dim, comm_dim, n_agents).to(device)
            target_actor.load_state_dict(actor.state_dict())
            
            # Communication-aware Critic
            critic = CommunicationCritic(total_obs_dim, total_action_dim, total_comm_dim).to(device)
            target_critic = CommunicationCritic(total_obs_dim, total_action_dim, total_comm_dim).to(device)
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
        
        print(f"Initialized CommMADDPG with {n_agents} agents")
        print(f"  Observation dim: {obs_dim}")
        print(f"  Action dim: {action_dim}")
        print(f"  Communication dim: {comm_dim}")
    
    def select_actions(self, observations, explore=True):
        """
        Select actions for all agents with communication.
        
        Process:
        1. All agents generate messages based on their observations
        2. Messages are exchanged
        3. Each agent selects action considering others' messages
        
        Args:
            observations: Dict of {agent_id: observation}
            explore: Whether to add exploration noise
        
        Returns:
            Dict of {agent_id: action}
        """
        actions = {}
        messages = {}
        
        # Step 1: Generate messages from all agents
        for i in range(self.n_agents):
            obs_tensor = torch.FloatTensor(observations[i]).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Generate message (no incoming messages yet)
                _, message = self.actors[i](obs_tensor, messages_from_others=None)
                messages[i] = message.cpu().numpy()[0]
        
        # Step 2: Each agent selects action based on own obs + others' messages
        for i in range(self.n_agents):
            obs_tensor = torch.FloatTensor(observations[i]).unsqueeze(0).to(self.device)
            
            # Collect messages from other agents (exclude own message)
            other_messages = []
            for j in range(self.n_agents):
                if j != i:
                    other_messages.append(messages[j])
            
            # Concatenate messages from others
            messages_tensor = torch.FloatTensor(np.concatenate(other_messages)).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action, _ = self.actors[i](obs_tensor, messages_tensor)
                action = action.cpu().numpy()[0]
            
            # Add exploration noise
            if explore:
                noise = np.random.normal(0, self.noise_scale, size=action.shape)
                action = np.clip(action + noise, -1.0, 1.0)
            
            actions[i] = action
        
        return actions
    
    def store_transition(self, obs, actions, rewards, next_obs, dones):
        """
        Store transition in replay buffer.
        Note: We don't store messages separately since they're computed on-the-fly
        """
        self.replay_buffer.push((obs, actions, rewards, next_obs, dones))
    
    def update(self, batch_size=64):
        """
        Update all agents using CommMADDPG algorithm.
        
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
            
            obs_batch.append([obs[i] for i in range(self.n_agents)])
            actions_batch.append([actions[i] for i in range(self.n_agents)])
            rewards_batch.append([rewards[i] for i in range(self.n_agents)])
            next_obs_batch.append([next_obs[i] for i in range(self.n_agents)])
            dones_batch.append([dones[i] for i in range(self.n_agents)])
        
        # Convert to tensors [batch_size, n_agents, dim]
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
            
            # Generate next actions and messages from target actors
            next_actions = []
            next_messages = []
            
            for i in range(self.n_agents):
                # First pass: generate messages
                _, message = self.target_actors[i](next_obs_batch[:, i], messages_from_others=None)
                next_messages.append(message)
            
            for i in range(self.n_agents):
                # Second pass: generate actions with messages
                other_msgs = [next_messages[j] for j in range(self.n_agents) if j != i]
                other_msgs_concat = torch.cat(other_msgs, dim=-1)
                action, _ = self.target_actors[i](next_obs_batch[:, i], other_msgs_concat)
                next_actions.append(action)
            
            next_actions = torch.stack(next_actions, dim=1)
            next_messages = torch.stack(next_messages, dim=1)
            
            # Flatten for critic
            next_obs_flat = next_obs_batch.reshape(batch_size, -1)
            next_actions_flat = next_actions.reshape(batch_size, -1)
            next_messages_flat = next_messages.reshape(batch_size, -1)
            
            # Compute target Q-value
            with torch.no_grad():
                target_q = self.target_critics[agent_id](next_obs_flat, next_actions_flat, next_messages_flat)
                target_q = rewards_batch[:, agent_id].unsqueeze(1) + \
                           self.gamma * target_q * (1 - dones_batch[:, agent_id].unsqueeze(1))
            
            # Current Q-value (need to recompute messages from current obs)
            current_messages = []
            for i in range(self.n_agents):
                _, message = self.actors[i](obs_batch[:, i], messages_from_others=None)
                current_messages.append(message)
            current_messages = torch.stack(current_messages, dim=1)
            
            obs_flat = obs_batch.reshape(batch_size, -1)
            actions_flat = actions_batch.reshape(batch_size, -1)
            messages_flat = current_messages.reshape(batch_size, -1)
            
            current_q = self.critics[agent_id](obs_flat, actions_flat, messages_flat)
            
            # Critic loss
            critic_loss = nn.MSELoss()(current_q, target_q)
            
            # Update critic
            self.critic_optimizers[agent_id].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[agent_id].parameters(), 1.0)
            self.critic_optimizers[agent_id].step()
            
            # === Update Actor ===
            
            # Generate current actions and messages with gradient
            current_actions = []
            current_messages_for_actor = []
            
            for i in range(self.n_agents):
                _, message = self.actors[i](obs_batch[:, i], messages_from_others=None)
                current_messages_for_actor.append(message)
            
            for i in range(self.n_agents):
                other_msgs = [current_messages_for_actor[j] for j in range(self.n_agents) if j != i]
                other_msgs_concat = torch.cat(other_msgs, dim=-1)
                
                if i == agent_id:
                    # Current agent - allow gradient flow
                    action, _ = self.actors[i](obs_batch[:, i], other_msgs_concat)
                else:
                    # Other agents - detach
                    with torch.no_grad():
                        action, _ = self.actors[i](obs_batch[:, i], other_msgs_concat)
                
                current_actions.append(action)
            
            current_actions = torch.stack(current_actions, dim=1)
            current_messages_stacked = torch.stack(current_messages_for_actor, dim=1)
            current_actions_flat = current_actions.reshape(batch_size, -1)
            current_messages_flat = current_messages_stacked.reshape(batch_size, -1)
            
            # Actor loss: maximize Q-value
            actor_loss = -self.critics[agent_id](obs_flat, current_actions_flat, current_messages_flat).mean()
            
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
    
    def get_communication_pattern(self, observations):
        """
        Analyze what agents are communicating.
        Useful for debugging and interpretation.
        
        Returns:
            messages: Dict of {agent_id: message_vector}
        """
        messages = {}
        
        for i in range(self.n_agents):
            obs_tensor = torch.FloatTensor(observations[i]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, message = self.actors[i](obs_tensor, messages_from_others=None)
                messages[i] = message.cpu().numpy()[0]
        
        return messages
    
    def save(self, path):
        """Save all agent models."""
        checkpoint = {
            'actors': [actor.state_dict() for actor in self.actors],
            'critics': [critic.state_dict() for critic in self.critics],
            'target_actors': [actor.state_dict() for actor in self.target_actors],
            'target_critics': [critic.state_dict() for critic in self.target_critics],
            'comm_dim': self.comm_dim
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load all agent models."""
        checkpoint = torch.load(path, map_location=self.device)
        for i in range(self.n_agents):
            self.actors[i].load_state_dict(checkpoint['actors'][i])
            self.critics[i].load_state_dict(checkpoint['critics'][i])
            self.target_actors[i].load_state_dict(checkpoint['target_actors'][i])
            self.target_critics[i].load_state_dict(checkpoint['target_critics'][i])
        print(f"Model loaded from {path}")