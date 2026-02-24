"""
Improved MADDPG Training for 100 Agents
Fixes the scaling issues discovered in 20-agent experiments

Key improvements:
1. Larger network capacity (512 hidden units)
2. Better learning rates and noise decay
3. Early stopping to prevent collapse
4. Gradient clipping for stability
5. Checkpoint management (save best, not final)
"""

import sys
sys.path.append('..')

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from src.environments.grid_env import SmartGridEnv


# ============================================================================
# IMPROVED NETWORK ARCHITECTURE
# ============================================================================

class ImprovedActor(nn.Module):
    """Larger actor network for high-dimensional observations"""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=512):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Added normalization
            nn.ReLU(),
            nn.Dropout(0.1),  # Prevent overfitting
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        # Better initialization for deep networks
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, obs):
        return self.network(obs)


class ImprovedCritic(nn.Module):
    """Larger critic network for centralized training"""
    
    def __init__(self, total_obs_dim, total_action_dim, hidden_dim=512):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(total_obs_dim + total_action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, obs, actions):
        x = torch.cat([obs, actions], dim=-1)
        return self.network(x)


# ============================================================================
# IMPROVED MADDPG AGENT
# ============================================================================

class ImprovedMADDPGAgent:
    """Enhanced MADDPG with better scaling properties"""
    
    def __init__(
        self,
        n_agents,
        obs_dim,
        action_dim,
        hidden_dim=512,  # Increased from 128
        lr_actor=3e-4,   # Reduced from 1e-3
        lr_critic=3e-4,  # Reduced from 1e-3
        gamma=0.99,
        tau=0.01,
        buffer_capacity=100000,
        device='cpu'
    ):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device
        
        # Networks
        self.actors = []
        self.critics = []
        self.target_actors = []
        self.target_critics = []
        self.actor_optimizers = []
        self.critic_optimizers = []
        
        total_obs_dim = obs_dim * n_agents
        total_action_dim = action_dim * n_agents
        
        print(f"Creating networks with hidden_dim={hidden_dim}")
        print(f"Actor: {obs_dim} → {hidden_dim} → {hidden_dim} → {action_dim}")
        print(f"Critic: {total_obs_dim + total_action_dim} → {hidden_dim} → {hidden_dim} → 1")
        
        for i in range(n_agents):
            # Improved networks
            actor = ImprovedActor(obs_dim, action_dim, hidden_dim).to(device)
            target_actor = ImprovedActor(obs_dim, action_dim, hidden_dim).to(device)
            target_actor.load_state_dict(actor.state_dict())
            
            critic = ImprovedCritic(total_obs_dim, total_action_dim, hidden_dim).to(device)
            target_critic = ImprovedCritic(total_obs_dim, total_action_dim, hidden_dim).to(device)
            target_critic.load_state_dict(critic.state_dict())
            
            # Optimizers with weight decay
            actor_optimizer = torch.optim.Adam(
                actor.parameters(), 
                lr=lr_actor,
                weight_decay=1e-5  # L2 regularization
            )
            critic_optimizer = torch.optim.Adam(
                critic.parameters(), 
                lr=lr_critic,
                weight_decay=1e-5
            )
            
            self.actors.append(actor)
            self.critics.append(critic)
            self.target_actors.append(target_actor)
            self.target_critics.append(target_critic)
            self.actor_optimizers.append(actor_optimizer)
            self.critic_optimizers.append(critic_optimizer)
        
        # Replay buffer
        from collections import deque
        import random
        self.replay_buffer = deque(maxlen=buffer_capacity)
        
        # Exploration
        self.noise_scale = 0.2
        self.noise_decay = 0.999  # Faster decay
        self.noise_min = 0.05
        
        # Training stats
        self.update_count = 0
    
    def select_actions(self, observations, explore=True):
        """Select actions for all agents"""
        actions = {}
        
        for i in range(self.n_agents):
            obs_tensor = torch.FloatTensor(observations[i]).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action = self.actors[i](obs_tensor).cpu().numpy()[0]
            
            if explore:
                noise = np.random.normal(0, self.noise_scale, size=action.shape)
                action = np.clip(action + noise, -1.0, 1.0)
            
            actions[i] = action
        
        return actions
    
    def store_transition(self, obs, actions, rewards, next_obs, dones):
        """Store transition in replay buffer"""
        self.replay_buffer.append((obs, actions, rewards, next_obs, dones))
    
    def update(self, batch_size=128):
        """Update with gradient clipping and monitoring"""
        if len(self.replay_buffer) < batch_size:
            return None
        
        # Sample batch
        import random
        batch = random.sample(self.replay_buffer, batch_size)
        
        # Unpack
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
        
        # Convert to tensors
        obs_batch = torch.FloatTensor(np.array(obs_batch)).to(self.device)
        actions_batch = torch.FloatTensor(np.array(actions_batch)).to(self.device)
        rewards_batch = torch.FloatTensor(np.array(rewards_batch)).to(self.device)
        next_obs_batch = torch.FloatTensor(np.array(next_obs_batch)).to(self.device)
        dones_batch = torch.FloatTensor(np.array(dones_batch)).to(self.device)
        
        total_actor_loss = 0
        total_critic_loss = 0
        
        for agent_id in range(self.n_agents):
            # Update Critic
            next_actions = []
            for i in range(self.n_agents):
                next_action = self.target_actors[i](next_obs_batch[:, i])
                next_actions.append(next_action)
            next_actions = torch.stack(next_actions, dim=1)
            
            next_obs_flat = next_obs_batch.reshape(batch_size, -1)
            next_actions_flat = next_actions.reshape(batch_size, -1)
            
            with torch.no_grad():
                target_q = self.target_critics[agent_id](next_obs_flat, next_actions_flat)
                target_q = rewards_batch[:, agent_id].unsqueeze(1) + \
                           self.gamma * target_q * (1 - dones_batch[:, agent_id].unsqueeze(1))
            
            obs_flat = obs_batch.reshape(batch_size, -1)
            actions_flat = actions_batch.reshape(batch_size, -1)
            current_q = self.critics[agent_id](obs_flat, actions_flat)
            
            critic_loss = nn.MSELoss()(current_q, target_q)
            
            self.critic_optimizers[agent_id].zero_grad()
            critic_loss.backward()
            # GRADIENT CLIPPING
            torch.nn.utils.clip_grad_norm_(self.critics[agent_id].parameters(), 0.5)
            self.critic_optimizers[agent_id].step()
            
            # Update Actor
            current_actions = []
            for i in range(self.n_agents):
                if i == agent_id:
                    current_action = self.actors[i](obs_batch[:, i])
                else:
                    with torch.no_grad():
                        current_action = self.actors[i](obs_batch[:, i])
                current_actions.append(current_action)
            current_actions = torch.stack(current_actions, dim=1)
            current_actions_flat = current_actions.reshape(batch_size, -1)
            
            actor_loss = -self.critics[agent_id](obs_flat, current_actions_flat).mean()
            
            self.actor_optimizers[agent_id].zero_grad()
            actor_loss.backward()
            # GRADIENT CLIPPING
            torch.nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), 0.5)
            self.actor_optimizers[agent_id].step()
            
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
        
        # Soft update
        self._soft_update_targets()
        
        # Decay noise
        self.noise_scale = max(self.noise_min, self.noise_scale * self.noise_decay)
        
        self.update_count += 1
        
        return {
            'actor_loss': total_actor_loss / self.n_agents,
            'critic_loss': total_critic_loss / self.n_agents,
            'noise_scale': self.noise_scale,
            'update_count': self.update_count
        }
    
    def _soft_update_targets(self):
        """Soft update target networks"""
        for i in range(self.n_agents):
            for param, target_param in zip(self.actors[i].parameters(), 
                                          self.target_actors[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.critics[i].parameters(), 
                                          self.target_critics[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, path):
        """Save model"""
        checkpoint = {
            'actors': [actor.state_dict() for actor in self.actors],
            'critics': [critic.state_dict() for critic in self.critics],
            'target_actors': [actor.state_dict() for actor in self.target_actors],
            'target_critics': [critic.state_dict() for critic in self.target_critics],
        }
        torch.save(checkpoint, path)
    
    def load(self, path):
        """Load model"""
        checkpoint = torch.load(path)
        for i in range(self.n_agents):
            self.actors[i].load_state_dict(checkpoint['actors'][i])
            self.critics[i].load_state_dict(checkpoint['critics'][i])
            self.target_actors[i].load_state_dict(checkpoint['target_actors'][i])
            self.target_critics[i].load_state_dict(checkpoint['target_critics'][i])


# ============================================================================
# TRAINING WITH EARLY STOPPING
# ============================================================================

def train_improved_100agents(
    n_agents=100,
    n_episodes=1000,
    test_mode=False,  # Set True for quick 100-episode test
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """Train with early stopping and better monitoring"""
    
    if test_mode:
        n_episodes = 100
        print("\n" + "="*70)
        print("⚡ QUICK TEST MODE: 100 episodes")
        print("="*70 + "\n")
    
    # Setup
    results_dir = Path('../results/improved_100agents')
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / 'checkpoints').mkdir(exist_ok=True)
    (results_dir / 'figures').mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n{'='*70}")
    print(f"IMPROVED 100-AGENT TRAINING")
    print(f"Device: {device}")
    print(f"Episodes: {n_episodes}")
    print(f"{'='*70}\n")
    
    # Environment
    env = SmartGridEnv(n_agents=n_agents)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"Observation space: {obs_dim} dimensions")
    print(f"Action space: {action_dim} dimensions")
    print(f"Compression ratio: {obs_dim}:{action_dim} = {obs_dim/action_dim:.1f}:1\n")
    
    # Agent
    maddpg = ImprovedMADDPGAgent(
        n_agents=n_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=512,  # 4x larger than before
        device=device
    )
    
    # Training tracking
    episode_rewards = []
    best_reward = -float('inf')
    best_episode = 0
    patience_counter = 0
    patience_limit = 150  # Stop if no improvement for 150 episodes
    
    print(f"\nStarting training...\n")
    
    for episode in tqdm(range(n_episodes), desc="Training"):
        obs_dict, _ = env.reset()
        episode_reward = 0
        
        for step in range(24):
            actions = maddpg.select_actions(obs_dict, explore=True)
            next_obs, rewards, dones, truncated, info = env.step(actions)
            
            maddpg.store_transition(obs_dict, actions, rewards, next_obs, dones)
            
            if len(maddpg.replay_buffer) >= 256:
                maddpg.update(batch_size=256)
            
            episode_reward += sum(rewards.values())
            obs_dict = next_obs
            
            if dones['__all__']:
                break
        
        episode_rewards.append(episode_reward)
        
        # Check for improvement
        recent_mean = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        
        if recent_mean > best_reward:
            best_reward = recent_mean
            best_episode = episode
            patience_counter = 0
            
            # Save best model
            checkpoint_path = results_dir / 'checkpoints' / f'best_model_{timestamp}.pt'
            maddpg.save(str(checkpoint_path))
        else:
            patience_counter += 1
        
        # Periodic logging
        if (episode + 1) % 50 == 0:
            print(f"\n{'='*70}")
            print(f"Episode {episode + 1}/{n_episodes}")
            print(f"  Recent mean (last 100): {recent_mean:.2f}")
            print(f"  Best reward: {best_reward:.2f} (episode {best_episode})")
            print(f"  Noise scale: {maddpg.noise_scale:.4f}")
            print(f"  Patience: {patience_counter}/{patience_limit}")
            print(f"{'='*70}")
        
        # Early stopping
        if patience_counter >= patience_limit and not test_mode:
            print(f"\n⚠️ Early stopping at episode {episode + 1}")
            print(f"No improvement for {patience_limit} episodes")
            print(f"Best reward: {best_reward:.2f} at episode {best_episode}")
            break
    
    # Save final model
    final_path = results_dir / 'checkpoints' / f'final_model_{timestamp}.pt'
    maddpg.save(str(final_path))
    
    # Plot training curve
    plot_training_results(episode_rewards, best_episode, results_dir, timestamp)
    
    # Test the best model
    print(f"\n{'='*70}")
    print("TESTING BEST MODEL")
    print(f"{'='*70}\n")
    
    best_checkpoint = results_dir / 'checkpoints' / f'best_model_{timestamp}.pt'
    maddpg.load(str(best_checkpoint))
    
    test_rewards = []
    final_batteries = []
    
    for episode in tqdm(range(50), desc="Testing"):
        obs_dict, _ = env.reset()
        episode_reward = 0
        
        for step in range(24):
            actions = maddpg.select_actions(obs_dict, explore=False)
            obs_dict, rewards, dones, truncated, info = env.step(actions)
            episode_reward += sum(rewards.values())
            
            if dones['__all__']:
                break
        
        test_rewards.append(episode_reward)
        final_batteries.append(np.mean(info['battery_levels']))
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Training:")
    print(f"  Best reward: {best_reward:.2f}")
    print(f"  Final reward: {np.mean(episode_rewards[-100:]):.2f}")
    print(f"\nTesting (50 episodes):")
    print(f"  Mean: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    print(f"  Range: [{np.min(test_rewards):.1f}, {np.max(test_rewards):.1f}]")
    print(f"  Battery: {np.mean(final_batteries):.1%} ± {np.std(final_batteries):.1%}")
    print(f"{'='*70}\n")
    
    return maddpg, episode_rewards, test_rewards


def plot_training_results(rewards, best_episode, results_dir, timestamp):
    """Plot training progress"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(rewards, alpha=0.3, color='blue', label='Raw')
    
    window = 50
    if len(rewards) > window:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(rewards)), smoothed, 
               color='blue', linewidth=2, label='Smoothed (50)')
    
    # Mark best episode
    ax.axvline(best_episode, color='green', linestyle='--', 
              linewidth=2, label=f'Best (ep {best_episode})')
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Reward', fontsize=12)
    ax.set_title('Improved 100-Agent Training', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = results_dir / 'figures' / f'training_{timestamp}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Training curve saved: {save_path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true',
                       help='Quick test mode (100 episodes)')
    parser.add_argument('--full', action='store_true',
                       help='Full training (1000 episodes)')
    parser.add_argument('--n_agents', type=int, default=100)
    parser.add_argument('--n_episodes', type=int, default=1000)
    
    args = parser.parse_args()
    
    if args.test:
        print("\n🔬 Running quick test...")
        train_improved_100agents(
            n_agents=args.n_agents,
            n_episodes=100,
            test_mode=True
        )
    elif args.full:
        print("\n🚀 Running full training...")
        train_improved_100agents(
            n_agents=args.n_agents,
            n_episodes=args.n_episodes,
            test_mode=False
        )
    else:
        print("\nUsage:")
        print("  Quick test (100 episodes, ~2 hours):")
        print("    python train_improved_100agents.py --test")
        print("\n  Full training (1000 episodes with early stopping):")
        print("    python train_improved_100agents.py --full")