import sys
sys.path.append('..')

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.environments.grid_env import SmartGridEnv
from src.agents.maddpg_agent import MADDPGAgent
import json
from datetime import datetime

def train_maddpg(
    n_agents=20,
    n_episodes=2000,
    max_steps=24,
    batch_size=64,
    update_frequency=1,  # Update every step
    device='cuda' if torch.cuda.is_available() else 'cpu',
    save_frequency=100
):
    """Train MADDPG agents in the grid environment."""
    
    # Create environment
    env = SmartGridEnv(n_agents=n_agents)
    
    # Initialize MADDPG
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    maddpg = MADDPGAgent(
        n_agents=n_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device
    )
    
    # Training metrics
    episode_rewards = []
    episode_individual_rewards = {i: [] for i in range(n_agents)}
    episode_lengths = []
    actor_losses = []
    critic_losses = []
    
    print(f"Training MADDPG with {n_agents} agents on {device}")
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
    
    best_reward = -float('inf')
    
    for episode in tqdm(range(n_episodes), desc="Training"):
        obs_dict, info = env.reset()
        episode_reward = 0
        individual_rewards = {i: 0 for i in range(n_agents)}
        
        for step in range(max_steps):
            # Select actions for all agents
            actions = maddpg.select_actions(obs_dict, explore=True)
            
            # Step environment
            next_obs_dict, rewards_dict, dones_dict, truncated, info = env.step(actions)
            
            # Store transition
            maddpg.store_transition(obs_dict, actions, rewards_dict, next_obs_dict, dones_dict)
            
            # Update networks
            if step % update_frequency == 0:
                update_info = maddpg.update(batch_size)
                if update_info is not None:
                    actor_losses.append(update_info['actor_loss'])
                    critic_losses.append(update_info['critic_loss'])
            
            # Accumulate rewards
            for i in range(n_agents):
                individual_rewards[i] += rewards_dict[i]
            episode_reward += sum(rewards_dict.values())
            
            obs_dict = next_obs_dict
            
            if dones_dict['__all__']:
                break
        
        # Store metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        for i in range(n_agents):
            episode_individual_rewards[i].append(individual_rewards[i])
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            maddpg.save('../results/checkpoints/maddpg_best.pt')
        
        # Periodic logging
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_individual = [np.mean(episode_individual_rewards[i][-50:]) for i in range(n_agents)]
            print(f"\nEpisode {episode + 1}")
            print(f"Avg Total Reward (last 50): {avg_reward:.2f}")
            print(f"Avg Individual Rewards: {[f'{r:.2f}' for r in avg_individual]}")
            print(f"Best Reward: {best_reward:.2f}")
            print(f"Noise Scale: {maddpg.noise_scale:.4f}")
        
        # Save checkpoint
        if (episode + 1) % save_frequency == 0:
            maddpg.save(f'../results/checkpoints/maddpg_episode_{episode+1}.pt')
    
    # Save final model
    maddpg.save('../results/checkpoints/maddpg_final.pt')
    
    # Save training metrics
    save_metrics({
        'episode_rewards': episode_rewards,
        'individual_rewards': episode_individual_rewards,
        'episode_lengths': episode_lengths,
        'actor_losses': actor_losses,
        'critic_losses': critic_losses,
        'best_reward': best_reward,
        'n_agents': n_agents,
        'n_episodes': n_episodes
    })
    
    # Plot results
    plot_training_results(episode_rewards, episode_individual_rewards, actor_losses, critic_losses)
    
    return maddpg, episode_rewards

def save_metrics(metrics):
    """Save training metrics to file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            serializable_metrics[key] = {k: [float(x) for x in v] for k, v in value.items()}
        elif isinstance(value, list):
            serializable_metrics[key] = [float(x) for x in value]
        else:
            serializable_metrics[key] = float(value) if isinstance(value, (np.floating, np.integer)) else value
    
    with open(f'../results/metrics_maddpg_{timestamp}.json', 'w') as f:
        json.dump(serializable_metrics, f, indent=2)

def plot_training_results(total_rewards, individual_rewards, actor_losses, critic_losses):
    """Plot comprehensive training curves."""
    fig = plt.figure(figsize=(16, 10))
    
    # Create grid
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Total rewards
    ax1 = fig.add_subplot(gs[0, :])
    window = 50
    if len(total_rewards) > window:
        smoothed = np.convolve(total_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(total_rewards, alpha=0.3, label='Raw', color='blue')
        ax1.plot(range(window-1, len(total_rewards)), smoothed, label='Smoothed', color='blue', linewidth=2)
    else:
        ax1.plot(total_rewards, label='Total Reward', color='blue')
    ax1.set_title('Total Episode Rewards (All Agents)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Individual agent rewards
    ax2 = fig.add_subplot(gs[1, :])
    for i, rewards in individual_rewards.items():
        if len(rewards) > window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax2.plot(range(window-1, len(rewards)), smoothed, label=f'Agent {i}', linewidth=2)
        else:
            ax2.plot(rewards, label=f'Agent {i}')
    ax2.set_title('Individual Agent Rewards', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Reward')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Actor loss
    ax3 = fig.add_subplot(gs[2, 0])
    if len(actor_losses) > window:
        smoothed = np.convolve(actor_losses, np.ones(window)/window, mode='valid')
        ax3.plot(actor_losses, alpha=0.3, color='orange')
        ax3.plot(range(window-1, len(actor_losses)), smoothed, color='orange', linewidth=2)
    else:
        ax3.plot(actor_losses, color='orange')
    ax3.set_title('Actor Loss', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Update Step')
    ax3.set_ylabel('Loss')
    ax3.grid(True, alpha=0.3)
    
    # 4. Critic loss
    ax4 = fig.add_subplot(gs[2, 1])
    if len(critic_losses) > window:
        smoothed = np.convolve(critic_losses, np.ones(window)/window, mode='valid')
        ax4.plot(critic_losses, alpha=0.3, color='green')
        ax4.plot(range(window-1, len(critic_losses)), smoothed, color='green', linewidth=2)
    else:
        ax4.plot(critic_losses, color='green')
    ax4.set_title('Critic Loss', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Update Step')
    ax4.set_ylabel('Loss')
    ax4.grid(True, alpha=0.3)
    
    plt.savefig('../results/figures/maddpg_training.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    maddpg, rewards = train_maddpg(
        n_agents=20,
        n_episodes=2000,
        batch_size=64
    )