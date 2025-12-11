import sys
sys.path.append('..')

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.environments.grid_env import SmartGridEnv
from src.agents.ppo_agent import PPOAgent

def train_single_agent(
    n_episodes=1000,
    max_steps=24,
    update_frequency=2048,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """Train a single PPO agent in the grid environment."""
    
    # Create environment (single agent)
    env = SmartGridEnv(n_agents=1)
    
    # Initialize agent
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PPOAgent(obs_dim, action_dim, device=device)
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    step_count = 0
    
    print(f"Training on {device}")
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
    
    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset()
        obs = obs[0]  # Single agent
        episode_reward = 0
        
        for step in range(max_steps):
            # Select action
            action, log_prob, value = agent.select_action(obs)
            
            # Step environment
            actions_dict = {0: action}
            next_obs, rewards, dones, truncated, info = env.step(actions_dict)
            next_obs = next_obs[0]
            reward = rewards[0]
            done = dones[0]
            
            # Store transition
            agent.store_transition(obs, action, log_prob, reward, value, done)
            
            episode_reward += reward
            step_count += 1
            obs = next_obs
            
            # Update policy
            if step_count % update_frequency == 0:
                update_info = agent.update(next_obs)
                if episode % 50 == 0:
                    print(f"\nUpdate at step {step_count}: {update_info}")
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        
        # Log progress
        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"\nEpisode {episode}, Avg Reward (last 50): {avg_reward:.2f}")
    
    # Save model
    agent.save('../results/checkpoints/ppo_single_agent.pt')
    
    # Plot results
    plot_training_results(episode_rewards, episode_lengths)
    
    return agent, episode_rewards

def plot_training_results(rewards, lengths):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Smooth rewards
    window = 50
    smoothed_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
    
    axes[0].plot(rewards, alpha=0.3, label='Raw')
    axes[0].plot(range(window-1, len(rewards)), smoothed_rewards, label='Smoothed')
    axes[0].set_title('Episode Rewards')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(lengths)
    axes[1].set_title('Episode Lengths')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Steps')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('../results/figures/single_agent_training.png')
    plt.show()

if __name__ == '__main__':
    agent, rewards = train_single_agent(n_episodes=1000)