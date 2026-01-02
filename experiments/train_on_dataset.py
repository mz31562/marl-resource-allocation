import sys
sys.path.append('..')

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.environments.dataset_driven_env import DatasetDrivenSmartGridEnv
from src.agents.maddpg_agent import MADDPGAgent
from src.data.data_generator import SmartGridDataGenerator
import json
from datetime import datetime


def train_on_dataset(
    dataset_path,
    n_agents=5,
    n_episodes=1000,
    terminal_battery_value=10.0,
    save_name='maddpg_dataset'
):
    """Train MADDPG on dataset."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on {device}")
    print(f"Dataset: {dataset_path}")
    
    # Create environment
    env = DatasetDrivenSmartGridEnv(
        dataset_path=dataset_path,
        n_agents=n_agents,
        episode_length=24,
        mode='train',
        terminal_battery_value=terminal_battery_value
    )
    
    # Initialize agent
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
    final_batteries = []
    
    print(f"\nTraining for {n_episodes} episodes...")
    best_reward = -float('inf')
    
    for episode in tqdm(range(n_episodes), desc="Training"):
        obs_dict, info = env.reset()
        episode_reward = 0
        
        for step in range(24):
            actions = maddpg.select_actions(obs_dict, explore=True)
            next_obs_dict, rewards_dict, dones_dict, truncated, info = env.step(actions)
            
            maddpg.store_transition(obs_dict, actions, rewards_dict, next_obs_dict, dones_dict)
            
            # Update every 4 steps with smaller batch (optimized for 100 agents)
            if (step + 1) % 8 == 0 and len(maddpg.replay_buffer) >= 128:
                maddpg.update(batch_size=128)  # Changed from 64
            
            episode_reward += sum(rewards_dict.values())
            obs_dict = next_obs_dict
            
            if dones_dict['__all__']:
                break
        
        episode_rewards.append(episode_reward)
        final_batteries.append(info['avg_battery'])
        
        if episode_reward > best_reward:
            best_reward = episode_reward
            maddpg.save(f'../results/checkpoints/{save_name}_best.pt')
        
        if (episode + 1) % 100 == 0:
            recent_reward = np.mean(episode_rewards[-100:])
            recent_battery = np.mean(final_batteries[-100:])
            print(f"\nEpisode {episode+1}: Reward={recent_reward:.2f}, "
                  f"Battery={recent_battery:.3f}, Best={best_reward:.2f}")
    
    # Save final model
    maddpg.save(f'../results/checkpoints/{save_name}_final.pt')
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("Evaluating on test set...")
    print("="*60)
    
    test_env = DatasetDrivenSmartGridEnv(
        dataset_path=dataset_path,
        n_agents=n_agents,
        mode='test',
        terminal_battery_value=terminal_battery_value
    )
    
    test_rewards = []
    test_batteries = []
    
    for episode in tqdm(range(100), desc="Testing"):
        obs_dict, info = test_env.reset()
        episode_reward = 0
        
        for step in range(24):
            actions = maddpg.select_actions(obs_dict, explore=False)
            next_obs_dict, rewards_dict, dones_dict, truncated, info = test_env.step(actions)
            episode_reward += sum(rewards_dict.values())
            obs_dict = next_obs_dict
            
            if dones_dict['__all__']:
                break
        
        test_rewards.append(episode_reward)
        test_batteries.append(info['avg_battery'])
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Training Performance: {np.mean(episode_rewards[-100:]):.2f} ± {np.std(episode_rewards[-100:]):.2f}")
    print(f"Test Performance: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    print(f"Test Battery Level: {np.mean(test_batteries):.3f} ± {np.std(test_batteries):.3f}")
    print(f"{'='*60}")
    
    # Plot results
    plot_training_results(episode_rewards, test_rewards, save_name)
    
    return maddpg, episode_rewards, test_rewards


def plot_training_results(train_rewards, test_rewards, save_name):
    """Plot training and test results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training curve
    ax1 = axes[0]
    window = 50
    smoothed = np.convolve(train_rewards, np.ones(window)/window, mode='valid')
    ax1.plot(train_rewards, alpha=0.3, color='blue', label='Raw')
    ax1.plot(range(window-1, len(train_rewards)), smoothed, 
            color='blue', linewidth=2, label='Smoothed')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Training Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Test distribution
    ax2 = axes[1]
    ax2.hist(test_rewards, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(np.mean(test_rewards), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(test_rewards):.2f}')
    ax2.set_xlabel('Episode Reward')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Test Performance Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'../results/figures/{save_name}_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Results saved to ../results/figures/{save_name}_results.png")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, 
                       default='../data/processed/smart_grid_dataset_20agents_365days.npz',
                       help='Path to dataset')
    parser.add_argument('--n_agents', type=int, default=20)
    parser.add_argument('--n_episodes', type=int, default=2000)
    parser.add_argument('--terminal_beta', type=float, default=10.0)
    parser.add_argument('--name', type=str, default='maddpg_dataset')
    
    args = parser.parse_args()
    
    # Check if dataset exists, if not generate it
    import os
    if not os.path.exists(args.dataset):
        print(f"Dataset not found at {args.dataset}")
        print("Generating new dataset...")
        
        from src.data.data_generator import SmartGridDataGenerator
        
        generator = SmartGridDataGenerator(
            n_agents=args.n_agents,
            n_days=365,
            seed=42
        )
        data = generator.generate_full_dataset()
        
        # Create directory if needed
        os.makedirs(os.path.dirname(args.dataset), exist_ok=True)
        generator.save_dataset(data, args.dataset)
        generator.visualize_dataset(data, days_to_plot=7)
    
    # Train
    maddpg, train_rewards, test_rewards = train_on_dataset(
        dataset_path=args.dataset,
        n_agents=args.n_agents,
        n_episodes=args.n_episodes,
        terminal_battery_value=args.terminal_beta,
        save_name=args.name
    )
