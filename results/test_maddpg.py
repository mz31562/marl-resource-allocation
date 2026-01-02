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
import os

def test_maddpg(
    model_path='../results/checkpoints/maddpg_best.pt',
    n_agents=5,
    n_episodes=365,
    max_steps=24,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    save_results=True
):
    """
    Test trained MADDPG agents in the grid environment.
    
    Args:
        model_path: Path to the trained model checkpoint
        n_agents: Number of agents (must match training)
        n_episodes: Number of test episodes (365 for year-long evaluation)
        max_steps: Maximum steps per episode (24 for hourly simulation)
        device: Device to run on ('cuda' or 'cpu')
        save_results: Whether to save test results
    
    Returns:
        Dictionary containing test metrics
    """
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    print(f"Testing MADDPG model from: {model_path}")
    print(f"Device: {device}")
    print(f"Testing for {n_episodes} episodes with {n_agents} agents\n")
    
    # Create environment
    env = SmartGridEnv(n_agents=n_agents)
    
    # Initialize MADDPG agent
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    maddpg = MADDPGAgent(
        n_agents=n_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device
    )
    
    # Load trained model
    maddpg.load(model_path)
    print(f"Model loaded successfully!\n")
    
    # Testing metrics
    episode_rewards = []
    episode_individual_rewards = {i: [] for i in range(n_agents)}
    episode_lengths = []
    episode_details = []  # Store detailed info for each episode
    
    # Run testing
    for episode in tqdm(range(n_episodes), desc="Testing"):
        obs_dict, info = env.reset()
        episode_reward = 0
        individual_rewards = {i: 0 for i in range(n_agents)}
        step_rewards = []
        
        for step in range(max_steps):
            # Select actions WITHOUT exploration
            actions = maddpg.select_actions(obs_dict, explore=False)
            
            # Step environment
            next_obs_dict, rewards_dict, dones_dict, truncated, info = env.step(actions)
            
            # Accumulate rewards
            step_reward = sum(rewards_dict.values())
            step_rewards.append(step_reward)
            
            for i in range(n_agents):
                individual_rewards[i] += rewards_dict[i]
            episode_reward += step_reward
            
            obs_dict = next_obs_dict
            
            if dones_dict['__all__']:
                break
        
        # Store metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        
        for i in range(n_agents):
            episode_individual_rewards[i].append(individual_rewards[i])
        
        # Store detailed episode info
        episode_details.append({
            'episode': episode + 1,
            'total_reward': episode_reward,
            'length': step + 1,
            'step_rewards': step_rewards,
            'individual_rewards': individual_rewards
        })
        
        # Periodic logging
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"\nEpisode {episode + 1}/{n_episodes}")
            print(f"Average Reward (last 50): {avg_reward:.2f}")
            print(f"Current Episode Reward: {episode_reward:.2f}")
    
    # Calculate statistics
    stats = calculate_statistics(episode_rewards, episode_individual_rewards, episode_lengths)
    
    # Print summary
    print_test_summary(stats, n_episodes, n_agents)
    
    # Prepare results dictionary
    results = {
        'model_path': model_path,
        'n_agents': n_agents,
        'n_episodes': n_episodes,
        'max_steps': max_steps,
        'statistics': stats,
        'episode_rewards': episode_rewards,
        'individual_rewards': episode_individual_rewards,
        'episode_lengths': episode_lengths,
        'episode_details': episode_details
    }
    
    # Save results
    if save_results:
        save_test_results(results)
    
    # Plot results
    plot_test_results(episode_rewards, episode_individual_rewards, stats)
    
    return results

def calculate_statistics(episode_rewards, individual_rewards, episode_lengths):
    """Calculate comprehensive statistics from test results."""
    
    stats = {
        'total_reward': {
            'mean': np.mean(episode_rewards),
            'std': np.std(episode_rewards),
            'min': np.min(episode_rewards),
            'max': np.max(episode_rewards),
            'median': np.median(episode_rewards)
        },
        'episode_length': {
            'mean': np.mean(episode_lengths),
            'std': np.std(episode_lengths),
            'min': np.min(episode_lengths),
            'max': np.max(episode_lengths)
        },
        'individual_agents': {}
    }
    
    # Per-agent statistics
    for agent_id, rewards in individual_rewards.items():
        stats['individual_agents'][agent_id] = {
            'mean': np.mean(rewards),
            'std': np.std(rewards),
            'min': np.min(rewards),
            'max': np.max(rewards)
        }
    
    return stats

def print_test_summary(stats, n_episodes, n_agents):
    """Print a comprehensive summary of test results."""
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"\nTotal Episodes Tested: {n_episodes}")
    print(f"Number of Agents: {n_agents}")
    
    print("\n--- Total Reward Statistics ---")
    print(f"Mean:   {stats['total_reward']['mean']:.2f}")
    print(f"Std:    {stats['total_reward']['std']:.2f}")
    print(f"Min:    {stats['total_reward']['min']:.2f}")
    print(f"Max:    {stats['total_reward']['max']:.2f}")
    print(f"Median: {stats['total_reward']['median']:.2f}")
    
    print("\n--- Episode Length Statistics ---")
    print(f"Mean:   {stats['episode_length']['mean']:.2f}")
    print(f"Std:    {stats['episode_length']['std']:.2f}")
    
    print("\n--- Per-Agent Performance ---")
    agent_means = [stats['individual_agents'][i]['mean'] for i in range(n_agents)]
    print(f"Best Agent (avg):  Agent {np.argmax(agent_means)} with {np.max(agent_means):.2f}")
    print(f"Worst Agent (avg): Agent {np.argmin(agent_means)} with {np.min(agent_means):.2f}")
    print(f"Agent Variance:    {np.std(agent_means):.2f}")
    
    print("\n" + "="*60 + "\n")

def save_test_results(results):
    """Save test results to JSON file."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results directory if it doesn't exist
    os.makedirs('../results/test_results', exist_ok=True)
    
    # Prepare serializable version
    serializable_results = {
        'model_path': results['model_path'],
        'n_agents': results['n_agents'],
        'n_episodes': results['n_episodes'],
        'max_steps': results['max_steps'],
        'statistics': results['statistics'],
        'episode_rewards': [float(x) for x in results['episode_rewards']],
        'individual_rewards': {
            str(k): [float(x) for x in v] 
            for k, v in results['individual_rewards'].items()
        },
        'episode_lengths': [int(x) for x in results['episode_lengths']]
    }
    
    # Save to file
    filepath = f'../results/test_results/maddpg_test_{timestamp}.json'
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Test results saved to: {filepath}")

def plot_test_results(episode_rewards, individual_rewards, stats):
    """Create comprehensive visualizations of test results."""
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Episode rewards over time
    ax1 = fig.add_subplot(gs[0, :])
    episodes = range(1, len(episode_rewards) + 1)
    ax1.plot(episodes, episode_rewards, alpha=0.6, linewidth=1, color='blue')
    
    # Add rolling average
    window = 30
    if len(episode_rewards) > window:
        rolling_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window, len(episode_rewards) + 1), rolling_avg, 
                color='red', linewidth=2, label=f'{window}-Episode Moving Average')
    
    # Add mean line
    ax1.axhline(y=stats['total_reward']['mean'], color='green', 
               linestyle='--', linewidth=2, label=f"Mean: {stats['total_reward']['mean']:.2f}")
    
    ax1.set_title('Episode Rewards Over Testing Period (365 Episodes)', 
                 fontsize=14, fontweight='bold')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Reward distribution histogram
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(episode_rewards, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax2.axvline(x=stats['total_reward']['mean'], color='red', 
               linestyle='--', linewidth=2, label='Mean')
    ax2.axvline(x=stats['total_reward']['median'], color='green', 
               linestyle='--', linewidth=2, label='Median')
    ax2.set_title('Distribution of Episode Rewards', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Total Reward')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Box plot of agent performance
    ax3 = fig.add_subplot(gs[1, 1])
    agent_data = [individual_rewards[i] for i in range(len(individual_rewards))]
    bp = ax3.boxplot(agent_data, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax3.set_title('Individual Agent Performance Distribution', 
                 fontsize=12, fontweight='bold')
    ax3.set_xlabel('Agent ID')
    ax3.set_ylabel('Reward')
    ax3.grid(True, alpha=0.3)
    
    # 4. Agent mean rewards comparison
    ax4 = fig.add_subplot(gs[2, 0])
    agent_means = [stats['individual_agents'][i]['mean'] for i in range(len(individual_rewards))]
    agent_ids = range(len(individual_rewards))
    bars = ax4.bar(agent_ids, agent_means, color='skyblue', edgecolor='black')
    
    # Color best and worst agents
    best_idx = np.argmax(agent_means)
    worst_idx = np.argmin(agent_means)
    bars[best_idx].set_color('green')
    bars[worst_idx].set_color('red')
    
    ax4.set_title('Average Reward per Agent', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Agent ID')
    ax4.set_ylabel('Average Reward')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Performance statistics table
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('tight')
    ax5.axis('off')
    
    table_data = [
        ['Metric', 'Value'],
        ['Mean Reward', f"{stats['total_reward']['mean']:.2f}"],
        ['Std Dev', f"{stats['total_reward']['std']:.2f}"],
        ['Min Reward', f"{stats['total_reward']['min']:.2f}"],
        ['Max Reward', f"{stats['total_reward']['max']:.2f}"],
        ['Median Reward', f"{stats['total_reward']['median']:.2f}"],
        ['Avg Episode Length', f"{stats['episode_length']['mean']:.2f}"]
    ]
    
    table = ax5.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax5.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)
    
    # Save figure
    os.makedirs('../results/figures', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'../results/figures/maddpg_test_{timestamp}.png', 
               dpi=300, bbox_inches='tight')
    print(f"Test plots saved to: ../results/figures/maddpg_test_{timestamp}.png")
    plt.show()

if __name__ == '__main__':
    results = test_maddpg(
        model_path='../results/checkpoints/maddpg_20agents_best.pt',  # Update path
        n_agents=20,  
        n_episodes=365,
        max_steps=24,
        save_results=True
    )
    
    print("Testing complete!")