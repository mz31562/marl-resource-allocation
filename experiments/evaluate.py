import sys
sys.path.append('..')

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.environments.dataset_driven_env import DatasetDrivenSmartGridEnv
from src.agents.maddpg_agent import MADDPGAgent


def evaluate_model(
    checkpoint_path,
    dataset_path,
    n_agents=5,
    n_episodes=100
):
    """
    Comprehensive evaluation of trained model.
    """
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"{'='*70}")
    print(f"EVALUATING MODEL")
    print(f"{'='*70}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Device: {device}\n")
    
    # Create test environment
    env = DatasetDrivenSmartGridEnv(
        dataset_path=dataset_path,
        n_agents=n_agents,
        mode='test',
        terminal_battery_value=10.0
    )
    
    # Load trained agent
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    maddpg = MADDPGAgent(
        n_agents=n_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device
    )
    
    print("Loading model...")
    maddpg.load(checkpoint_path)
    print("✓ Model loaded\n")
    
    # Evaluation metrics
    episode_rewards = []
    individual_rewards = {i: [] for i in range(n_agents)}
    final_batteries = []
    grid_violations = 0
    total_grid_penalties = []
    
    print(f"Running {n_episodes} evaluation episodes...\n")
    
    for episode in tqdm(range(n_episodes), desc="Evaluating"):
        obs_dict, info = env.reset()
        episode_reward = 0
        agent_rewards = {i: 0 for i in range(n_agents)}
        episode_grid_penalty = 0
        
        for step in range(24):
            # Select actions (deterministic)
            actions = maddpg.select_actions(obs_dict, explore=False)
            
            # Step environment
            next_obs_dict, rewards_dict, dones_dict, truncated, info = env.step(actions)
            
            # Track metrics
            episode_reward += sum(rewards_dict.values())
            for i in range(n_agents):
                agent_rewards[i] += rewards_dict[i]
            
            if 'grid_penalty' in info and info['grid_penalty'] < 0:
                episode_grid_penalty += info['grid_penalty']
                grid_violations += 1
            
            obs_dict = next_obs_dict
            
            if dones_dict['__all__']:
                break
        
        # Store episode metrics
        episode_rewards.append(episode_reward)
        final_batteries.append(info['avg_battery'])
        total_grid_penalties.append(episode_grid_penalty)
        
        for i in range(n_agents):
            individual_rewards[i].append(agent_rewards[i])
    
    # Calculate statistics
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'median_reward': np.median(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_battery': np.mean(final_batteries),
        'std_battery': np.std(final_batteries),
        'grid_violations': grid_violations,
        'violation_rate': grid_violations / (n_episodes * 24) * 100,
        'mean_grid_penalty': np.mean(total_grid_penalties),
    }
    
    # Individual agent fairness
    individual_means = [np.mean(individual_rewards[i]) for i in range(n_agents)]
    results['agent_fairness_std'] = np.std(individual_means)
    results['agent_fairness_range'] = np.max(individual_means) - np.min(individual_means)
    
    # Print results
    print(f"\n{'='*70}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*70}\n")
    
    print(f"Episode Performance:")
    print(f"  Mean Reward:     {results['mean_reward']:>8.2f} ± {results['std_reward']:.2f}")
    print(f"  Median Reward:   {results['median_reward']:>8.2f}")
    print(f"  Min Reward:      {results['min_reward']:>8.2f}")
    print(f"  Max Reward:      {results['max_reward']:>8.2f}")
    
    print(f"\nBattery Management:")
    print(f"  Mean Final Level: {results['mean_battery']:>7.1%} ± {results['std_battery']:.1%}")
    
    print(f"\nGrid Stability:")
    print(f"  Violations:       {results['grid_violations']:>8d} / {n_episodes * 24}")
    print(f"  Violation Rate:   {results['violation_rate']:>7.2f}%")
    print(f"  Mean Penalty:     {results['mean_grid_penalty']:>8.2f}")
    
    print(f"\nAgent Fairness:")
    print(f"  Std Dev:          {results['agent_fairness_std']:>8.2f}")
    print(f"  Range:            {results['agent_fairness_range']:>8.2f}")
    
    print(f"\n{'='*70}\n")
    
    # Create visualizations
    plot_evaluation_results(
        episode_rewards,
        individual_rewards,
        final_batteries,
        total_grid_penalties
    )
    
    return results


def plot_evaluation_results(episode_rewards, individual_rewards, batteries, penalties):
    """Create comprehensive evaluation plots."""
    
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Reward distribution
    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(episode_rewards, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(np.mean(episode_rewards), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(episode_rewards):.2f}')
    ax1.axvline(np.median(episode_rewards), color='green', linestyle='--',
               linewidth=2, label=f'Median: {np.median(episode_rewards):.2f}')
    ax1.set_xlabel('Episode Reward')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Reward Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Rewards over episodes
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(episode_rewards, alpha=0.6, color='blue')
    ax2.axhline(np.mean(episode_rewards), color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Reward')
    ax2.set_title('Episode Rewards')
    ax2.grid(True, alpha=0.3)
    
    # 3. Individual agent performance
    ax3 = plt.subplot(2, 3, 3)
    individual_means = [np.mean(individual_rewards[i]) for i in individual_rewards.keys()]
    individual_stds = [np.std(individual_rewards[i]) for i in individual_rewards.keys()]
    agents = list(individual_rewards.keys())
    ax3.bar(agents, individual_means, yerr=individual_stds, capsize=3, alpha=0.7, color='orange')
    ax3.set_xlabel('Agent ID')
    ax3.set_ylabel('Mean Reward')
    ax3.set_title('Individual Agent Performance')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Battery level distribution
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(batteries, bins=20, alpha=0.7, color='green', edgecolor='black')
    ax4.axvline(np.mean(batteries), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {np.mean(batteries):.3f}')
    ax4.axvline(0.3, color='orange', linestyle='--', linewidth=1, 
               alpha=0.5, label='Target: 0.30')
    ax4.set_xlabel('Final Battery Level')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Battery Management')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Grid penalties
    ax5 = plt.subplot(2, 3, 5)
    violation_episodes = [i for i, p in enumerate(penalties) if p < 0]
    ax5.scatter(violation_episodes, [penalties[i] for i in violation_episodes], 
               color='red', alpha=0.6, s=30)
    ax5.axhline(0, color='black', linestyle='-', linewidth=1)
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Grid Penalty')
    ax5.set_title(f'Grid Violations ({len(violation_episodes)} episodes)')
    ax5.grid(True, alpha=0.3)
    
    # 6. Cumulative reward
    ax6 = plt.subplot(2, 3, 6)
    cumulative = np.cumsum(episode_rewards)
    ax6.plot(cumulative, linewidth=2, color='purple')
    ax6.set_xlabel('Episode')
    ax6.set_ylabel('Cumulative Reward')
    ax6.set_title('Cumulative Performance')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/figures/evaluation_results.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved to ../results/figures/evaluation_results.png")
    plt.close()


def compare_with_baseline():
    """Compare trained model with random baseline."""
    from src.agents.random_agent import RandomAgent
    
    print(f"\n{'='*70}")
    print("COMPARING WITH BASELINE")
    print(f"{'='*70}\n")
    
    # Test environment
    env = DatasetDrivenSmartGridEnv(
        dataset_path='../data/processed/dataset_20agents_365days.npz',
        n_agents=5,
        mode='test'
    )
    
    # Random agent
    random_rewards = []
    for episode in tqdm(range(100), desc="Random Baseline"):
        obs_dict, _ = env.reset()
        episode_reward = 0
        
        for step in range(24):
            actions = {i: env.action_space.sample() for i in range(20)}
            obs_dict, rewards, dones, _, _ = env.step(actions)
            episode_reward += sum(rewards.values())
        
        random_rewards.append(episode_reward)
    
    print(f"\nBaseline Performance: {np.mean(random_rewards):.2f} ± {np.std(random_rewards):.2f}")
    
    return np.mean(random_rewards)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                    default='../results/checkpoints/maddpg_5agents_test_best.pt',
                    help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str,
                    default='../data/processed/dataset_5agents.npz',
                    help='Path to dataset')
    parser.add_argument('--n_agents', type=int, default=20)
    parser.add_argument('--n_episodes', type=int, default=100)
    parser.add_argument('--compare_baseline', action='store_true',
                       help='Also evaluate random baseline')
    
    args = parser.parse_args()
    
    # Evaluate trained model
    results = evaluate_model(
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset,
        n_agents=args.n_agents,
        n_episodes=args.n_episodes
    )
    
    # Compare with baseline
    if args.compare_baseline:
        baseline_reward = compare_with_baseline()
        improvement = ((results['mean_reward'] - baseline_reward) / abs(baseline_reward)) * 100
        print(f"\nImprovement over baseline: {improvement:+.1f}%")