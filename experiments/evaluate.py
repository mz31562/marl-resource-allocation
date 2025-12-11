import sys
sys.path.append('..')

import torch
import numpy as np
from tqdm import tqdm
from src.environments.grid_env import SmartGridEnv
from src.agents.maddpg_agent import MADDPGAgent
from src.agents.ppo_agent import PPOAgent
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def load_trained_maddpg(n_agents, obs_dim, action_dim, checkpoint_path):
    """Load trained MADDPG model."""
    maddpg = MADDPGAgent(n_agents=n_agents, obs_dim=obs_dim, action_dim=action_dim)
    maddpg.load(checkpoint_path)
    return maddpg

def evaluate_policy(maddpg, env, n_episodes=100, deterministic=True):
    """
    Evaluate trained policy.
    
    Returns detailed metrics including:
    - Episode rewards
    - Individual agent rewards
    - Grid stability
    - Battery usage
    - Social welfare
    """
    episode_rewards = []
    individual_rewards = {i: [] for i in range(env.n_agents)}
    grid_penalties = []
    battery_usage = {i: [] for i in range(env.n_agents)}
    final_batteries = {i: [] for i in range(env.n_agents)}
    
    for episode in tqdm(range(n_episodes), desc="Evaluating"):
        obs_dict, info = env.reset()
        episode_reward = 0
        agent_rewards = {i: 0 for i in range(env.n_agents)}
        episode_grid_penalty = 0
        
        episode_battery_history = {i: [] for i in range(env.n_agents)}
        
        for step in range(24):
            # Select actions (no exploration)
            actions = maddpg.select_actions(obs_dict, explore=False)
            
            # Step
            next_obs, rewards, dones, truncated, info = env.step(actions)
            
            # Track metrics
            episode_reward += sum(rewards.values())
            for i in range(env.n_agents):
                agent_rewards[i] += rewards[i]
                episode_battery_history[i].append(info['battery_levels'][i])
            
            if 'grid_penalty' in info:
                episode_grid_penalty += info['grid_penalty']
            
            obs_dict = next_obs
            
            if dones['__all__']:
                break
        
        episode_rewards.append(episode_reward)
        grid_penalties.append(episode_grid_penalty)
        
        for i in range(env.n_agents):
            individual_rewards[i].append(agent_rewards[i])
            battery_usage[i].append(np.std(episode_battery_history[i]))
            final_batteries[i].append(episode_battery_history[i][-1])
    
    return {
        'episode_rewards': episode_rewards,
        'individual_rewards': individual_rewards,
        'grid_penalties': grid_penalties,
        'battery_usage': battery_usage,
        'final_batteries': final_batteries
    }

def calculate_metrics(results):
    """Calculate statistical metrics from results."""
    metrics = {}
    
    # Overall performance
    metrics['mean_reward'] = np.mean(results['episode_rewards'])
    metrics['std_reward'] = np.std(results['episode_rewards'])
    metrics['median_reward'] = np.median(results['episode_rewards'])
    
    # Individual agent performance
    individual_means = [np.mean(results['individual_rewards'][i]) 
                       for i in results['individual_rewards'].keys()]
    metrics['mean_individual_reward'] = np.mean(individual_means)
    metrics['fairness_gini'] = calculate_gini_coefficient(individual_means)
    
    # Grid stability
    metrics['mean_grid_penalty'] = np.mean(results['grid_penalties'])
    metrics['grid_violations'] = np.sum(np.array(results['grid_penalties']) < 0)
    
    # Battery management
    battery_final_means = [np.mean(results['final_batteries'][i]) 
                          for i in results['final_batteries'].keys()]
    metrics['mean_final_battery'] = np.mean(battery_final_means)
    metrics['battery_efficiency'] = 1.0 - np.mean([np.mean(results['battery_usage'][i]) 
                                                    for i in results['battery_usage'].keys()])
    
    return metrics

def calculate_gini_coefficient(values):
    """
    Calculate Gini coefficient for fairness analysis.
    0 = perfect equality, 1 = perfect inequality
    """
    values = np.array(values)
    values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * values)) / (n * np.sum(values)) - (n + 1) / n

def plot_evaluation_results(results, metrics):
    """Create comprehensive evaluation plots."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Reward distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(results['episode_rewards'], bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(metrics['mean_reward'], color='red', linestyle='--', linewidth=2, label='Mean')
    ax1.axvline(metrics['median_reward'], color='green', linestyle='--', linewidth=2, label='Median')
    ax1.set_xlabel('Episode Reward')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Reward Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Individual agent performance
    ax2 = fig.add_subplot(gs[0, 1])
    individual_means = [np.mean(results['individual_rewards'][i]) 
                       for i in results['individual_rewards'].keys()]
    individual_stds = [np.std(results['individual_rewards'][i]) 
                      for i in results['individual_rewards'].keys()]
    agents = list(range(len(individual_means)))
    ax2.bar(agents, individual_means, yerr=individual_stds, capsize=5, alpha=0.7, color='orange')
    ax2.set_xlabel('Agent ID')
    ax2.set_ylabel('Mean Reward')
    ax2.set_title(f'Individual Agent Performance (Gini: {metrics["fairness_gini"]:.3f})')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Grid penalties over episodes
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(results['grid_penalties'], alpha=0.6, color='red')
    ax3.axhline(0, color='black', linestyle='--', linewidth=1)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Grid Penalty')
    ax3.set_title(f'Grid Stability (Violations: {metrics["grid_violations"]})')
    ax3.grid(True, alpha=0.3)
    
    # 4. Battery final levels distribution
    ax4 = fig.add_subplot(gs[1, 0])
    all_final_batteries = []
    for i in results['final_batteries'].keys():
        all_final_batteries.extend(results['final_batteries'][i])
    ax4.hist(all_final_batteries, bins=20, alpha=0.7, color='green', edgecolor='black')
    ax4.axvline(metrics['mean_final_battery'], color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('Final Battery Level')
    ax4.set_ylabel('Frequency')
    ax4.set_title(f'Battery Management (Mean: {metrics["mean_final_battery"]:.2f})')
    ax4.grid(True, alpha=0.3)
    
    # 5. Battery usage variance (efficiency)
    ax5 = fig.add_subplot(gs[1, 1])
    battery_usage_means = [np.mean(results['battery_usage'][i]) 
                          for i in results['battery_usage'].keys()]
    ax5.bar(agents, battery_usage_means, alpha=0.7, color='purple')
    ax5.set_xlabel('Agent ID')
    ax5.set_ylabel('Battery Usage Variance')
    ax5.set_title(f'Battery Efficiency (Score: {metrics["battery_efficiency"]:.3f})')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Reward correlation matrix (cooperation analysis)
    ax6 = fig.add_subplot(gs[1, 2])
    reward_matrix = np.array([results['individual_rewards'][i] 
                             for i in results['individual_rewards'].keys()])
    corr_matrix = np.corrcoef(reward_matrix)
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, ax=ax6, cbar_kws={'label': 'Correlation'})
    ax6.set_title('Agent Reward Correlation')
    ax6.set_xlabel('Agent ID')
    ax6.set_ylabel('Agent ID')
    
    # 7. Cumulative rewards over episodes
    ax7 = fig.add_subplot(gs[2, :])
    cumulative = np.cumsum(results['episode_rewards'])
    ax7.plot(cumulative, linewidth=2, color='blue')
    ax7.set_xlabel('Episode')
    ax7.set_ylabel('Cumulative Reward')
    ax7.set_title('Learning Progress (Cumulative Reward)')
    ax7.grid(True, alpha=0.3)
    
    plt.savefig('../results/figures/evaluation_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.show()

def statistical_comparison(method1_results, method2_results, method1_name, method2_name):
    """
    Perform statistical significance testing.
    Uses Mann-Whitney U test (non-parametric).
    """
    rewards1 = method1_results['episode_rewards']
    rewards2 = method2_results['episode_rewards']
    
    # Mann-Whitney U test
    statistic, p_value = stats.mannwhitneyu(rewards1, rewards2, alternative='two-sided')
    
    print(f"\n=== Statistical Comparison ===")
    print(f"{method1_name} vs {method2_name}")
    print(f"Mean: {np.mean(rewards1):.2f} vs {np.mean(rewards2):.2f}")
    print(f"Median: {np.median(rewards1):.2f} vs {np.median(rewards2):.2f}")
    print(f"Mann-Whitney U statistic: {statistic:.2f}")
    print(f"P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print(f"✓ Difference is statistically significant (p < 0.05)")
        if np.mean(rewards1) > np.mean(rewards2):
            print(f"  {method1_name} performs significantly better")
        else:
            print(f"  {method2_name} performs significantly better")
    else:
        print(f"✗ No statistically significant difference (p >= 0.05)")
    
    return p_value

def main():
    """Main evaluation pipeline."""
    n_agents = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create environment
    env = SmartGridEnv(n_agents=n_agents)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print("Loading trained models...")
    
    # Load MADDPG
    maddpg = load_trained_maddpg(n_agents, obs_dim, action_dim, 
                                '../results/checkpoints/maddpg_best.pt')
    
    print("\nEvaluating MADDPG...")
    maddpg_results = evaluate_policy(maddpg, env, n_episodes=100)
    maddpg_metrics = calculate_metrics(maddpg_results)
    
    print("\n=== MADDPG Performance ===")
    for key, value in maddpg_metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Plot results
    plot_evaluation_results(maddpg_results, maddpg_metrics)
    
    # TODO: Compare with baselines if available
    
    return maddpg_results, maddpg_metrics

if __name__ == '__main__':
    results, metrics = main()