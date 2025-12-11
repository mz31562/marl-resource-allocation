import sys
sys.path.append('..')

import numpy as np
from tqdm import tqdm
from src.environments.grid_env import SmartGridEnv
import matplotlib.pyplot as plt

def evaluate_random_policy(n_agents=5, n_episodes=100):
    """Evaluate random action baseline."""
    env = SmartGridEnv(n_agents=n_agents)
    episode_rewards = []
    
    for episode in tqdm(range(n_episodes), desc="Random Policy"):
        obs, info = env.reset()
        episode_reward = 0
        
        for step in range(24):
            actions = {i: env.action_space.sample() for i in range(n_agents)}
            next_obs, rewards, dones, truncated, info = env.step(actions)
            episode_reward += sum(rewards.values())
            
            if dones['__all__']:
                break
        
        episode_rewards.append(episode_reward)
    
    return np.mean(episode_rewards), np.std(episode_rewards)

def evaluate_greedy_policy(n_agents=5, n_episodes=100):
    """
    Evaluate greedy heuristic baseline.
    Strategy: Always charge when price is low, discharge when high.
    """
    env = SmartGridEnv(n_agents=n_agents)
    episode_rewards = []
    
    for episode in tqdm(range(n_episodes), desc="Greedy Policy"):
        obs_dict, info = env.reset()
        episode_reward = 0
        
        for step in range(24):
            actions = {}
            
            for i in range(n_agents):
                obs = obs_dict[i]
                battery = obs[0]  # Battery level
                price = obs[2]     # Grid price
                
                # Simple heuristic
                if price < 0.25:  # Low price
                    charge_rate = 0.5 if battery < 0.8 else 0.0
                    grid_interaction = 0.3  # Buy from grid
                elif price > 0.35:  # High price
                    charge_rate = -0.5 if battery > 0.3 else 0.0
                    grid_interaction = -0.3  # Sell to grid
                else:  # Medium price
                    charge_rate = 0.0
                    grid_interaction = 0.0
                
                actions[i] = np.array([charge_rate, grid_interaction])
            
            next_obs, rewards, dones, truncated, info = env.step(actions)
            episode_reward += sum(rewards.values())
            obs_dict = next_obs
            
            if dones['__all__']:
                break
        
        episode_rewards.append(episode_reward)
    
    return np.mean(episode_rewards), np.std(episode_rewards)

def evaluate_centralized_optimal(n_agents=5, n_episodes=100):
    """
    Evaluate centralized controller (upper bound).
    Uses perfect information and coordination.
    """
    env = SmartGridEnv(n_agents=n_agents)
    episode_rewards = []
    
    for episode in tqdm(range(n_episodes), desc="Centralized Optimal"):
        obs_dict, info = env.reset()
        episode_reward = 0
        
        for step in range(24):
            actions = {}
            
            # Get global state
            batteries = [obs_dict[i][0] for i in range(n_agents)]
            avg_battery = np.mean(batteries)
            price = obs_dict[0][2]  # Same for all
            
            for i in range(n_agents):
                battery = batteries[i]
                
                # Coordinated strategy
                if price < 0.25 and avg_battery < 0.7:
                    # Everyone charges when cheap
                    charge_rate = 0.4 if battery < 0.9 else 0.0
                    grid_interaction = 0.2
                elif price > 0.35 and avg_battery > 0.3:
                    # Everyone discharges when expensive
                    charge_rate = -0.4 if battery > 0.2 else 0.0
                    grid_interaction = -0.2
                else:
                    charge_rate = 0.0
                    grid_interaction = 0.0
                
                # Balance load to avoid grid penalty
                total_grid = sum([0.2 if p < 0.25 else -0.2 if p > 0.35 else 0.0 
                                 for p in [price] * n_agents])
                if total_grid > env.grid_capacity / 10:
                    grid_interaction *= 0.5  # Reduce to avoid penalty
                
                actions[i] = np.array([charge_rate, grid_interaction])
            
            next_obs, rewards, dones, truncated, info = env.step(actions)
            episode_reward += sum(rewards.values())
            obs_dict = next_obs
            
            if dones['__all__']:
                break
        
        episode_rewards.append(episode_reward)
    
    return np.mean(episode_rewards), np.std(episode_rewards)

def compare_baselines():
    """Run all baselines and compare."""
    n_agents = 5
    n_episodes = 100
    
    print("Evaluating baselines...")
    
    random_mean, random_std = evaluate_random_policy(n_agents, n_episodes)
    greedy_mean, greedy_std = evaluate_greedy_policy(n_agents, n_episodes)
    optimal_mean, optimal_std = evaluate_centralized_optimal(n_agents, n_episodes)
    
    print("\n=== Baseline Results ===")
    print(f"Random Policy:        {random_mean:.2f} ± {random_std:.2f}")
    print(f"Greedy Heuristic:     {greedy_mean:.2f} ± {greedy_std:.2f}")
    print(f"Centralized Optimal:  {optimal_mean:.2f} ± {optimal_std:.2f}")
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['Random', 'Greedy', 'Centralized\nOptimal']
    means = [random_mean, greedy_mean, optimal_mean]
    stds = [random_std, greedy_std, optimal_std]
    
    bars = ax.bar(methods, means, yerr=stds, capsize=10, color=['red', 'orange', 'green'], alpha=0.7)
    
    ax.set_ylabel('Average Episode Reward', fontsize=12)
    ax.set_title('Baseline Policy Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../results/figures/baseline_comparison.png', dpi=300)
    plt.show()
    
    return {
        'random': (random_mean, random_std),
        'greedy': (greedy_mean, greedy_std),
        'optimal': (optimal_mean, optimal_std)
    }

if __name__ == '__main__':
    baselines = compare_baselines()