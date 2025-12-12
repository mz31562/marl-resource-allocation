import sys
sys.path.append('..')

import torch
import numpy as np
from tqdm import tqdm
from src.environments.grid_env import SmartGridEnv
from src.agents.maddpg_agent import MADDPGAgent
import matplotlib.pyplot as plt
import json

def ablation_num_agents(agent_counts=[2, 5, 10, 20], n_episodes=500):
    """
    Ablation study: Effect of number of agents on performance.
    
    Research Question: How does system scale with more agents?
    """
    results = {}
    
    for n_agents in agent_counts:
        print(f"\n=== Training with {n_agents} agents ===")
        
        env = SmartGridEnv(n_agents=n_agents)
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        maddpg = MADDPGAgent(n_agents=n_agents, obs_dim=obs_dim, action_dim=action_dim)
        
        episode_rewards = []
        convergence_episode = None
        
        for episode in tqdm(range(n_episodes)):
            obs_dict, info = env.reset()
            episode_reward = 0
            
            for step in range(24):
                actions = maddpg.select_actions(obs_dict, explore=True)
                next_obs, rewards, dones, truncated, info = env.step(actions)
                
                maddpg.store_transition(obs_dict, actions, rewards, next_obs, dones)
                maddpg.update(batch_size=64)
                
                episode_reward += sum(rewards.values())
                obs_dict = next_obs
                
                if dones['__all__']:
                    break
            
            episode_rewards.append(episode_reward)
            
            # Check convergence (90% of last 100 episodes above threshold)
            if convergence_episode is None and len(episode_rewards) >= 100:
                recent_rewards = episode_rewards[-100:]
                if np.mean(recent_rewards) > 0:  # Positive reward threshold
                    convergence_episode = episode
        
        results[n_agents] = {
            'rewards': episode_rewards,
            'final_mean': np.mean(episode_rewards[-50:]),
            'final_std': np.std(episode_rewards[-50:]),
            'convergence_episode': convergence_episode
        }
    
    # Plot results
    plot_ablation_num_agents(results, n_episodes)
    
    return results

def ablation_communication(n_agents=5, n_episodes=500):
    """
    Ablation study: Effect of agent communication.
    
    Research Question: Does observing neighbor actions help?
    """
    pass

def ablation_reward_structure(n_agents=5, n_episodes=500):
    """
    Ablation study: Different reward structures.
    
    Test:
    1. Individual only (no grid penalty)
    2. Collective only (shared reward)
    3. Mixed (current)
    """
    pass

def plot_ablation_num_agents(results, n_episodes):
    """Plot ablation study results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    agent_counts = sorted(results.keys())
    
    ax1 = axes[0, 0]
    for n in agent_counts:
        rewards = results[n]['rewards']
        window = 50
        if len(rewards) > window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(rewards)), smoothed, label=f'{n} agents', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('Learning Curves by Number of Agents')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    means = [results[n]['final_mean'] for n in agent_counts]
    stds = [results[n]['final_std'] for n in agent_counts]
    ax2.bar(range(len(agent_counts)), means, yerr=stds, 
            tick_label=agent_counts, capsize=5, alpha=0.7)
    ax2.set_xlabel('Number of Agents')
    ax2.set_ylabel('Final Mean Reward')
    ax2.set_title('Final Performance')
    ax2.grid(True, alpha=0.3, axis='y')
    
    ax3 = axes[1, 0]
    convergence = [results[n]['convergence_episode'] or n_episodes 
                  for n in agent_counts]
    ax3.plot(agent_counts, convergence, marker='o', linewidth=2, markersize=8)
    ax3.set_xlabel('Number of Agents')
    ax3.set_ylabel('Episodes to Convergence')
    ax3.set_title('Convergence Speed')
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    per_agent_reward = [results[n]['final_mean'] / n for n in agent_counts]
    ax4.plot(agent_counts, per_agent_reward, marker='s', linewidth=2, markersize=8, color='green')
    ax4.set_xlabel('Number of Agents')
    ax4.set_ylabel('Reward per Agent')
    ax4.set_title('Scalability Analysis')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/figures/ablation_num_agents.png', dpi=300)
    plt.show()

def main():
    """Run all ablation studies."""
    print("=== Running Ablation Studies ===")
    
    # 1. Number of agents
    agent_results = ablation_num_agents(agent_counts=[2, 5, 10], n_episodes=500)
    
    # Save results
    with open('../results/ablation_results.json', 'w') as f:
        # Convert to serializable format
        serializable = {
            str(k): {
                'rewards': [float(r) for r in v['rewards']],
                'final_mean': float(v['final_mean']),
                'final_std': float(v['final_std']),
                'convergence_episode': int(v['convergence_episode']) if v['convergence_episode'] else None
            }
            for k, v in agent_results.items()
        }
        json.dump(serializable, f, indent=2)

if __name__ == '__main__':
    main()