import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.animation import FuncAnimation
import networkx as nx

def visualize_episode_heatmap(env_history, save_path=None):
    """
    Create heatmap showing agent actions over an episode.
    
    Args:
        env_history: List of dicts containing step data
    """
    n_agents = len(env_history[0]['battery_levels'])
    n_steps = len(env_history)
    
    # Extract data
    battery_levels = np.array([step['battery_levels'] for step in env_history])
    actions = np.array([step['actions'] for step in env_history])
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 1. Battery levels over time
    im1 = axes[0].imshow(battery_levels.T, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    axes[0].set_ylabel('Agent ID')
    axes[0].set_title('Battery Levels Over Episode')
    plt.colorbar(im1, ax=axes[0], label='Battery Level')
    
    # 2. Charge rates
    charge_rates = actions[:, :, 0]  # First action dimension
    im2 = axes[1].imshow(charge_rates.T, aspect='auto', cmap='RdBu', vmin=-1, vmax=1)
    axes[1].set_ylabel('Agent ID')
    axes[1].set_title('Charge Rates (Red=Discharge, Blue=Charge)')
    plt.colorbar(im2, ax=axes[1], label='Charge Rate')
    
    # 3. Grid interactions
    grid_actions = actions[:, :, 1]  # Second action dimension
    im3 = axes[2].imshow(grid_actions.T, aspect='auto', cmap='PiYG', vmin=-1, vmax=1)
    axes[2].set_xlabel('Time Step (Hour)')
    axes[2].set_ylabel('Agent ID')
    axes[2].set_title('Grid Interactions (Green=Sell, Purple=Buy)')
    plt.colorbar(im3, ax=axes[2], label='Grid Interaction')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_agent_coordination(correlation_matrix, agent_ids=None):
    """
    Visualize coordination between agents.
    
    Args:
        correlation_matrix: Correlation of agent rewards
    """
    if agent_ids is None:
        agent_ids = [f'Agent {i}' for i in range(len(correlation_matrix))]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', 
                cmap='coolwarm', center=0, 
                xticklabels=agent_ids, yticklabels=agent_ids,
                ax=ax, cbar_kws={'label': 'Reward Correlation'})
    
    ax.set_title('Agent Coordination Analysis', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

def visualize_policy_behavior(maddpg, env, n_episodes=5):
    """
    Visualize learned policy behavior across multiple episodes.
    """
    fig, axes = plt.subplots(n_episodes, 3, figsize=(15, 4*n_episodes))
    
    for ep in range(n_episodes):
        obs_dict, info = env.reset()
        
        battery_history = {i: [] for i in range(env.n_agents)}
        charge_history = {i: [] for i in range(env.n_agents)}
        grid_history = {i: [] for i in range(env.n_agents)}
        
        for step in range(24):
            actions = maddpg.select_actions(obs_dict, explore=False)
            next_obs, rewards, dones, truncated, info = env.step(actions)
            
            for i in range(env.n_agents):
                battery_history[i].append(info['battery_levels'][i])
                charge_history[i].append(actions[i][0])
                grid_history[i].append(actions[i][1])
            
            obs_dict = next_obs
            if dones['__all__']:
                break
        
        # Plot for this episode
        for i in range(env.n_agents):
            axes[ep, 0].plot(battery_history[i], label=f'Agent {i}', alpha=0.7)
            axes[ep, 1].plot(charge_history[i], label=f'Agent {i}', alpha=0.7)
            axes[ep, 2].plot(grid_history[i], label=f'Agent {i}', alpha=0.7)
        
        axes[ep, 0].set_ylabel(f'Ep {ep+1}')
        axes[ep, 0].set_title('Battery Levels' if ep == 0 else '')
        axes[ep, 1].set_title('Charge Rates' if ep == 0 else '')
        axes[ep, 2].set_title('Grid Interactions' if ep == 0 else '')
        
        for ax in axes[ep]:
            ax.grid(True, alpha=0.3)
            if ep == 0:
                ax.legend(loc='upper right', fontsize=8)
    
    axes[-1, 0].set_xlabel('Time Step')
    axes[-1, 1].set_xlabel('Time Step')
    axes[-1, 2].set_xlabel('Time Step')
    
    plt.tight_layout()
    plt.savefig('../results/figures/policy_behavior.png', dpi=300)
    plt.show()

def create_learning_animation(rewards_history, save_path='../results/learning_animation.gif'):
    """
    Create animated visualization of learning progress.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    def update(frame):
        ax.clear()
        ax.plot(rewards_history[:frame], color='blue', linewidth=2)
        
        # Add smoothed line
        if frame > 50:
            window = 50
            smoothed = np.convolve(rewards_history[:frame], np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, frame), smoothed, color='red', linewidth=2, label='Smoothed')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Episode Reward')
        ax.set_title(f'Learning Progress (Episode {frame}/{len(rewards_history)})')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(0, len(rewards_history))
        ax.set_ylim(min(rewards_history), max(rewards_history))
    
    anim = FuncAnimation(fig, update, frames=len(rewards_history), interval=50, repeat=True)
    anim.save(save_path, writer='pillow', fps=20)
    plt.close()

def plot_nash_equilibrium_analysis(agent_strategies):
    """
    Analyze if learned strategies form Nash equilibrium.
    
    Args:
        agent_strategies: List of strategy vectors for each agent
    """
    # TODO: Implement Nash equilibrium check
    # This would involve:
    # 1. Fix all agents' strategies except one
    # 2. Check if deviating improves that agent's reward
    # 3. Repeat for all agents
    pass