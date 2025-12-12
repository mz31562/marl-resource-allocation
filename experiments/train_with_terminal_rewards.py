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

def train_with_terminal_reward(
    n_agents=5,
    terminal_battery_value=10.0,  
    n_episodes=1000,
    save_name='maddpg_terminal'
):
    """
    Train MADDPG with terminal battery rewards.
    
    Args:
        terminal_battery_value: Reward coefficient for final battery level
                               0.0 = no incentive (current behavior)
                               10.0 = strong incentive (~25-35% reserves)
                               20.0 = very strong incentive (~40-50% reserves)
    """
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training with terminal_battery_value = {terminal_battery_value}")
    print(f"Device: {device}")
    
    env = SmartGridEnv(
        n_agents=n_agents,
        terminal_battery_value=terminal_battery_value
    )
    
    # Initialize agent
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    maddpg = MADDPGAgent(n_agents=n_agents, obs_dim=obs_dim, 
                        action_dim=action_dim, device=device)
    
    # Training metrics
    episode_rewards = []
    episode_individual_rewards = {i: [] for i in range(n_agents)}
    final_battery_levels = []
    
    print(f"\nTraining {n_agents} agents for {n_episodes} episodes...")
    
    best_reward = -float('inf')
    
    for episode in tqdm(range(n_episodes), desc="Training"):
        obs_dict, info = env.reset()
        episode_reward = 0
        individual_rewards = {i: 0 for i in range(n_agents)}
        
        for step in range(24):
            actions = maddpg.select_actions(obs_dict, explore=True)
            next_obs_dict, rewards_dict, dones_dict, truncated, info = env.step(actions)
            maddpg.store_transition(obs_dict, actions, rewards_dict, 
                                   next_obs_dict, dones_dict)
            maddpg.update(batch_size=64)
            
            for i in range(n_agents):
                individual_rewards[i] += rewards_dict[i]
            episode_reward += sum(rewards_dict.values())
            
            obs_dict = next_obs_dict
            
            if dones_dict['__all__']:
                break
        
        episode_rewards.append(episode_reward)
        for i in range(n_agents):
            episode_individual_rewards[i].append(individual_rewards[i])
        final_battery_levels.append(np.mean(info['battery_levels']))
        
        if episode_reward > best_reward:
            best_reward = episode_reward
            maddpg.save(f'../results/checkpoints/{save_name}_best.pt')
        
        if (episode + 1) % 100 == 0:
            recent_reward = np.mean(episode_rewards[-100:])
            recent_battery = np.mean(final_battery_levels[-100:])
            
            recent_terminal_bonus = 0
            if terminal_battery_value > 0:
                recent_terminal_bonus = recent_battery * terminal_battery_value * n_agents
            
            print(f"\nEpisode {episode + 1}/{n_episodes}")
            print(f"  Avg reward (last 100): {recent_reward:.2f}")
            print(f"  Avg final battery: {recent_battery:.3f}")
            print(f"  Expected terminal bonus: {recent_terminal_bonus:.2f}")
            print(f"  Best reward: {best_reward:.2f}")
            print(f"  Noise scale: {maddpg.noise_scale:.4f}")
    
    maddpg.save(f'../results/checkpoints/{save_name}_final.pt')
    
    serializable_metrics = {
        'terminal_battery_value': float(terminal_battery_value),
        'n_agents': int(n_agents),
        'n_episodes': int(n_episodes),
        'episode_rewards': [float(x) for x in episode_rewards],
        'individual_rewards': {
            str(k): [float(x) for x in v] 
            for k, v in episode_individual_rewards.items()
        },
        'final_battery_levels': [float(x) for x in final_battery_levels],
        'best_reward': float(best_reward)
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'../results/metrics_{save_name}_{timestamp}.json', 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    
    print(f"\n✓ Metrics saved to results/metrics_{save_name}_{timestamp}.json")
    
    plot_terminal_reward_results(episode_rewards, final_battery_levels, 
                                terminal_battery_value)
    
    return maddpg, serializable_metrics


def plot_terminal_reward_results(rewards, batteries, beta):
    """Plot training results with focus on battery management."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    window = 50
    
    ax1 = axes[0]
    ax1.plot(rewards, alpha=0.3, color='blue', label='Raw')
    if len(rewards) > window:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(rewards)), smoothed, 
                color='blue', linewidth=2, label=f'Smoothed ({window})')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title(f'Training Progress (β = {beta})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2 = axes[1]
    ax2.plot(batteries, alpha=0.3, color='green', label='Raw')
    if len(batteries) > window:
        smoothed = np.convolve(batteries, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(batteries)), smoothed, 
                color='green', linewidth=2, label=f'Smoothed ({window})')
    ax2.axhline(y=0.0, color='red', linestyle='--', linewidth=1, label='No reserves')
    if beta >= 10.0:
        ax2.axhline(y=0.3, color='orange', linestyle='--', linewidth=1, 
                   alpha=0.5, label='Target: 30%')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Mean Final Battery Level')
    ax2.set_title('Battery Reserve Learning')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'../results/figures/terminal_reward_beta_{beta}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Plots saved to results/figures/terminal_reward_beta_{beta}.png")


def compare_terminal_reward_variants():
    """
    Run ablation study on terminal reward values.
    Compare: β = 0, 5, 10, 20
    """
    
    print("=" * 70)
    print("TERMINAL REWARD ABLATION STUDY")
    print("=" * 70)
    
    variants = [
        (0.0, "no_terminal"),
        (5.0, "low_terminal"),
        (10.0, "medium_terminal"),
        (20.0, "high_terminal")
    ]
    
    results = {}
    
    for beta, name in variants:
        print(f"\n{'='*70}")
        print(f"Training with β = {beta}")
        print(f"{'='*70}\n")
        
        agent, metrics = train_with_terminal_reward(
            n_agents=5,
            terminal_battery_value=beta,
            n_episodes=1000,
            save_name=name
        )
        
        results[beta] = {
            'final_reward': np.mean(metrics['episode_rewards'][-100:]),
            'final_battery': np.mean(metrics['final_battery_levels'][-100:]),
            'agent': agent,
            'metrics': metrics
        }
    
    plot_ablation_comparison(results)
    
    return results


def plot_ablation_comparison(results):
    """Compare different terminal reward values."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    betas = sorted(results.keys())
    
    ax1 = axes[0]
    final_rewards = [results[b]['final_reward'] for b in betas]
    ax1.bar(range(len(betas)), final_rewards, color='blue', alpha=0.7)
    ax1.set_xticks(range(len(betas)))
    ax1.set_xticklabels([f'β={b}' for b in betas])
    ax1.set_ylabel('Mean Episode Reward (last 100)')
    ax1.set_title('Final Performance by Terminal Reward')
    ax1.grid(True, alpha=0.3, axis='y')
    
    ax2 = axes[1]
    final_batteries = [results[b]['final_battery'] for b in betas]
    bars = ax2.bar(range(len(betas)), final_batteries, color='green', alpha=0.7)
    ax2.set_xticks(range(len(betas)))
    ax2.set_xticklabels([f'β={b}' for b in betas])
    ax2.set_ylabel('Mean Final Battery Level')
    ax2.set_title('Battery Reserves by Terminal Reward')
    ax2.axhline(y=0.3, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Target: 30%')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, final_batteries):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. Trade-off analysis
    ax3 = axes[2]
    ax3.scatter(final_batteries, final_rewards, s=200, alpha=0.6, c=betas, cmap='viridis')
    for i, b in enumerate(betas):
        ax3.annotate(f'β={b}', 
                    (final_batteries[i], final_rewards[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold')
    ax3.set_xlabel('Final Battery Level')
    ax3.set_ylabel('Episode Reward')
    ax3.set_title('Reward vs Reserve Trade-off')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/figures/terminal_reward_ablation.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*70)
    print("ABLATION STUDY RESULTS")
    print("="*70)
    print(f"{'Beta':>10} {'Final Reward':>15} {'Final Battery':>15} {'Interpretation':>20}")
    print("-"*70)
    
    interpretations = {
        0.0: "No reserves",
        5.0: "Weak reserves",
        10.0: "Moderate reserves",
        20.0: "Strong reserves"
    }
    
    for b in betas:
        print(f"{b:>10.1f} {results[b]['final_reward']:>15.2f} "
              f"{results[b]['final_battery']:>15.3f} {interpretations[b]:>20}")
    
    print("="*70)


def evaluate_terminal_variants():
    """Evaluate all trained variants side-by-side."""
    from src.agents.maddpg_agent import MADDPGAgent
    from src.environments.grid_env import SmartGridEnv
    
    variants = [
        ('no_terminal', 0.0, 'No Reserves'),
        ('low_terminal', 5.0, 'Weak Reserves'),
        ('medium_terminal', 10.0, 'Moderate Reserves'),
        ('high_terminal', 20.0, 'Strong Reserves')
    ]
    
    results = {}
    print("\nEvaluating trained models...")
    
    for name, beta, label in variants:
        try:
            env = SmartGridEnv(n_agents=5, terminal_battery_value=0.0)
            agent = MADDPGAgent(
                n_agents=5,
                obs_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0]
            )
            agent.load(f'../results/checkpoints/{name}_best.pt')
            
            episode_rewards = []
            final_batteries = []
            
            for ep in range(100):
                obs_dict, _ = env.reset()
                ep_reward = 0
                
                for step in range(24):
                    actions = agent.select_actions(obs_dict, explore=False)
                    next_obs, rewards, dones, _, info = env.step(actions)
                    ep_reward += sum(rewards.values())
                    obs_dict = next_obs
                    
                    if dones['__all__']:
                        break
                
                episode_rewards.append(ep_reward)
                final_batteries.append(np.mean(info['battery_levels']))
            
            results[label] = {
                'reward': np.mean(episode_rewards),
                'reward_std': np.std(episode_rewards),
                'battery': np.mean(final_batteries),
                'battery_std': np.std(final_batteries)
            }
            
            print(f"\n{label} (trained with β={beta}):")
            print(f"  Reward: {results[label]['reward']:.2f} ± {results[label]['reward_std']:.2f}")
            print(f"  Battery: {results[label]['battery']:.3f} ± {results[label]['battery_std']:.3f}")
            
        except FileNotFoundError:
            print(f"Model {name} not found, skipping...")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['single', 'ablation', 'evaluate'], 
                       default='single',
                       help='Run mode: single training, full ablation, or evaluate')
    parser.add_argument('--beta', type=float, default=10.0,  # CHANGED DEFAULT
                       help='Terminal battery value (for single mode)')
    parser.add_argument('--n_agents', type=int, default=5,
                       help='Number of agents')
    parser.add_argument('--n_episodes', type=int, default=1000,
                       help='Number of training episodes')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        # Train single variant
        agent, metrics = train_with_terminal_reward(
            n_agents=args.n_agents,
            terminal_battery_value=args.beta,
            n_episodes=args.n_episodes
        )
        
    elif args.mode == 'ablation':
        # Run full ablation study
        results = compare_terminal_reward_variants()
        
    elif args.mode == 'evaluate':
        # Evaluate all trained models
        results = evaluate_terminal_variants()