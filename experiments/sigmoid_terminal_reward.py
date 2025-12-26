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

def sigmoid_terminal_reward(battery_level, target=0.35, steepness=15, max_reward=10.0):
    """
    Calculate terminal reward using sigmoid function.
    
    Sigmoid provides diminishing returns - battery savings are valuable up to
    the target point, then plateau. This prevents infinite hoarding.
    
    Args:
        battery_level: Final battery level (0.0 to 1.0)
        target: Midpoint where reward reaches 50% of max (e.g., 0.35 = 35%)
        steepness: Controls how quickly reward saturates (higher = steeper curve)
        max_reward: Maximum possible reward
    
    Returns:
        Reward value between 0 and max_reward
        
    Examples:
        battery=0.0  -> reward ≈ 0.0  (empty battery)
        battery=0.35 -> reward ≈ 5.0  (target - 50% of max)
        battery=0.5  -> reward ≈ 9.5  (good reserves)
        battery=0.8  -> reward ≈ 10.0 (max - not much better than 0.5!)
    """
    # Sigmoid function centered at target
    reward = max_reward / (1 + np.exp(-steepness * (battery_level - target)))
    return reward


def plot_sigmoid_curve(target=0.35, steepness=15, max_reward=10.0):
    """Visualize the sigmoid reward curve."""
    battery_levels = np.linspace(0, 1, 100)
    rewards = [sigmoid_terminal_reward(b, target, steepness, max_reward) 
               for b in battery_levels]
    
    plt.figure(figsize=(10, 6))
    plt.plot(battery_levels, rewards, linewidth=2, color='blue')
    plt.axvline(x=target, color='red', linestyle='--', linewidth=2, 
                label=f'Target: {target*100:.0f}%')
    plt.axhline(y=max_reward/2, color='orange', linestyle='--', 
                linewidth=1, alpha=0.5, label=f'50% Reward')
    
    # Mark key points
    plt.scatter([0, target, 0.5, 1.0], 
               [sigmoid_terminal_reward(x, target, steepness, max_reward) 
                for x in [0, target, 0.5, 1.0]],
               color='red', s=100, zorder=5)
    
    plt.xlabel('Final Battery Level', fontsize=12)
    plt.ylabel('Terminal Reward', fontsize=12)
    plt.title(f'Sigmoid Terminal Reward (target={target}, steepness={steepness})', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('../results/figures/sigmoid_reward_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nSigmoid Reward Examples:")
    print(f"{'Battery Level':<20} {'Reward':<15} {'% of Max':<15}")
    print("-" * 50)
    for b in [0.0, 0.1, 0.2, 0.3, target, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        r = sigmoid_terminal_reward(b, target, steepness, max_reward)
        pct = (r / max_reward) * 100
        print(f"{b*100:>5.0f}%{'':<14} {r:>6.2f}{'':<8} {pct:>5.1f}%")


def train_sigmoid_terminal(
    n_agents=5,
    target=0.35,
    steepness=15,
    max_reward=10.0,
    n_episodes=1000,
    save_name='maddpg_sigmoid'
):
    """
    Train MADDPG with sigmoid terminal battery rewards.
    
    Args:
        target: Target battery level (midpoint of sigmoid, e.g., 0.35 = 35%)
        steepness: How quickly reward saturates (10-20 recommended)
        max_reward: Maximum terminal reward per agent
    """
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nTraining with Sigmoid Terminal Reward")
    print(f"  Target: {target*100:.0f}%")
    print(f"  Steepness: {steepness}")
    print(f"  Max Reward: {max_reward}")
    print(f"  Device: {device}\n")
    
    # Create environment (standard initialization)
    env = SmartGridEnv(n_agents=n_agents)
    
    # Initialize agent
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    maddpg = MADDPGAgent(n_agents=n_agents, obs_dim=obs_dim, 
                        action_dim=action_dim, device=device)
    
    # Training metrics
    episode_rewards = []
    episode_individual_rewards = {i: [] for i in range(n_agents)}
    final_battery_levels = []
    terminal_rewards_received = []
    
    print(f"Training {n_agents} agents for {n_episodes} episodes...")
    
    best_reward = -float('inf')
    
    for episode in tqdm(range(n_episodes), desc="Training"):
        obs_dict, info = env.reset()
        episode_reward = 0
        individual_rewards = {i: 0 for i in range(n_agents)}
        
        for step in range(24):
            actions = maddpg.select_actions(obs_dict, explore=True)
            next_obs_dict, rewards_dict, dones_dict, truncated, info = env.step(actions)
            
            # Add sigmoid terminal reward at the end of episode
            if dones_dict['__all__'] or step == 23:
                for i in range(n_agents):
                    battery_level = info['battery_levels'][i]
                    terminal_reward = sigmoid_terminal_reward(battery_level, target, steepness, max_reward)
                    rewards_dict[i] += terminal_reward
            
            maddpg.store_transition(obs_dict, actions, rewards_dict, 
                                   next_obs_dict, dones_dict)
            maddpg.update(batch_size=64)
            
            for i in range(n_agents):
                individual_rewards[i] += rewards_dict[i]
            episode_reward += sum(rewards_dict.values())
            
            obs_dict = next_obs_dict
            
            if dones_dict['__all__']:
                break
        
        # Record metrics
        avg_battery = np.mean(info['battery_levels'])
        terminal_reward = sigmoid_terminal_reward(avg_battery, target, steepness, max_reward)
        
        episode_rewards.append(episode_reward)
        for i in range(n_agents):
            episode_individual_rewards[i].append(individual_rewards[i])
        final_battery_levels.append(avg_battery)
        terminal_rewards_received.append(terminal_reward * n_agents)
        
        if episode_reward > best_reward:
            best_reward = episode_reward
            maddpg.save(f'../results/checkpoints/{save_name}_best.pt')
        
        if (episode + 1) % 100 == 0:
            recent_reward = np.mean(episode_rewards[-100:])
            recent_battery = np.mean(final_battery_levels[-100:])
            recent_terminal = np.mean(terminal_rewards_received[-100:])
            
            print(f"\nEpisode {episode + 1}/{n_episodes}")
            print(f"  Avg total reward: {recent_reward:.2f}")
            print(f"  Avg final battery: {recent_battery:.3f} ({recent_battery*100:.1f}%)")
            print(f"  Avg terminal reward: {recent_terminal:.2f}")
            print(f"  Best reward: {best_reward:.2f}")
            print(f"  Noise scale: {maddpg.noise_scale:.4f}")
    
    maddpg.save(f'../results/checkpoints/{save_name}_final.pt')
    
    # Save metrics
    serializable_metrics = {
        'reward_type': 'sigmoid',
        'target': float(target),
        'steepness': float(steepness),
        'max_reward': float(max_reward),
        'n_agents': int(n_agents),
        'n_episodes': int(n_episodes),
        'episode_rewards': [float(x) for x in episode_rewards],
        'individual_rewards': {
            str(k): [float(x) for x in v] 
            for k, v in episode_individual_rewards.items()
        },
        'final_battery_levels': [float(x) for x in final_battery_levels],
        'terminal_rewards': [float(x) for x in terminal_rewards_received],
        'best_reward': float(best_reward)
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'../results/metrics_{save_name}_{timestamp}.json', 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    
    print(f"\n✓ Metrics saved to results/metrics_{save_name}_{timestamp}.json")
    
    # Plot results
    plot_sigmoid_training_results(episode_rewards, final_battery_levels, 
                                  terminal_rewards_received, target, steepness)
    
    return maddpg, serializable_metrics


def plot_sigmoid_training_results(rewards, batteries, terminal_rewards, target, steepness):
    """Plot comprehensive training results for sigmoid reward."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    window = 50
    
    # 1. Total rewards over time
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(rewards, alpha=0.3, color='blue', label='Raw')
    if len(rewards) > window:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(rewards)), smoothed, 
                color='blue', linewidth=2, label=f'Smoothed ({window})')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Training Progress: Total Rewards', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Battery levels over time
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(batteries, alpha=0.3, color='green', label='Raw')
    if len(batteries) > window:
        smoothed = np.convolve(batteries, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(batteries)), smoothed, 
                color='green', linewidth=2, label=f'Smoothed ({window})')
    ax2.axhline(y=target, color='red', linestyle='--', linewidth=2, 
                label=f'Target: {target*100:.0f}%')
    ax2.axhline(y=0.0, color='orange', linestyle='--', linewidth=1, 
                alpha=0.5, label='Empty')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Mean Final Battery Level')
    ax2.set_title('Battery Reserve Learning', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Terminal rewards over time
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(terminal_rewards, alpha=0.3, color='purple', label='Raw')
    if len(terminal_rewards) > window:
        smoothed = np.convolve(terminal_rewards, np.ones(window)/window, mode='valid')
        ax3.plot(range(window-1, len(terminal_rewards)), smoothed, 
                color='purple', linewidth=2, label=f'Smoothed ({window})')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Terminal Reward (all agents)')
    ax3.set_title('Terminal Rewards Received', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Distribution of final battery levels
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.hist(batteries, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax4.axvline(x=target, color='red', linestyle='--', linewidth=2, 
                label=f'Target: {target*100:.0f}%')
    ax4.axvline(x=np.mean(batteries), color='blue', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(batteries)*100:.1f}%')
    ax4.set_xlabel('Final Battery Level')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Final Battery Levels', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Sigmoid reward curve with actual data
    ax5 = fig.add_subplot(gs[2, 1])
    battery_range = np.linspace(0, 1, 100)
    reward_curve = [sigmoid_terminal_reward(b, target, steepness, 10.0) for b in battery_range]
    ax5.plot(battery_range, reward_curve, linewidth=2, color='blue', label='Reward Curve')
    
    # Scatter actual achieved battery levels
    # Sample to avoid overcrowding
    sample_indices = np.random.choice(len(batteries), min(500, len(batteries)), replace=False)
    sample_batteries = [batteries[i] for i in sample_indices]
    sample_rewards = [sigmoid_terminal_reward(b, target, steepness, 10.0) for b in sample_batteries]
    ax5.scatter(sample_batteries, sample_rewards, alpha=0.3, s=10, color='red', 
                label='Actual Episodes')
    
    ax5.axvline(x=target, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax5.set_xlabel('Battery Level')
    ax5.set_ylabel('Terminal Reward')
    ax5.set_title('Reward Function vs Actual Performance', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    plt.savefig(f'../results/figures/sigmoid_training_target_{target}_steep_{steepness}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✓ Training plots saved")


def compare_sigmoid_variants():
    """
    Compare different sigmoid configurations:
    - Different targets: 0.25, 0.35, 0.45
    - Different steepness: 10, 15, 20
    """
    
    print("=" * 70)
    print("SIGMOID TERMINAL REWARD COMPARISON")
    print("=" * 70)
    
    # Test different targets with fixed steepness
    target_variants = [
        (0.25, 15, "low_target"),
        (0.35, 15, "medium_target"),
        (0.45, 15, "high_target")
    ]
    
    # Test different steepness with fixed target
    steepness_variants = [
        (0.35, 10, "gentle_curve"),
        (0.35, 15, "medium_curve"),
        (0.35, 20, "steep_curve")
    ]
    
    results = {}
    
    print("\n--- Testing Different Targets ---")
    for target, steepness, name in target_variants:
        print(f"\nTraining with target={target*100:.0f}%, steepness={steepness}")
        agent, metrics = train_sigmoid_terminal(
            n_agents=5,
            target=target,
            steepness=steepness,
            n_episodes=1000,
            save_name=f'sigmoid_{name}'
        )
        
        results[name] = {
            'target': target,
            'steepness': steepness,
            'final_reward': np.mean(metrics['episode_rewards'][-100:]),
            'final_battery': np.mean(metrics['final_battery_levels'][-100:]),
            'metrics': metrics
        }
    
    print("\n--- Testing Different Steepness ---")
    for target, steepness, name in steepness_variants:
        if name in results:
            continue  # Skip if already trained
        print(f"\nTraining with target={target*100:.0f}%, steepness={steepness}")
        agent, metrics = train_sigmoid_terminal(
            n_agents=5,
            target=target,
            steepness=steepness,
            n_episodes=1000,
            save_name=f'sigmoid_{name}'
        )
        
        results[name] = {
            'target': target,
            'steepness': steepness,
            'final_reward': np.mean(metrics['episode_rewards'][-100:]),
            'final_battery': np.mean(metrics['final_battery_levels'][-100:]),
            'metrics': metrics
        }
    
    # Plot comparison
    plot_sigmoid_comparison(results)
    
    return results


def plot_sigmoid_comparison(results):
    """Compare different sigmoid configurations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Final rewards comparison
    ax1 = axes[0, 0]
    names = list(results.keys())
    rewards = [results[n]['final_reward'] for n in names]
    bars = ax1.bar(range(len(names)), rewards, color='blue', alpha=0.7)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.set_ylabel('Mean Reward (last 100)')
    ax1.set_title('Final Performance Comparison', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, rewards):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Final battery levels comparison
    ax2 = axes[0, 1]
    batteries = [results[n]['final_battery'] for n in names]
    targets = [results[n]['target'] for n in names]
    
    bars = ax2.bar(range(len(names)), batteries, color='green', alpha=0.7, label='Achieved')
    ax2.scatter(range(len(names)), targets, color='red', s=100, marker='*', 
               label='Target', zorder=5)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.set_ylabel('Battery Level')
    ax2.set_title('Battery Reserves: Target vs Achieved', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, batteries):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Reward curves comparison
    ax3 = axes[1, 0]
    battery_range = np.linspace(0, 1, 100)
    
    for name in names:
        target = results[name]['target']
        steepness = results[name]['steepness']
        curve = [sigmoid_terminal_reward(b, target, steepness, 10.0) for b in battery_range]
        ax3.plot(battery_range, curve, linewidth=2, label=f'{name}')
    
    ax3.set_xlabel('Battery Level')
    ax3.set_ylabel('Terminal Reward')
    ax3.set_title('Reward Curves Comparison', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary table
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')
    
    table_data = [['Config', 'Target', 'Steep', 'Reward', 'Battery']]
    for name in names:
        table_data.append([
            name[:15],
            f"{results[name]['target']:.2f}",
            f"{results[name]['steepness']:.0f}",
            f"{results[name]['final_reward']:.1f}",
            f"{results[name]['final_battery']:.2f}"
        ])
    
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.15, 0.15, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('../results/figures/sigmoid_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train MADDPG with sigmoid terminal rewards')
    parser.add_argument('--mode', choices=['single', 'compare', 'visualize'], 
                       default='single',
                       help='Run mode')
    parser.add_argument('--target', type=float, default=0.35,
                       help='Target battery level (0.0-1.0)')
    parser.add_argument('--steepness', type=float, default=15,
                       help='Sigmoid steepness (10-20 recommended)')
    parser.add_argument('--max_reward', type=float, default=10.0,
                       help='Maximum terminal reward per agent')
    parser.add_argument('--n_agents', type=int, default=5,
                       help='Number of agents')
    parser.add_argument('--n_episodes', type=int, default=1000,
                       help='Number of training episodes')
    
    args = parser.parse_args()
    
    if args.mode == 'visualize':
        # Just visualize the sigmoid curve
        print("\nVisualizing sigmoid reward curve...")
        plot_sigmoid_curve(args.target, args.steepness, args.max_reward)
        
    elif args.mode == 'single':
        # Train single model
        agent, metrics = train_sigmoid_terminal(
            n_agents=args.n_agents,
            target=args.target,
            steepness=args.steepness,
            max_reward=args.max_reward,
            n_episodes=args.n_episodes
        )
        
    elif args.mode == 'compare':
        # Compare different configurations
        results = compare_sigmoid_variants()