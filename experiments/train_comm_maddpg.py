import sys
sys.path.append('..')

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.environments.grid_env import SmartGridEnv
from src.agents.maddpg_with_communication import CommMADDPGAgent
import json
from datetime import datetime
import os

def train_comm_maddpg(
    n_agents=5,
    comm_dim=32,
    n_episodes=1000,
    max_steps=24,
    batch_size=64,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    save_name='comm_maddpg'
):
    """
    Train MADDPG with explicit communication.
    
    Args:
        comm_dim: Dimension of communication messages (16-64 recommended)
                  Larger = more expressive, but harder to train
    """
    
    print("="*70)
    print("TRAINING MADDPG WITH EXPLICIT COMMUNICATION")
    print("="*70)
    print(f"Device: {device}")
    print(f"Agents: {n_agents}")
    print(f"Communication dimension: {comm_dim}")
    print(f"Episodes: {n_episodes}\n")
    
    # Create environment
    env = SmartGridEnv(n_agents=n_agents)
    
    # Initialize CommMADDPG
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = CommMADDPGAgent(
        n_agents=n_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        comm_dim=comm_dim,
        device=device
    )
    
    # Training metrics
    episode_rewards = []
    episode_individual_rewards = {i: [] for i in range(n_agents)}
    communication_entropy = []  # Track message diversity
    
    best_reward = -float('inf')
    
    print("\nStarting training...\n")
    
    for episode in tqdm(range(n_episodes), desc="Training"):
        obs_dict, info = env.reset()
        episode_reward = 0
        individual_rewards = {i: 0 for i in range(n_agents)}
        episode_messages = []
        
        for step in range(max_steps):
            # Select actions (with communication!)
            actions = agent.select_actions(obs_dict, explore=True)
            
            # Optional: Log communication patterns
            if episode % 100 == 0 and step == 0:
                messages = agent.get_communication_pattern(obs_dict)
                episode_messages.append(messages)
            
            # Environment step
            next_obs_dict, rewards_dict, dones_dict, truncated, info = env.step(actions)
            
            # Store transition
            agent.store_transition(obs_dict, actions, rewards_dict, next_obs_dict, dones_dict)
            
            # Update networks
            update_info = agent.update(batch_size)
            
            # Accumulate rewards
            for i in range(n_agents):
                individual_rewards[i] += rewards_dict[i]
            episode_reward += sum(rewards_dict.values())
            
            obs_dict = next_obs_dict
            
            if dones_dict['__all__']:
                break
        
        # Store metrics
        episode_rewards.append(episode_reward)
        for i in range(n_agents):
            episode_individual_rewards[i].append(individual_rewards[i])
        
        # Calculate message entropy (diversity measure)
        if episode_messages:
            msg_diversity = calculate_message_diversity(episode_messages[0])
            communication_entropy.append(msg_diversity)
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            os.makedirs('../results/checkpoints', exist_ok=True)
            agent.save(f'../results/checkpoints/{save_name}_best.pt')
        
        # Periodic logging
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_individual = [np.mean(episode_individual_rewards[i][-100:]) 
                            for i in range(n_agents)]
            
            print(f"\nEpisode {episode + 1}/{n_episodes}")
            print(f"  Avg total reward: {avg_reward:.2f}")
            print(f"  Avg individual rewards: {[f'{r:.2f}' for r in avg_individual]}")
            print(f"  Best reward: {best_reward:.2f}")
            print(f"  Noise scale: {agent.noise_scale:.4f}")
            
            if communication_entropy:
                print(f"  Message diversity: {np.mean(communication_entropy[-100:]):.3f}")
    
    # Save final model
    agent.save(f'../results/checkpoints/{save_name}_final.pt')
    
    # Save metrics
    save_metrics({
        'episode_rewards': episode_rewards,
        'individual_rewards': episode_individual_rewards,
        'communication_entropy': communication_entropy,
        'best_reward': best_reward,
        'n_agents': n_agents,
        'comm_dim': comm_dim,
        'n_episodes': n_episodes
    }, save_name)
    
    # Plot results
    plot_comm_training_results(episode_rewards, episode_individual_rewards, 
                               communication_entropy, save_name)
    
    return agent, episode_rewards


def calculate_message_diversity(messages):
    """
    Calculate diversity of messages across agents.
    Higher = more varied communication, Lower = similar messages
    """
    message_vectors = np.array([messages[i] for i in range(len(messages))])
    
    # Calculate pairwise cosine similarities
    similarities = []
    for i in range(len(messages)):
        for j in range(i + 1, len(messages)):
            dot_product = np.dot(message_vectors[i], message_vectors[j])
            norm_i = np.linalg.norm(message_vectors[i])
            norm_j = np.linalg.norm(message_vectors[j])
            
            if norm_i > 0 and norm_j > 0:
                similarity = dot_product / (norm_i * norm_j)
                similarities.append(similarity)
    
    # Diversity = 1 - average similarity
    if similarities:
        avg_similarity = np.mean(similarities)
        diversity = 1 - avg_similarity
        return diversity
    return 0.0


def save_metrics(metrics, save_name):
    """Save training metrics."""
    os.makedirs('../results', exist_ok=True)
    
    serializable = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            serializable[key] = {str(k): [float(x) for x in v] 
                               for k, v in value.items()}
        elif isinstance(value, list):
            serializable[key] = [float(x) for x in value]
        else:
            serializable[key] = float(value) if isinstance(value, (int, float, np.number)) else value
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f'../results/metrics_{save_name}_{timestamp}.json'
    
    with open(filepath, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    print(f"\n✓ Metrics saved to {filepath}")


def plot_comm_training_results(rewards, individual_rewards, comm_entropy, save_name):
    """Plot comprehensive training results with communication analysis."""
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    window = 50
    
    # 1. Total rewards
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(rewards, alpha=0.3, color='blue', label='Raw')
    if len(rewards) > window:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(rewards)), smoothed, 
                color='blue', linewidth=2, label=f'Smoothed ({window})')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Training Progress with Communication', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Individual agent rewards
    ax2 = fig.add_subplot(gs[1, 0])
    for i in individual_rewards.keys():
        rewards_list = individual_rewards[i]
        if len(rewards_list) > window:
            smoothed = np.convolve(rewards_list, np.ones(window)/window, mode='valid')
            ax2.plot(range(window-1, len(rewards_list)), smoothed, 
                    label=f'Agent {i}', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Individual Reward')
    ax2.set_title('Per-Agent Performance', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Communication diversity
    ax3 = fig.add_subplot(gs[1, 1])
    if comm_entropy:
        ax3.plot(comm_entropy, alpha=0.3, color='purple', label='Raw')
        if len(comm_entropy) > window:
            smoothed = np.convolve(comm_entropy, np.ones(window)/window, mode='valid')
            ax3.plot(range(window-1, len(comm_entropy)), smoothed, 
                    color='purple', linewidth=2, label=f'Smoothed ({window})')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Message Diversity')
    ax3.set_title('Communication Diversity Over Training', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.text(0.05, 0.95, 'Higher = More varied messages\nLower = Similar messages', 
            transform=ax3.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 4. Reward distribution
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.hist(rewards, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax4.axvline(x=np.mean(rewards), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(rewards):.2f}')
    ax4.set_xlabel('Episode Reward')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Reward Distribution', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Agent performance comparison
    ax5 = fig.add_subplot(gs[2, 1])
    agent_means = [np.mean(individual_rewards[i]) for i in individual_rewards.keys()]
    agent_ids = list(individual_rewards.keys())
    bars = ax5.bar(agent_ids, agent_means, color='skyblue', edgecolor='black')
    
    # Highlight best and worst
    best_idx = np.argmax(agent_means)
    worst_idx = np.argmin(agent_means)
    bars[best_idx].set_color('green')
    bars[worst_idx].set_color('orange')
    
    ax5.set_xlabel('Agent ID')
    ax5.set_ylabel('Average Reward')
    ax5.set_title('Agent Performance Comparison', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    for i, (agent_id, mean_val) in enumerate(zip(agent_ids, agent_means)):
        ax5.text(agent_id, mean_val, f'{mean_val:.1f}', 
                ha='center', va='bottom', fontsize=9)
    
    os.makedirs('../results/figures', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'../results/figures/{save_name}_training_{timestamp}.png', 
               dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Training plots saved")


def compare_with_without_comm():
    """
    Compare MADDPG with and without communication.
    This is the key experiment to show communication helps!
    """
    from src.agents.maddpg_agent import MADDPGAgent 
    from src.agents.maddpg_with_communication import CommMADDPGAgent
    
    print("="*70)
    print("COMPARISON: MADDPG vs CommMADDPG")
    print("="*70)
    
    n_agents = 5
    n_episodes = 1000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Train standard MADDPG
    print("\n1. Training standard MADDPG (no communication)...")
    env = SmartGridEnv(n_agents=n_agents)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    standard_agent = MADDPGAgent(
        n_agents=n_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device
    )
    
    standard_rewards = train_standard(standard_agent, env, n_episodes)
    
    # Train CommMADDPG
    print("\n2. Training CommMADDPG (with communication)...")
    comm_agent, comm_rewards = train_comm_maddpg(
        n_agents=n_agents,
        comm_dim=32,
        n_episodes=n_episodes,
        save_name='comm_vs_standard_comm'
    )
    
    # Plot comparison
    plot_comparison(standard_rewards, comm_rewards)
    
    return standard_agent, comm_agent


def train_standard(agent, env, n_episodes):
    """Train standard MADDPG for comparison."""
    rewards = []
    
    for episode in tqdm(range(n_episodes), desc="Training Standard MADDPG"):
        obs_dict, _ = env.reset()
        episode_reward = 0
        
        for step in range(24):
            actions = agent.select_actions(obs_dict, explore=True)
            next_obs_dict, rewards_dict, dones_dict, _, _ = env.step(actions)
            agent.store_transition(obs_dict, actions, rewards_dict, next_obs_dict, dones_dict)
            agent.update(batch_size=64)
            
            episode_reward += sum(rewards_dict.values())
            obs_dict = next_obs_dict
            
            if dones_dict['__all__']:
                break
        
        rewards.append(episode_reward)
    
    agent.save('../results/checkpoints/standard_maddpg_comparison.pt')
    return rewards


def plot_comparison(standard_rewards, comm_rewards):
    """Plot side-by-side comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    window = 50
    
    # Learning curves
    ax1 = axes[0]
    
    # Standard MADDPG
    if len(standard_rewards) > window:
        smoothed = np.convolve(standard_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(standard_rewards)), smoothed, 
                label='Standard MADDPG', linewidth=2, color='blue')
    
    # CommMADDPG
    if len(comm_rewards) > window:
        smoothed = np.convolve(comm_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(comm_rewards)), smoothed, 
                label='CommMADDPG', linewidth=2, color='green')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Learning Curves Comparison', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Final performance comparison
    ax2 = axes[1]
    final_window = 100
    
    standard_final = np.mean(standard_rewards[-final_window:])
    comm_final = np.mean(comm_rewards[-final_window:])
    
    bars = ax2.bar(['Standard MADDPG', 'CommMADDPG'], 
                   [standard_final, comm_final],
                   color=['blue', 'green'], alpha=0.7, edgecolor='black')
    
    ax2.set_ylabel('Average Reward (last 100 episodes)')
    ax2.set_title('Final Performance Comparison', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, [standard_final, comm_final]):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Calculate improvement
    improvement = ((comm_final - standard_final) / abs(standard_final)) * 100
    ax2.text(0.5, 0.95, f'Communication Improvement: {improvement:+.1f}%',
            transform=ax2.transAxes, ha='center', va='top', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('../results/figures/maddpg_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n{'='*70}")
    print("COMPARISON RESULTS")
    print(f"{'='*70}")
    print(f"Standard MADDPG:  {standard_final:.2f}")
    print(f"CommMADDPG:       {comm_final:.2f}")
    print(f"Improvement:      {improvement:+.1f}%")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'compare'], default='train')
    parser.add_argument('--n_agents', type=int, default=5)
    parser.add_argument('--comm_dim', type=int, default=32)
    parser.add_argument('--n_episodes', type=int, default=1000)
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # Train CommMADDPG only
        agent, rewards = train_comm_maddpg(
            n_agents=args.n_agents,
            comm_dim=args.comm_dim,
            n_episodes=args.n_episodes
        )
    
    elif args.mode == 'compare':
        # Compare with and without communication
        standard_agent, comm_agent = compare_with_without_comm()