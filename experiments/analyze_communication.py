"""
Tools to analyze and visualize what agents are communicating.
"""

import sys
sys.path.append('..')

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.environments.grid_env import SmartGridEnv
from src.agents.maddpg_with_communication import CommMADDPGAgent

def analyze_communication_patterns(agent, env, n_episodes=10):
    """
    Analyze what information agents are communicating.
    
    This helps understand:
    - What agents "talk about"
    - Which agents communicate most
    - How messages change based on state
    """
    
    print("Analyzing communication patterns...")
    
    all_messages = []
    all_states = []
    all_actions = []
    
    for episode in range(n_episodes):
        obs_dict, _ = env.reset()
        episode_messages = []
        episode_states = []
        episode_actions = []
        
        for step in range(24):
            # Get current state features
            states = {i: obs_dict[i] for i in range(agent.n_agents)}
            
            # Get communication messages
            messages = agent.get_communication_pattern(obs_dict)
            
            # Get actions
            actions = agent.select_actions(obs_dict, explore=False)
            
            # Store
            episode_messages.append(messages)
            episode_states.append(states)
            episode_actions.append(actions)
            
            # Step
            obs_dict, _, dones, _, _ = env.step(actions)
            
            if dones['__all__']:
                break
        
        all_messages.extend(episode_messages)
        all_states.extend(episode_states)
        all_actions.extend(episode_actions)
    
    return all_messages, all_states, all_actions


def visualize_message_heatmap(messages, n_agents, save_path='../results/figures/message_heatmap.png'):
    """
    Visualize communication messages as a heatmap.
    Shows what each agent is "broadcasting" at each timestep.
    """
    
    # Stack all messages [timesteps, agents, comm_dim]
    message_array = np.array([[messages[t][i] for i in range(n_agents)] 
                             for t in range(len(messages))])
    
    fig, axes = plt.subplots(1, n_agents, figsize=(4*n_agents, 6))
    if n_agents == 1:
        axes = [axes]
    
    for agent_id in range(n_agents):
        agent_messages = message_array[:, agent_id, :]  # [timesteps, comm_dim]
        
        im = axes[agent_id].imshow(agent_messages.T, aspect='auto', cmap='RdBu', 
                                   vmin=-1, vmax=1, interpolation='nearest')
        axes[agent_id].set_title(f'Agent {agent_id} Messages', fontweight='bold')
        axes[agent_id].set_xlabel('Time Step')
        axes[agent_id].set_ylabel('Message Dimension')
        
        plt.colorbar(im, ax=axes[agent_id], label='Message Value')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Message heatmap saved to {save_path}")


def analyze_message_correlation_with_state(messages, states, n_agents):
    """
    Correlate message dimensions with state features.
    This reveals what information agents encode in their messages.
    
    For example:
    - Message dimension 0 might correlate with battery level
    - Message dimension 5 might correlate with price
    """
    
    # Extract state features (assuming standard format)
    state_feature_names = ['battery', 'price', 'demand', 'solar', 'time']
    
    correlations = {agent_id: {} for agent_id in range(n_agents)}
    
    for agent_id in range(n_agents):
        # Get messages and states for this agent
        agent_messages = np.array([messages[t][agent_id] for t in range(len(messages))])
        agent_states = np.array([states[t][agent_id] for t in range(len(states))])
        
        # Calculate correlation between each message dimension and state feature
        for feature_idx, feature_name in enumerate(state_feature_names[:agent_states.shape[1]]):
            feature_values = agent_states[:, feature_idx]
            
            correlations[agent_id][feature_name] = []
            
            for msg_dim in range(agent_messages.shape[1]):
                msg_values = agent_messages[:, msg_dim]
                corr = np.corrcoef(feature_values, msg_values)[0, 1]
                correlations[agent_id][feature_name].append(corr)
    
    return correlations


def plot_message_state_correlation(correlations, n_agents, save_path='../results/figures/message_correlation.png'):
    """
    Plot heatmap showing which message dimensions correlate with which state features.
    """
    
    state_features = list(correlations[0].keys())
    
    fig, axes = plt.subplots(1, n_agents, figsize=(5*n_agents, 5))
    if n_agents == 1:
        axes = [axes]
    
    for agent_id in range(n_agents):
        # Create correlation matrix
        corr_matrix = np.array([correlations[agent_id][feature] 
                               for feature in state_features])
        
        im = axes[agent_id].imshow(corr_matrix, aspect='auto', cmap='coolwarm', 
                                  vmin=-1, vmax=1)
        axes[agent_id].set_title(f'Agent {agent_id} Message Encoding', fontweight='bold')
        axes[agent_id].set_xlabel('Message Dimension')
        axes[agent_id].set_ylabel('State Feature')
        axes[agent_id].set_yticks(range(len(state_features)))
        axes[agent_id].set_yticklabels(state_features)
        
        plt.colorbar(im, ax=axes[agent_id], label='Correlation')
        
        # Annotate high correlations
        for i in range(len(state_features)):
            for j in range(corr_matrix.shape[1]):
                if abs(corr_matrix[i, j]) > 0.5:
                    axes[agent_id].text(j, i, f'{corr_matrix[i, j]:.2f}',
                                       ha='center', va='center', 
                                       color='white' if abs(corr_matrix[i, j]) > 0.7 else 'black',
                                       fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Correlation plot saved to {save_path}")
    
    # Print insights
    print("\nKey Insights:")
    for agent_id in range(n_agents):
        print(f"\nAgent {agent_id}:")
        for feature in state_features:
            max_corr_idx = np.argmax(np.abs(correlations[agent_id][feature]))
            max_corr = correlations[agent_id][feature][max_corr_idx]
            if abs(max_corr) > 0.5:
                print(f"  {feature:>10} → Message dim {max_corr_idx:2d} (corr={max_corr:+.2f})")


def visualize_communication_network(messages, threshold=0.3, save_path='../results/figures/comm_network.png'):
    """
    Visualize agents as a network based on message similarity.
    Agents with similar messages are connected.
    """
    
    n_agents = len(messages[0])
    
    # Calculate average message for each agent
    avg_messages = np.array([np.mean([messages[t][i] for t in range(len(messages))], axis=0)
                            for i in range(n_agents)])
    
    # Calculate pairwise similarities
    similarity_matrix = np.zeros((n_agents, n_agents))
    for i in range(n_agents):
        for j in range(n_agents):
            if i != j:
                sim = np.dot(avg_messages[i], avg_messages[j]) / \
                      (np.linalg.norm(avg_messages[i]) * np.linalg.norm(avg_messages[j]))
                similarity_matrix[i, j] = max(0, sim)  # Only positive correlations
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Position agents in a circle
    angles = np.linspace(0, 2*np.pi, n_agents, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    
    # Draw edges (communication links)
    for i in range(n_agents):
        for j in range(i+1, n_agents):
            if similarity_matrix[i, j] > threshold:
                ax.plot([x[i], x[j]], [y[i], y[j]], 
                       'gray', alpha=similarity_matrix[i, j], 
                       linewidth=similarity_matrix[i, j]*5)
    
    # Draw nodes
    ax.scatter(x, y, s=1000, c='skyblue', edgecolors='black', linewidth=2, zorder=5)
    
    # Label nodes
    for i in range(n_agents):
        ax.text(x[i], y[i], f'A{i}', ha='center', va='center', 
               fontsize=14, fontweight='bold', zorder=6)
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Agent Communication Network\n(Edge thickness = message similarity)', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Communication network saved to {save_path}")


def compare_with_without_communication_decisions(comm_agent, standard_agent, env, n_episodes=5):
    """
    Compare decisions made with vs without communication.
    Shows how communication changes behavior.
    """
    
    print("\nComparing decisions with and without communication...")
    
    decision_differences = []
    
    for episode in range(n_episodes):
        obs_dict, _ = env.reset()
        
        for step in range(24):
            # Get actions with communication
            comm_actions = comm_agent.select_actions(obs_dict, explore=False)
            
            # Get actions without communication
            standard_actions = standard_agent.select_actions(obs_dict, explore=False)
            
            # Calculate difference
            action_diff = np.mean([np.linalg.norm(comm_actions[i] - standard_actions[i]) 
                                  for i in range(comm_agent.n_agents)])
            decision_differences.append(action_diff)
            
            # Step (use comm_agent's action)
            obs_dict, _, dones, _, _ = env.step(comm_actions)
            
            if dones['__all__']:
                break
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(decision_differences, linewidth=2)
    plt.axhline(y=np.mean(decision_differences), color='red', linestyle='--', 
               label=f'Average: {np.mean(decision_differences):.3f}')
    plt.xlabel('Time Step')
    plt.ylabel('Action Difference (L2 norm)')
    plt.title('Impact of Communication on Decisions', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../results/figures/communication_impact.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Average decision difference: {np.mean(decision_differences):.3f}")
    print("Higher values = Communication changes behavior more significantly")


def run_full_analysis(model_path, n_agents=5):
    """
    Run complete communication analysis suite.
    """
    
    print("="*70)
    print("COMMUNICATION ANALYSIS")
    print("="*70)
    
    # Load model
    env = SmartGridEnv(n_agents=n_agents)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = CommMADDPGAgent(
        n_agents=n_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        comm_dim=32
    )
    agent.load(model_path)
    
    # Collect data
    print("\n1. Collecting communication data...")
    messages, states, actions = analyze_communication_patterns(agent, env, n_episodes=10)
    
    # Visualizations
    print("\n2. Creating visualizations...")
    
    print("  - Message heatmap...")
    visualize_message_heatmap(messages, n_agents)
    
    print("  - Message-state correlation...")
    correlations = analyze_message_correlation_with_state(messages, states, n_agents)
    plot_message_state_correlation(correlations, n_agents)
    
    print("  - Communication network...")
    visualize_communication_network(messages)
    
    print("\n" + "="*70)
    print("Analysis complete! Check ../results/figures/ for visualizations.")
    print("="*70)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, 
                       default='../results/checkpoints/comm_maddpg_best.pt')
    parser.add_argument('--n_agents', type=int, default=5)
    
    args = parser.parse_args()
    
    run_full_analysis(args.model_path, args.n_agents)