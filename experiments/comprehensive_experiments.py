"""
Progressive Scaling Experiments for Smart Grid MARL
Matches the research narrative: 5 → 20 → 100 agents

Usage:
    python comprehensive_experiments.py --progressive
    python comprehensive_experiments.py --quick_test
"""

import sys
sys.path.append('..')

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os
from datetime import datetime
from pathlib import Path

# Import project modules
from src.environments.grid_env import SmartGridEnv
from src.agents.maddpg_agent import MADDPGAgent


class ProgressiveExperimentRunner:
    """Runs experiments progressively from 5 → 20 → 100 agents"""
    
    def __init__(self, results_dir='../results/progressive'):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.results_dir / 'figures').mkdir(exist_ok=True)
        (self.results_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.results_dir / 'data').mkdir(exist_ok=True)
        (self.results_dir / 'logs').mkdir(exist_ok=True)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Results storage
        self.results = {
            'baselines': {},
            'scaling': {}
        }
        
        self.log(f"\n{'='*70}")
        self.log(f"PROGRESSIVE SCALING EXPERIMENTS")
        self.log(f"Device: {self.device}")
        self.log(f"Results: {self.results_dir}")
        self.log(f"{'='*70}\n")
    
    def log(self, message):
        """Log message to console and file"""
        print(message)
        log_file = self.results_dir / 'logs' / f'experiment_log_{self.timestamp}.txt'
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")
    
    # ========================================================================
    # BASELINES (Chapter 1-2)
    # ========================================================================
    
    def run_baselines(self, n_agents=5, n_episodes=100):
        """Run random and greedy baselines"""
        self.log(f"\n{'='*70}")
        self.log(f"CHAPTER 1-2: BASELINE EXPERIMENTS ({n_agents} agents)")
        self.log(f"{'='*70}\n")
        
        # Random baseline
        self.log("Running Random Policy...")
        random_results = self._run_random_baseline(n_agents, n_episodes)
        
        # Greedy baseline
        self.log("\nRunning Greedy Heuristic...")
        greedy_results = self._run_greedy_baseline(n_agents, n_episodes)
        
        # Store results
        self.results['baselines'][f'{n_agents}agents'] = {
            'random': random_results,
            'greedy': greedy_results
        }
        
        # Compare baselines
        self.log(f"\n{'='*70}")
        self.log("BASELINE COMPARISON")
        self.log(f"{'='*70}")
        self.log(f"Random:  {random_results['mean_reward']:>6.2f} ± {random_results['std_reward']:.2f}")
        self.log(f"Greedy:  {greedy_results['mean_reward']:>6.2f} ± {greedy_results['std_reward']:.2f}")
        self.log(f"Improvement: {greedy_results['mean_reward'] - random_results['mean_reward']:>+6.2f}")
        
        # Plot comparison
        self._plot_baseline_comparison(random_results, greedy_results, n_agents)
        
        return random_results, greedy_results
    
    def _run_random_baseline(self, n_agents, n_episodes):
        """Random policy"""
        env = SmartGridEnv(n_agents=n_agents)
        episode_rewards = []
        grid_violations = 0
        total_steps = 0
        final_batteries = []
        
        for episode in tqdm(range(n_episodes), desc="Random Policy"):
            obs_dict, _ = env.reset()
            episode_reward = 0
            
            for step in range(24):
                actions = {i: env.action_space.sample() for i in range(n_agents)}
                obs_dict, rewards, dones, truncated, info = env.step(actions)
                
                episode_reward += sum(rewards.values())
                total_steps += 1
                
                if 'grid_penalty' in info and info['grid_penalty'] < 0:
                    grid_violations += 1
                
                if dones['__all__']:
                    break
            
            episode_rewards.append(episode_reward)
            final_batteries.append(np.mean(info['battery_levels']))
        
        return {
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'grid_violations': grid_violations,
            'violation_rate': float(grid_violations / total_steps * 100),
            'mean_battery': float(np.mean(final_batteries)),
            'std_battery': float(np.std(final_batteries))
        }
    
    def _run_greedy_baseline(self, n_agents, n_episodes):
        """Greedy heuristic"""
        env = SmartGridEnv(n_agents=n_agents)
        episode_rewards = []
        grid_violations = 0
        total_steps = 0
        final_batteries = []
        
        for episode in tqdm(range(n_episodes), desc="Greedy Policy"):
            obs_dict, _ = env.reset()
            episode_reward = 0
            
            for step in range(24):
                actions = {}
                
                for i in range(n_agents):
                    obs = obs_dict[i]
                    battery = obs[0]
                    price = obs[2]
                    
                    if price < 0.25:
                        charge_rate = 0.5 if battery < 0.8 else 0.0
                        grid_interaction = 0.3
                    elif price > 0.35:
                        charge_rate = -0.5 if battery > 0.3 else 0.0
                        grid_interaction = -0.3
                    else:
                        charge_rate = 0.0
                        grid_interaction = 0.0
                    
                    actions[i] = np.array([charge_rate, grid_interaction])
                
                obs_dict, rewards, dones, truncated, info = env.step(actions)
                episode_reward += sum(rewards.values())
                total_steps += 1
                
                if 'grid_penalty' in info and info['grid_penalty'] < 0:
                    grid_violations += 1
                
                if dones['__all__']:
                    break
            
            episode_rewards.append(episode_reward)
            final_batteries.append(np.mean(info['battery_levels']))
        
        return {
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'grid_violations': grid_violations,
            'violation_rate': float(grid_violations / total_steps * 100),
            'mean_battery': float(np.mean(final_batteries)),
            'std_battery': float(np.std(final_batteries))
        }
    
    def _plot_baseline_comparison(self, random_results, greedy_results, n_agents):
        """Plot baseline comparison"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        methods = ['Random', 'Greedy']
        means = [random_results['mean_reward'], greedy_results['mean_reward']]
        stds = [random_results['std_reward'], greedy_results['std_reward']]
        
        bars = ax.bar(methods, means, yerr=stds, capsize=10, 
                     color=['red', 'orange'], alpha=0.7)
        
        ax.set_ylabel('Mean Episode Reward', fontsize=12)
        ax.set_title(f'Baseline Comparison ({n_agents} Agents)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean:.1f}', ha='center', va='bottom', 
                   fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        save_path = self.results_dir / 'figures' / f'baseline_comparison_{n_agents}agents.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.log(f"✓ Baseline plot saved: {save_path.name}")
    
    # ========================================================================
    # PROGRESSIVE TRAINING (Chapter 3-7)
    # ========================================================================
    
    def run_progressive_training(self, agent_counts=[5, 20, 100], episodes_per_scale=None, quick_test=False):
        """Train progressively: 5 → 20 → 100 agents"""
        
        if quick_test:
            episodes_per_scale = {5: 500, 20: 800, 100: 1000}
            self.log("⚡ Quick test mode enabled")
        elif episodes_per_scale is None:
            episodes_per_scale = {5: 1000, 20: 1500, 100: 2000}
        
        self.log(f"\n{'='*70}")
        self.log(f"PROGRESSIVE TRAINING: {' → '.join(map(str, agent_counts))} agents")
        self.log(f"{'='*70}\n")
        
        for n_agents in agent_counts:
            n_episodes = episodes_per_scale.get(n_agents, 2000)
            
            self.log(f"\n{'='*70}")
            self.log(f"TRAINING: {n_agents} AGENTS ({n_episodes} episodes)")
            self.log(f"{'='*70}\n")
            
            # Train
            maddpg, train_results = self._train_maddpg(n_agents, n_episodes)
            
            # Test
            test_results = self._test_maddpg(maddpg, n_agents, n_episodes=100)
            
            # Analyze
            fairness_results = self._analyze_fairness(maddpg, n_agents, n_episodes=50)
            
            # Store
            self.results['scaling'][f'{n_agents}agents'] = {
                'training': train_results,
                'testing': test_results,
                'fairness': fairness_results
            }
            
            # Summary
            self._print_scale_summary(n_agents, train_results, test_results, fairness_results)
        
        # Create comparative analysis
        self._create_scaling_analysis(agent_counts)
        
        return self.results['scaling']
    
    def _train_maddpg(self, n_agents, n_episodes):
        """Train MADDPG for given agent count"""
        env = SmartGridEnv(n_agents=n_agents)
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        maddpg = MADDPGAgent(
            n_agents=n_agents,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=self.device
        )
        
        episode_rewards = []
        convergence_episode = None
        best_reward = -float('inf')
        
        self.log(f"Obs dim: {obs_dim}, Action dim: {action_dim}")
        self.log(f"Starting training on {self.device}...\n")
        
        for episode in tqdm(range(n_episodes), desc=f"Training {n_agents} agents"):
            obs_dict, _ = env.reset()
            episode_reward = 0
            
            for step in range(24):
                actions = maddpg.select_actions(obs_dict, explore=True)
                next_obs, rewards, dones, truncated, info = env.step(actions)
                
                maddpg.store_transition(obs_dict, actions, rewards, next_obs, dones)
                
                if len(maddpg.replay_buffer) >= 128:
                    maddpg.update(batch_size=128)
                
                episode_reward += sum(rewards.values())
                obs_dict = next_obs
                
                if dones['__all__']:
                    break
            
            episode_rewards.append(episode_reward)
            
            # Track convergence
            if len(episode_rewards) >= 100:
                recent_mean = np.mean(episode_rewards[-100:])
                recent_std = np.std(episode_rewards[-100:])
                
                if convergence_episode is None and recent_std < 50:
                    convergence_episode = episode
                    self.log(f"\n✓ Convergence at episode {episode} (std: {recent_std:.2f})")
                
                # Save best
                if recent_mean > best_reward:
                    best_reward = recent_mean
                    checkpoint_path = self.results_dir / 'checkpoints' / f'maddpg_{n_agents}agents_best.pt'
                    maddpg.save(str(checkpoint_path))
            
            # Periodic logging
            if (episode + 1) % 200 == 0:
                recent_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
                self.log(f"\nEpisode {episode + 1}/{n_episodes}")
                self.log(f"  Mean (last 100): {recent_reward:.2f}")
                self.log(f"  Noise: {maddpg.noise_scale:.4f}")
                self.log(f"  Best: {best_reward:.2f}")
        
        # Save final
        final_path = self.results_dir / 'checkpoints' / f'maddpg_{n_agents}agents_final.pt'
        maddpg.save(str(final_path))
        
        results = {
            'n_agents': n_agents,
            'n_episodes': n_episodes,
            'episode_rewards': [float(r) for r in episode_rewards],
            'convergence_episode': convergence_episode,
            'final_mean': float(np.mean(episode_rewards[-100:])),
            'final_std': float(np.std(episode_rewards[-100:])),
            'best_reward': float(best_reward),
            'checkpoint_path': str(final_path)
        }
        
        # Plot
        self._plot_training_curve(episode_rewards, n_agents)
        
        return maddpg, results
    
    def _test_maddpg(self, maddpg, n_agents, n_episodes=100):
        """Test trained model"""
        self.log(f"\nTesting {n_agents} agents...")
        
        env = SmartGridEnv(n_agents=n_agents)
        test_rewards = []
        final_batteries = []
        
        for episode in tqdm(range(n_episodes), desc=f"Testing {n_agents} agents"):
            obs_dict, _ = env.reset()
            episode_reward = 0
            
            for step in range(24):
                actions = maddpg.select_actions(obs_dict, explore=False)
                obs_dict, rewards, dones, truncated, info = env.step(actions)
                episode_reward += sum(rewards.values())
                
                if dones['__all__']:
                    break
            
            test_rewards.append(episode_reward)
            final_batteries.append(np.mean(info['battery_levels']))
        
        results = {
            'mean_reward': float(np.mean(test_rewards)),
            'std_reward': float(np.std(test_rewards)),
            'median_reward': float(np.median(test_rewards)),
            'min_reward': float(np.min(test_rewards)),
            'max_reward': float(np.max(test_rewards)),
            'test_rewards': [float(r) for r in test_rewards],
            'mean_battery': float(np.mean(final_batteries)),
            'std_battery': float(np.std(final_batteries))
        }
        
        # Plot
        self._plot_test_distribution(test_rewards, n_agents)
        
        return results
    
    def _analyze_fairness(self, maddpg, n_agents, n_episodes=50):
        """Analyze fairness across agents"""
        self.log(f"\nAnalyzing fairness for {n_agents} agents...")
        
        env = SmartGridEnv(n_agents=n_agents)
        individual_rewards = {i: [] for i in range(n_agents)}
        
        for episode in tqdm(range(n_episodes), desc=f"Fairness {n_agents} agents"):
            obs_dict, _ = env.reset()
            episode_individual = {i: 0 for i in range(n_agents)}
            
            for step in range(24):
                actions = maddpg.select_actions(obs_dict, explore=False)
                obs_dict, rewards, dones, truncated, info = env.step(actions)
                
                for i in range(n_agents):
                    episode_individual[i] += rewards[i]
                
                if dones['__all__']:
                    break
            
            for i in range(n_agents):
                individual_rewards[i].append(episode_individual[i])
        
        agent_means = [np.mean(individual_rewards[i]) for i in range(n_agents)]
        
        return {
            'agent_means': [float(m) for m in agent_means],
            'fairness_std': float(np.std(agent_means)),
            'fairness_range': float(np.max(agent_means) - np.min(agent_means)),
            'min_agent_reward': float(np.min(agent_means)),
            'max_agent_reward': float(np.max(agent_means))
        }
    
    def _plot_training_curve(self, rewards, n_agents):
        """Plot training progress"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(rewards, alpha=0.3, color='blue', label='Raw')
        
        window = 50
        if len(rewards) > window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(rewards)), smoothed, 
                   color='blue', linewidth=2, label=f'Smoothed ({window})')
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Total Reward', fontsize=12)
        ax.set_title(f'Training Progress ({n_agents} Agents)', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.results_dir / 'figures' / f'training_{n_agents}agents.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.log(f"✓ Training curve saved: {save_path.name}")
    
    def _plot_test_distribution(self, rewards, n_agents):
        """Plot test distribution"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(rewards, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax.axvline(np.mean(rewards), color='red', linestyle='--', 
                  linewidth=2, label=f'Mean: {np.mean(rewards):.2f}')
        
        ax.set_xlabel('Episode Reward', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Test Distribution ({n_agents} Agents)', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.results_dir / 'figures' / f'test_distribution_{n_agents}agents.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.log(f"✓ Test distribution saved: {save_path.name}")
    
    def _print_scale_summary(self, n_agents, train_results, test_results, fairness_results):
        """Print summary for this scale"""
        self.log(f"\n{'='*70}")
        self.log(f"SUMMARY: {n_agents} AGENTS")
        self.log(f"{'='*70}")
        self.log(f"Training:")
        self.log(f"  Final Mean:    {train_results['final_mean']:>8.2f} ± {train_results['final_std']:.2f}")
        self.log(f"  Best Reward:   {train_results['best_reward']:>8.2f}")
        self.log(f"  Convergence:   {train_results['convergence_episode'] or 'N/A'}")
        self.log(f"\nTesting:")
        self.log(f"  Mean:          {test_results['mean_reward']:>8.2f} ± {test_results['std_reward']:.2f}")
        self.log(f"  Range:         [{test_results['min_reward']:.1f}, {test_results['max_reward']:.1f}]")
        self.log(f"  Battery:       {test_results['mean_battery']:>7.1%} ± {test_results['std_battery']:.1%}")
        self.log(f"\nFairness:")
        self.log(f"  Std Across:    {fairness_results['fairness_std']:>8.3f}")
        self.log(f"  Range:         {fairness_results['fairness_range']:>8.2f}")
        self.log(f"{'='*70}\n")
    
    def _create_scaling_analysis(self, agent_counts):
        """Create comparative scaling analysis"""
        self.log(f"\n{'='*70}")
        self.log("SCALING ANALYSIS")
        self.log(f"{'='*70}\n")
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        # Extract data
        train_means = []
        test_means = []
        convergence_eps = []
        fairness_stds = []
        per_agent_rewards = []
        
        for n in agent_counts:
            key = f'{n}agents'
            if key in self.results['scaling']:
                data = self.results['scaling'][key]
                train_means.append(data['training']['final_mean'])
                test_means.append(data['testing']['mean_reward'])
                convergence_eps.append(data['training']['convergence_episode'] or 0)
                fairness_stds.append(data['fairness']['fairness_std'])
                per_agent_rewards.append(data['testing']['mean_reward'] / n)
        
        # 1. Training performance
        ax1 = axes[0, 0]
        ax1.plot(agent_counts, train_means, marker='o', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Agents')
        ax1.set_ylabel('Mean Training Reward')
        ax1.set_title('Training Performance')
        ax1.grid(True, alpha=0.3)
        
        # 2. Test performance
        ax2 = axes[0, 1]
        ax2.plot(agent_counts, test_means, marker='s', linewidth=2, 
                markersize=8, color='green')
        ax2.set_xlabel('Number of Agents')
        ax2.set_ylabel('Mean Test Reward')
        ax2.set_title('Test Performance')
        ax2.grid(True, alpha=0.3)
        
        # 3. Convergence speed
        ax3 = axes[0, 2]
        ax3.plot(agent_counts, convergence_eps, marker='^', linewidth=2, 
                markersize=8, color='orange')
        ax3.set_xlabel('Number of Agents')
        ax3.set_ylabel('Episodes to Convergence')
        ax3.set_title('Convergence Speed')
        ax3.grid(True, alpha=0.3)
        
        # 4. Per-agent reward
        ax4 = axes[1, 0]
        ax4.plot(agent_counts, per_agent_rewards, marker='D', linewidth=2, 
                markersize=8, color='purple')
        ax4.set_xlabel('Number of Agents')
        ax4.set_ylabel('Reward per Agent')
        ax4.set_title('Scalability Analysis')
        ax4.grid(True, alpha=0.3)
        
        # 5. Fairness
        ax5 = axes[1, 1]
        ax5.plot(agent_counts, fairness_stds, marker='*', linewidth=2, 
                markersize=10, color='red')
        ax5.set_xlabel('Number of Agents')
        ax5.set_ylabel('Fairness Std Dev')
        ax5.set_title('Fairness at Scale')
        ax5.grid(True, alpha=0.3)
        
        # 6. Comparison table
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        table_data = []
        for i, n in enumerate(agent_counts):
            table_data.append([
                f"{n}",
                f"{train_means[i]:.1f}",
                f"{test_means[i]:.1f}",
                f"{convergence_eps[i]}" if convergence_eps[i] > 0 else "N/A",
                f"{fairness_stds[i]:.3f}"
            ])
        
        table = ax6.table(cellText=table_data,
                         colLabels=['Agents', 'Train', 'Test', 'Conv.', 'Fair.'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        plt.suptitle('Scaling Analysis: 5 → 20 → 100 Agents', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.results_dir / 'figures' / 'scaling_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.log(f"✓ Scaling analysis saved: {save_path.name}")
        
        # Print scaling insights
        self.log("\nScaling Insights:")
        self.log(f"  5 → 20 agents: {(test_means[1] / test_means[0] - 1) * 100:+.1f}% performance change")
        self.log(f"  20 → 100 agents: {(test_means[2] / test_means[1] - 1) * 100:+.1f}% performance change")
        self.log(f"  Per-agent efficiency: {per_agent_rewards[-1] / per_agent_rewards[0]:.2f}x")
    
    def save_all_results(self):
        """Save comprehensive results"""
        results_file = self.results_dir / 'data' / f'progressive_results_{self.timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.log(f"\n✓ All results saved to {results_file}")


# ========================================================================
# MAIN EXECUTION
# ========================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Progressive scaling experiments')
    parser.add_argument('--progressive', action='store_true', 
                       help='Run progressive 5→20→100 training')
    parser.add_argument('--quick_test', action='store_true', 
                       help='Quick test (reduced episodes)')
    parser.add_argument('--agents', type=int, nargs='+', default=[5, 20, 100],
                       help='Agent counts (default: 5 20 100)')
    parser.add_argument('--baseline_only', action='store_true',
                       help='Run only baseline experiments')
    
    args = parser.parse_args()
    
    runner = ProgressiveExperimentRunner()
    
    if args.baseline_only:
        # Just run baselines
        runner.run_baselines(n_agents=5, n_episodes=100)
    
    elif args.progressive or args.quick_test:
        # Run full progressive experiment
        
        # 1. Baselines (5 agents)
        runner.run_baselines(n_agents=5, n_episodes=100)
        
        # 2. Progressive training
        runner.run_progressive_training(
            agent_counts=args.agents,
            quick_test=args.quick_test
        )
        
        # 3. Save everything
        runner.save_all_results()
        
        runner.log("\n" + "="*70)
        runner.log("✓ ALL EXPERIMENTS COMPLETED!")
        runner.log(f"Results saved to: {runner.results_dir}")
        runner.log("="*70)
    
    else:
        # Show help
        parser.print_help()
        print("\n💡 Recommended usage:")
        print("  python comprehensive_experiments.py --progressive")
        print("  python comprehensive_experiments.py --quick_test")


if __name__ == '__main__':
    main()