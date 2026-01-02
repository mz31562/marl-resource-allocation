"""
Windows Setup and Run Script
Save as: setup_and_run.py in root directory
"""

import os
import sys

def check_directories():
    """Ensure all required directories exist."""
    print("Checking directory structure...")
    
    dirs = [
        'data',
        'data/raw',
        'data/processed',
        'results',
        'results/checkpoints',
        'results/figures'
    ]
    
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"  ✓ Created: {d}")
        else:
            print(f"  ✓ Exists: {d}")

def test_imports():
    """Test if all required modules can be imported."""
    print("\nTesting imports...")
    
    try:
        from src.data.data_generator import SmartGridDataGenerator
        print("  ✓ SmartGridDataGenerator")
    except Exception as e:
        print(f"  ✗ Error importing SmartGridDataGenerator: {e}")
        return False
    
    try:
        from src.environments.dataset_driven_env import DatasetDrivenSmartGridEnv
        print("  ✓ DatasetDrivenSmartGridEnv")
    except Exception as e:
        print(f"  ✗ Error importing DatasetDrivenSmartGridEnv: {e}")
        return False
    
    try:
        from src.agents.maddpg_agent import MADDPGAgent
        print("  ✓ MADDPGAgent")
    except Exception as e:
        print(f"  ✗ Error importing MADDPGAgent: {e}")
        return False
    
    print("  ✓ All imports successful!")
    return True

def generate_dataset():
    """Generate a dataset."""
    print("\n" + "="*60)
    print("GENERATING DATASET")
    print("="*60)
    
    from src.data.data_generator import SmartGridDataGenerator
    
    generator = SmartGridDataGenerator(
        n_agents=5,
        n_days=365,
        seed=42
    )
    
    print("\nGenerating data...")
    data = generator.generate_full_dataset()
    
    output_path = os.path.join('data', 'processed', 'dataset_20agents_365days.npz')
    generator.save_dataset(data, output_path)
    
    # Visualize
    try:
        generator.visualize_dataset(data, days_to_plot=7)
    except Exception as e:
        print(f"Warning: Could not create visualization: {e}")
    
    print(f"\n✓ Dataset saved to: {output_path}")
    return output_path

def train_model(dataset_path):
    """Train MADDPG on the dataset."""
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    
    import torch
    import numpy as np
    from tqdm import tqdm
    
    from src.environments.dataset_driven_env import DatasetDrivenSmartGridEnv
    from src.agents.maddpg_agent import MADDPGAgent
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create environment
    env = DatasetDrivenSmartGridEnv(
        dataset_path=dataset_path,
        n_agents=5,
        episode_length=24,
        mode='train',
        terminal_battery_value=10.0
    )
    
    # Initialize agent
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    maddpg = MADDPGAgent(
        n_agents=5,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device
    )
    
    # Training
    n_episodes = 100  # Start with 100 for quick test
    episode_rewards = []
    final_batteries = []
    
    print(f"\nTraining for {n_episodes} episodes...")
    print("(This is a quick test - use more episodes for full training)\n")
    
    best_reward = -float('inf')
    
    for episode in tqdm(range(n_episodes), desc="Training"):
        obs_dict, info = env.reset()
        episode_reward = 0
        
        for step in range(24):
            actions = maddpg.select_actions(obs_dict, explore=True)
            next_obs_dict, rewards_dict, dones_dict, truncated, info = env.step(actions)
            
            maddpg.store_transition(obs_dict, actions, rewards_dict, next_obs_dict, dones_dict)
            maddpg.update(batch_size=64)
            
            episode_reward += sum(rewards_dict.values())
            obs_dict = next_obs_dict
            
            if dones_dict['__all__']:
                break
        
        episode_rewards.append(episode_reward)
        final_batteries.append(info['avg_battery'])
        
        if episode_reward > best_reward:
            best_reward = episode_reward
            checkpoint_path = os.path.join('results', 'checkpoints', 'maddpg_quick_test_best.pt')
            maddpg.save(checkpoint_path)
        
        if (episode + 1) % 20 == 0:
            recent_reward = np.mean(episode_rewards[-20:])
            recent_battery = np.mean(final_batteries[-20:])
            print(f"\nEpisode {episode+1}: Reward={recent_reward:.2f}, "
                  f"Battery={recent_battery:.3f}, Best={best_reward:.2f}")
    
    # Save final model
    final_checkpoint = os.path.join('results', 'checkpoints', 'maddpg_quick_test_final.pt')
    maddpg.save(final_checkpoint)
    
    print(f"\n✓ Training complete!")
    print(f"  Best checkpoint: {checkpoint_path}")
    print(f"  Final checkpoint: {final_checkpoint}")
    
    return maddpg, episode_rewards

def evaluate_model(maddpg, dataset_path):
    """Evaluate the trained model."""
    print("\n" + "="*60)
    print("EVALUATING MODEL")
    print("="*60)
    
    import numpy as np
    from tqdm import tqdm
    from src.environments.dataset_driven_env import DatasetDrivenSmartGridEnv
    
    # Test environment
    env = DatasetDrivenSmartGridEnv(
        dataset_path=dataset_path,
        n_agents=5,
        mode='test',
        terminal_battery_value=10.0
    )
    
    test_rewards = []
    test_batteries = []
    
    print("\nRunning 50 test episodes...")
    
    for episode in tqdm(range(50), desc="Testing"):
        obs_dict, info = env.reset()
        episode_reward = 0
        
        for step in range(24):
            actions = maddpg.select_actions(obs_dict, explore=False)
            next_obs_dict, rewards_dict, dones_dict, truncated, info = env.step(actions)
            episode_reward += sum(rewards_dict.values())
            obs_dict = next_obs_dict
            
            if dones_dict['__all__']:
                break
        
        test_rewards.append(episode_reward)
        test_batteries.append(info['avg_battery'])
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Test Performance: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    print(f"Test Battery Level: {np.mean(test_batteries):.3f} ± {np.std(test_batteries):.3f}")
    print(f"{'='*60}\n")

def main():
    """Main execution flow."""
    print("="*60)
    print("SMART GRID MARL - WINDOWS SETUP")
    print("="*60)
    
    # Step 1: Check directories
    check_directories()
    
    # Step 2: Test imports
    if not test_imports():
        print("\n✗ Import test failed. Please check your installation.")
        return
    
    # Step 3: Check if dataset exists
    dataset_path = os.path.join('data', 'processed', 'dataset_20agents_365days.npz')
    
    if os.path.exists(dataset_path):
        print(f"\n✓ Dataset found: {dataset_path}")
        generate = input("Generate new dataset anyway? (y/n): ").lower() == 'y'
    else:
        print(f"\n✗ Dataset not found: {dataset_path}")
        generate = True
    
    if generate:
        dataset_path = generate_dataset()
    
    # Step 4: Train
    train = input("\nStart training? (y/n): ").lower() == 'y'
    
    if train:
        maddpg, episode_rewards = train_model(dataset_path)
        
        # Step 5: Evaluate
        evaluate = input("\nEvaluate model on test set? (y/n): ").lower() == 'y'
        if evaluate:
            evaluate_model(maddpg, dataset_path)
    
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. For full training, run:")
    print("   cd experiments")
    print("   python train_on_dataset.py --n_episodes 1000")
    print("\n2. For evaluation:")
    print("   cd experiments")
    print("   python evaluate_dataset_model.py")
    print("\n3. For experiments:")
    print("   See experiments folder for various training scripts")
    print("="*60)

if __name__ == '__main__':
    main()