import json
import numpy as np
import glob

# Find the most recent metrics file
metrics_files = glob.glob('../results/metrics_maddpg_100agents_v1_*.json')
if metrics_files:
    latest_file = max(metrics_files)  # Most recent by timestamp
    print(f"Loading: {latest_file}")
    
    with open(latest_file, 'r') as f:
        metrics = json.load(f)
    
    rewards = metrics['episode_rewards']
    
    # Calculate gaps
    train_early = np.mean(rewards[0:200])
    train_late = np.mean(rewards[800:1000])
    test_performance = 460.52  # From your output
    
    print(f"\n{'='*50}")
    print(f"PERFORMANCE ANALYSIS")
    print(f"{'='*50}")
    print(f"Early training (ep 0-200):   {train_early:.2f}")
    print(f"Late training (ep 800-1000):  {train_late:.2f}")
    print(f"Test performance:             {test_performance:.2f}")
    print(f"\nEarly/Late gap:  {train_late - train_early:.2f} (learning progress)")
    print(f"Train/Test gap:  {train_late - test_performance:.2f} (generalization gap)")
    print(f"{'='*50}")
    
    # Interpretation
    if train_late - train_early > 150:
        print("✓ Strong learning signal - consider training longer")
    else:
        print("⚠ Learning may have plateaued")
    
    if train_late - test_performance > 150:
        print("⚠ Possible overfitting - increase regularization")
    elif train_late - test_performance < 50:
        print("✓ Good generalization!")
    else:
        print("✓ Moderate generalization - acceptable for MARL")

else:
    print("No metrics file found! Run the script below to save metrics during training.")