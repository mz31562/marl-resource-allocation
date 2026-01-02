import json
import matplotlib.pyplot as plt
import numpy as np
import time

while True:
    try:
        # Find latest metrics file
        import glob
        files = glob.glob('../results/metrics_*.json')
        if files:
            latest = max(files, key=lambda x: x.split('_')[-1])
            
            with open(latest, 'r') as f:
                data = json.load(f)
            
            rewards = data['episode_rewards']
            
            plt.clf()
            plt.plot(rewards, alpha=0.3)
            
            # Smooth
            if len(rewards) > 50:
                smoothed = np.convolve(rewards, np.ones(50)/50, mode='valid')
                plt.plot(range(49, len(rewards)), smoothed, linewidth=2)
            
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title(f'Training Progress ({len(rewards)} episodes)')
            plt.grid(True)
            plt.pause(5)  # Update every 5 seconds
            
        time.sleep(5)
    except KeyboardInterrupt:
        break
    except:
        time.sleep(5)