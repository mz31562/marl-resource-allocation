import gymnasium as gym
import numpy as np
from gymnasium import spaces
import json

class DatasetDrivenSmartGridEnv(gym.Env):
    """
    Smart Grid environment that uses real/generated datasets.
    
    Features:
    - Loads solar, consumption, and price data from files
    - Supports train/test split
    - Handles heterogeneous agents (different capacities)
    - More realistic dynamics than synthetic environment
    """
    
    def __init__(
        self,
        dataset_path: str,
        n_agents: int = 5,
        episode_length: int = 24,
        mode: str = 'train',  # 'train' or 'test'
        train_split: float = 0.8,
        terminal_battery_value: float = 0.0,
        seed: int = None
    ):
        super().__init__()
        
        self.episode_length = episode_length
        self.terminal_battery_value = terminal_battery_value
        self.mode = mode
        
        # Load dataset
        print(f"Loading dataset from {dataset_path}...")
        self._load_dataset(dataset_path)
        
        # Validate n_agents matches dataset
        assert n_agents == self.dataset_n_agents, \
            f"Requested {n_agents} agents but dataset has {self.dataset_n_agents}"
        
        self.n_agents = n_agents
        
        # Split data into train/test
        self._split_data(train_split)
        
        # Define spaces
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        obs_dim = 4 + (2 * (n_agents - 1))
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        if seed is not None:
            np.random.seed(seed)
        
        self.reset()
    
    def _load_dataset(self, path):
        """Load dataset from .npz file."""
        data = np.load(path, allow_pickle=True)
        
        self.solar_data = data['solar']  # [n_agents, n_hours]
        self.consumption_data = data['consumption']  # [n_agents, n_hours]
        self.price_data = data['prices']  # [n_hours]
        self.grid_capacity_data = data['grid_capacity']  # [n_hours]
        
        # Load metadata
        metadata = json.loads(str(data['metadata']))
        self.dataset_n_agents = metadata['n_agents']
        self.agent_metadata = metadata['agent_info']
        
        # Extract agent-specific parameters
        self.battery_capacities = np.array([
            agent['battery_capacity'] for agent in self.agent_metadata
        ])
        self.battery_efficiencies = np.array([
            agent['battery_efficiency'] for agent in self.agent_metadata
        ])
        self.solar_capacities = np.array([
            agent['solar_capacity'] for agent in self.agent_metadata
        ])
        
        self.n_hours_total = self.solar_data.shape[1]
        self.n_days_total = self.n_hours_total // 24
        
        print(f"✓ Loaded dataset: {self.dataset_n_agents} agents, {self.n_days_total} days")
    
    def _split_data(self, train_split):
        """Split data into train/test sets."""
        split_day = int(self.n_days_total * train_split)
        split_hour = split_day * 24
        
        if self.mode == 'train':
            self.solar = self.solar_data[:, :split_hour]
            self.consumption = self.consumption_data[:, :split_hour]
            self.prices = self.price_data[:split_hour]
            self.grid_capacity = self.grid_capacity_data[:split_hour]
            self.n_hours = split_hour
        else:  # test
            self.solar = self.solar_data[:, split_hour:]
            self.consumption = self.consumption_data[:, split_hour:]
            self.prices = self.price_data[split_hour:]
            self.grid_capacity = self.grid_capacity_data[split_hour:]
            self.n_hours = self.n_hours_total - split_hour
        
        self.n_days = self.n_hours // 24
        print(f"  → Using {self.mode} split: {self.n_days} days ({self.n_hours} hours)")
    
    def reset(self, seed=None, options=None):
        """Reset to random day in dataset."""
        super().reset(seed=seed)
        
        max_day = self.n_days - 1  # Leave room for 24-hour episode
        self.current_day = np.random.randint(0, max_day)
        
        self.current_step = 0
        
        # Initialize battery levels (random)
        self.battery_levels = np.random.uniform(0.3, 0.7, self.n_agents)
        
        # Store previous actions
        self.previous_actions = np.zeros((self.n_agents, 2))
        
        observations = self._get_observations()
        info = self._get_info()
        
        return observations, info
    
    def step(self, actions):
        """Execute one timestep using dataset values."""
        # Convert actions dict to array
        action_array = np.array([actions[i] for i in range(self.n_agents)])
        
        charge_rates = action_array[:, 0]
        grid_interactions = action_array[:, 1]
        
        # Get current hour in dataset
        hour_idx = self.current_day * 24 + self.current_step
        hour_idx = min(hour_idx, self.n_hours - 1)
        
        # Get actual values from dataset
        solar_generation = self.solar[:, hour_idx]  # Per agent
        consumption = self.consumption[:, hour_idx]  # Per agent
        grid_price = self.prices[hour_idx]
        grid_capacity = self.grid_capacity[hour_idx]
        
        rewards = np.zeros(self.n_agents)
        
        for i in range(self.n_agents):
            # Battery charge/discharge (scaled by agent's capacity)
            max_charge_rate = 0.2 * self.battery_capacities[i]  # 20% per hour
            charge_amount = charge_rates[i] * max_charge_rate / self.battery_capacities[i]
            
            # Grid interaction (scaled)
            grid_amount = grid_interactions[i] * 0.5
            
            # Energy balance (using actual solar and consumption from dataset)
            net_generation = solar_generation[i] - consumption[i]
            battery_change = net_generation + charge_amount * self.battery_capacities[i] + grid_amount
            
            # Update battery with efficiency
            if battery_change > 0:  # Charging
                battery_change *= self.battery_efficiencies[i]
            else:  # Discharging
                battery_change /= self.battery_efficiencies[i]
            
            self.battery_levels[i] = np.clip(
                self.battery_levels[i] + battery_change / self.battery_capacities[i],
                0.0,
                1.0
            )
            
            # Calculate rewards
            grid_cost = grid_price * max(0, grid_amount)
            grid_revenue = grid_price * 0.8 * abs(min(0, grid_amount))
            battery_penalty = -0.1 if self.battery_levels[i] < 0.2 else 0
            
            rewards[i] = -grid_cost + grid_revenue + battery_penalty
        
        # Collective penalty (grid overload)
        total_grid_load = np.sum(np.maximum(0, grid_interactions))
        grid_penalty = -10.0 if total_grid_load > grid_capacity / 10 else 0.0
        rewards += grid_penalty / self.n_agents
        
        # Store actions
        self.previous_actions = action_array
        
        # Increment step
        self.current_step += 1
        done = self.current_step >= self.episode_length
        truncated = False
        
        # Terminal rewards
        if done and self.terminal_battery_value > 0:
            for i in range(self.n_agents):
                terminal_bonus = self.battery_levels[i] * self.terminal_battery_value
                rewards[i] += terminal_bonus
        
        observations = self._get_observations()
        info = self._get_info()
        info['grid_load'] = total_grid_load
        info['grid_penalty'] = grid_penalty
        info['actual_solar'] = solar_generation
        info['actual_consumption'] = consumption
        
        # Convert to dicts
        obs_dict = {i: observations[i] for i in range(self.n_agents)}
        rewards_dict = {i: rewards[i] for i in range(self.n_agents)}
        dones_dict = {i: done for i in range(self.n_agents)}
        dones_dict['__all__'] = done
        
        return obs_dict, rewards_dict, dones_dict, truncated, info
    
    def _get_observations(self):
        """Get observations for all agents."""
        observations = []
        
        hour_idx = self.current_day * 24 + self.current_step
        hour_idx = min(hour_idx, self.n_hours - 1)
        
        # Get current conditions from dataset
        solar = self.solar[:, hour_idx].mean() / 10.0  # Normalize and average
        price = (self.prices[hour_idx] - 0.1) / 0.4  # Normalize to ~[-1, 1]
        time_normalized = (self.current_step % 24) / 24.0
        
        for i in range(self.n_agents):
            obs = [
                self.battery_levels[i],
                solar,
                price,
                time_normalized
            ]
            
            # Add neighbor actions
            for j in range(self.n_agents):
                if i != j:
                    obs.extend(self.previous_actions[j])
            
            observations.append(np.array(obs, dtype=np.float32))
        
        return observations
    
    def _get_info(self):
        """Get additional info."""
        return {
            'step': self.current_step,
            'day': self.current_day,
            'battery_levels': self.battery_levels.copy(),
            'avg_battery': np.mean(self.battery_levels),
            'battery_capacities': self.battery_capacities.copy()
        }


def train_on_dataset(
    dataset_path='smart_grid_dataset_20agents_365days.npz',
    n_episodes=1000,
    n_agents=5,
    save_name='maddpg_dataset_driven'
):
    """Train MADDPG using real dataset."""
    import torch
    from tqdm import tqdm
    import sys
    sys.path.append('..')
    from src.agents.maddpg_agent import MADDPGAgent
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on {device} with dataset: {dataset_path}")
    
    # Create dataset-driven environment
    env = DatasetDrivenSmartGridEnv(
        dataset_path=dataset_path,
        n_agents=n_agents,
        episode_length=24,
        mode='train',
        terminal_battery_value=10.0
    )
    
    # Initialize MADDPG
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    maddpg = MADDPGAgent(
        n_agents=n_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device
    )
    
    # Training metrics
    episode_rewards = []
    final_batteries = []
    
    print(f"\nTraining for {n_episodes} episodes...")
    
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
            maddpg.save(f'../results/checkpoints/{save_name}_best.pt')
        
        if (episode + 1) % 100 == 0:
            recent_reward = np.mean(episode_rewards[-100:])
            recent_battery = np.mean(final_batteries[-100:])
            print(f"\nEpisode {episode+1}: Avg Reward={recent_reward:.2f}, "
                  f"Avg Battery={recent_battery:.3f}, Best={best_reward:.2f}")
    
    print("\n✓ Training complete!")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_env = DatasetDrivenSmartGridEnv(
        dataset_path=dataset_path,
        n_agents=n_agents,
        mode='test',
        terminal_battery_value=10.0
    )
    
    test_rewards = []
    for episode in range(100):
        obs_dict, info = test_env.reset()
        episode_reward = 0
        
        for step in range(24):
            actions = maddpg.select_actions(obs_dict, explore=False)
            next_obs_dict, rewards_dict, dones_dict, truncated, info = test_env.step(actions)
            episode_reward += sum(rewards_dict.values())
            obs_dict = next_obs_dict
            
            if dones_dict['__all__']:
                break
        
        test_rewards.append(episode_reward)
    
    print(f"Test Performance: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    
    return maddpg, episode_rewards, test_rewards


if __name__ == '__main__':
    import sys
    
    # First, generate dataset if not exists
    try:
        data = np.load('smart_grid_dataset_20agents_365days.npz')
        print("Dataset found!")
    except FileNotFoundError:
        print("Dataset not found. Generating...")
        sys.path.append('..')
        from smart_grid_data_generator import SmartGridDataGenerator
        
        generator = SmartGridDataGenerator(n_agents=5, n_days=365, seed=42)
        data = generator.generate_full_dataset()
        generator.save_dataset(data, 'smart_grid_dataset_20agents_365days.npz')
    
    # Train on dataset
    maddpg, train_rewards, test_rewards = train_on_dataset(
        dataset_path='smart_grid_dataset_20agents_365days.npz',
        n_episodes=1000,
        n_agents=5
    )