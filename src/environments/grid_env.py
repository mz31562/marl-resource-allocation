import gymnasium as gym
import numpy as np
from gymnasium import spaces

class SmartGridEnv(gym.Env):
    """
    Multi-agent smart grid environment for energy management.
    
    Each agent represents a household with:
    - Solar panel
    - Battery storage
    - Connection to shared grid
    
    Args:
        n_agents: Number of households
        grid_capacity: Maximum grid load (kW)
        battery_capacity: Battery storage per agent (kWh)
        episode_length: Steps per episode (hours)
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(
        self,
        n_agents: int = 5,
        grid_capacity: float = 50.0,
        battery_capacity: float = 10.0,
        episode_length: int = 24,
        seed: int = None
    ):
        super().__init__()
        
        self.n_agents = n_agents
        self.grid_capacity = grid_capacity
        self.battery_capacity = battery_capacity
        self.episode_length = episode_length
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
        
        # Define action space (continuous)
        # [charge_rate, grid_interaction]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        # Define observation space
        # [battery_level, solar_gen, grid_price, time_of_day, neighbor_actions...]
        obs_dim = 4 + (2 * (n_agents - 1))  # Include neighbor actions
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Initialize state
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        self.current_step = 0
        
        # Initialize battery levels (random starting point)
        self.battery_levels = np.random.uniform(0.3, 0.7, self.n_agents)
        
        # Initialize solar generation profile (sinusoidal with noise)
        self.solar_profile = self._generate_solar_profile()
        
        # Initialize price profile (higher during peak hours)
        self.price_profile = self._generate_price_profile()
        
        # Store previous actions for observation
        self.previous_actions = np.zeros((self.n_agents, 2))
        
        # Get initial observations
        observations = self._get_observations()
        info = self._get_info()
        
        return observations, info
    
    def step(self, actions):
        """
        Execute one timestep.
        
        Args:
            actions: dict of {agent_id: action_array}
        
        Returns:
            observations, rewards, dones, truncated, info
        """
        # Convert actions dict to array
        action_array = np.array([actions[i] for i in range(self.n_agents)])
        
        # Extract charge rates and grid interactions
        charge_rates = action_array[:, 0]  # [-1, 1]
        grid_interactions = action_array[:, 1]  # [-1, 1]
        
        # Get current conditions
        current_hour = self.current_step
        solar_generation = self.solar_profile[current_hour]
        grid_price = self.price_profile[current_hour]
        
        # Simulate energy dynamics
        rewards = np.zeros(self.n_agents)
        
        for i in range(self.n_agents):
            # Solar generation (with some randomness)
            solar = solar_generation * np.random.uniform(0.8, 1.2)
            
            # Base consumption (fixed + random)
            consumption = 0.3 + np.random.uniform(0, 0.2)
            
            # Battery charge/discharge (scaled to reasonable rates)
            charge_amount = charge_rates[i] * 0.2  # Max 20% per hour
            
            # Grid interaction (scaled)
            grid_amount = grid_interactions[i] * 0.5  # Max 0.5 kW
            
            # Update battery level
            battery_change = solar - consumption + charge_amount + grid_amount
            self.battery_levels[i] = np.clip(
                self.battery_levels[i] + battery_change / self.battery_capacity,
                0.0,
                1.0
            )
            
            # Calculate individual reward
            # Reward for using solar, penalty for grid usage
            grid_cost = grid_price * max(0, grid_amount)  # Cost of buying
            grid_revenue = grid_price * 0.8 * abs(min(0, grid_amount))  # Revenue from selling
            
            battery_penalty = -0.1 if self.battery_levels[i] < 0.2 else 0  # Low battery penalty
            
            rewards[i] = -grid_cost + grid_revenue + battery_penalty
        
        # Calculate collective penalty (grid overload)
        total_grid_load = np.sum(np.maximum(0, grid_interactions))
        grid_penalty = -10.0 if total_grid_load > self.grid_capacity / 10 else 0.0
        
        # Add grid penalty to all agents (shared responsibility)
        rewards += grid_penalty / self.n_agents
        
        # Store actions for next observation
        self.previous_actions = action_array
        
        # Increment step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= self.episode_length
        truncated = False
        
        # Get new observations
        observations = self._get_observations()
        info = self._get_info()
        info['grid_load'] = total_grid_load
        info['grid_penalty'] = grid_penalty
        
        # Convert to dict format
        obs_dict = {i: observations[i] for i in range(self.n_agents)}
        rewards_dict = {i: rewards[i] for i in range(self.n_agents)}
        dones_dict = {i: done for i in range(self.n_agents)}
        dones_dict['__all__'] = done
        
        return obs_dict, rewards_dict, dones_dict, truncated, info
    
    def _get_observations(self):
        """Get observations for all agents."""
        observations = []
        
        current_hour = self.current_step % 24
        solar = self.solar_profile[current_hour]
        price = self.price_profile[current_hour]
        time_normalized = current_hour / 24.0
        
        for i in range(self.n_agents):
            # Own state
            obs = [
                self.battery_levels[i],
                solar,
                price,
                time_normalized
            ]
            
            # Add neighbor actions (for communication)
            for j in range(self.n_agents):
                if i != j:
                    obs.extend(self.previous_actions[j])
            
            observations.append(np.array(obs, dtype=np.float32))
        
        return observations
    
    def _get_info(self):
        """Get additional info."""
        return {
            'step': self.current_step,
            'battery_levels': self.battery_levels.copy(),
            'avg_battery': np.mean(self.battery_levels)
        }
    
    def _generate_solar_profile(self):
        """Generate realistic solar generation profile (24 hours)."""
        hours = np.arange(24)
        # Peak at noon, zero at night
        solar = np.maximum(0, np.sin((hours - 6) * np.pi / 12))
        # Add some day-to-day variation
        solar *= np.random.uniform(0.7, 1.3)
        return solar
    
    def _generate_price_profile(self):
        """Generate electricity price profile (24 hours)."""
        hours = np.arange(24)
        # Higher prices during peak hours (morning and evening)
        morning_peak = np.exp(-((hours - 8) ** 2) / 8)
        evening_peak = np.exp(-((hours - 19) ** 2) / 8)
        price = 0.2 + 0.2 * (morning_peak + evening_peak)
        return price
    
    def render(self, mode='human'):
        """Render the environment."""
        if mode == 'human':
            print(f"\n=== Step {self.current_step} ===")
            print(f"Battery levels: {self.battery_levels}")
            print(f"Average battery: {np.mean(self.battery_levels):.2f}")