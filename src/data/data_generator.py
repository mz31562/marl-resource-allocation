import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json

class SmartGridDataGenerator:
    """
    Generate realistic synthetic data for smart grid simulation.
    
    Produces:
    - Solar generation profiles (with weather effects)
    - Household consumption (with occupancy patterns)
    - Electricity pricing (TOU structure)
    - Grid capacity constraints
    """
    
    def __init__(self, n_agents=20, n_days=365, seed=42):
        self.n_agents = n_agents
        self.n_days = n_days
        self.n_hours = n_days * 24
        np.random.seed(seed)
        
    def generate_full_dataset(self):
        """Generate complete dataset for training/evaluation."""
        print(f"Generating {self.n_days} days of data for {self.n_agents} agents...")
        
        data = {
            'solar': self.generate_solar_profiles(),
            'consumption': self.generate_consumption_profiles(),
            'prices': self.generate_price_profiles(),
            'grid': self.generate_grid_constraints(),
            'metadata': self.generate_metadata()
        }
        
        return data
    
    def generate_solar_profiles(self):
        """
        Generate realistic solar generation profiles.
        
        Features:
        - Seasonal variation (higher in summer)
        - Daily sinusoidal pattern
        - Weather effects (clouds, rain)
        - Panel degradation over time
        """
        print("  → Generating solar profiles...")
        
        solar_data = np.zeros((self.n_agents, self.n_hours))
        
        for agent_id in range(self.n_agents):
            # Panel characteristics (heterogeneous)
            panel_capacity = np.random.uniform(3, 8)  # kW
            panel_efficiency = np.random.uniform(0.85, 0.98)
            panel_orientation = np.random.uniform(0.9, 1.1)  # South-facing bonus
            
            for day in range(self.n_days):
                # Seasonal variation (cosine over year)
                season_factor = 0.7 + 0.3 * np.cos(2 * np.pi * (day - 172) / 365)
                
                # Weather pattern for the day
                weather_type = np.random.choice(['clear', 'partly_cloudy', 'cloudy', 'rain'],
                                               p=[0.5, 0.3, 0.15, 0.05])
                weather_factors = {
                    'clear': 1.0,
                    'partly_cloudy': 0.7,
                    'cloudy': 0.3,
                    'rain': 0.1
                }
                weather_factor = weather_factors[weather_type]
                
                for hour in range(24):
                    hour_idx = day * 24 + hour
                    
                    # Base sinusoidal pattern (sunrise 6am, sunset 6pm)
                    sun_angle = np.sin((hour - 6) * np.pi / 12)
                    base_generation = max(0, sun_angle)
                    
                    # Add hour-to-hour variability (clouds passing)
                    noise = np.random.uniform(0.9, 1.1) if weather_type == 'partly_cloudy' else 1.0
                    
                    # Combine all factors
                    generation = (base_generation * 
                                season_factor * 
                                weather_factor * 
                                noise *
                                panel_capacity * 
                                panel_efficiency * 
                                panel_orientation)
                    
                    solar_data[agent_id, hour_idx] = max(0, generation)
        
        return solar_data
    
    def generate_consumption_profiles(self):
        """
        Generate realistic household consumption profiles.
        
        Features:
        - Baseload (always-on appliances)
        - Occupancy patterns (work from home vs office workers)
        - Appliance usage (cooking, HVAC, etc.)
        - Weekend vs weekday differences
        """
        print("  → Generating consumption profiles...")
        
        consumption_data = np.zeros((self.n_agents, self.n_hours))
        
        for agent_id in range(self.n_agents):
            # Household characteristics
            household_type = np.random.choice(['single', 'couple', 'family', 'work_from_home'],
                                            p=[0.2, 0.3, 0.3, 0.2])
            
            baseload = {
                'single': 0.2,
                'couple': 0.3,
                'family': 0.5,
                'work_from_home': 0.4
            }[household_type]
            
            for day in range(self.n_days):
                is_weekend = (day % 7) in [5, 6]
                
                for hour in range(24):
                    hour_idx = day * 24 + hour
                    
                    # Start with baseload
                    consumption = baseload
                    
                    # Add occupancy-based consumption
                    if household_type == 'work_from_home':
                        # Home all day
                        if 8 <= hour <= 17:
                            consumption += np.random.uniform(0.3, 0.6)
                    elif is_weekend:
                        # Home most of day on weekends
                        if 8 <= hour <= 22:
                            consumption += np.random.uniform(0.2, 0.5)
                    else:
                        # Weekday patterns
                        if 6 <= hour <= 8:  # Morning
                            consumption += np.random.uniform(0.5, 1.2)
                        elif 18 <= hour <= 22:  # Evening
                            consumption += np.random.uniform(0.8, 1.5)
                    
                    # Cooking times
                    if hour in [7, 8, 18, 19, 20]:
                        if np.random.random() < 0.6:
                            consumption += np.random.uniform(1.0, 2.5)
                    
                    # HVAC (seasonal)
                    season_factor = 0.7 + 0.3 * np.cos(2 * np.pi * (day - 172) / 365)
                    if season_factor > 0.85 or season_factor < 0.75:  # Hot summer or cold winter
                        consumption += np.random.uniform(0.5, 1.5)
                    
                    # Electric vehicle charging (20% of households)
                    if agent_id < self.n_agents * 0.2:
                        if 22 <= hour or hour <= 6:  # Night charging
                            if np.random.random() < 0.3:  # Not every night
                                consumption += np.random.uniform(3.0, 7.0)
                    
                    # Add noise
                    consumption *= np.random.uniform(0.95, 1.05)
                    
                    consumption_data[agent_id, hour_idx] = max(0.1, consumption)
        
        return consumption_data
    
    def generate_price_profiles(self):
        """
        Generate electricity pricing profiles.
        
        Features:
        - Time-of-Use (TOU) structure
        - Seasonal variation (higher in summer)
        - Peak/off-peak periods
        - Real-time price volatility
        """
        print("  → Generating price profiles...")
        
        prices = np.zeros(self.n_hours)
        
        # Base TOU structure ($/kWh)
        off_peak = 0.10
        mid_peak = 0.20
        on_peak = 0.35
        
        for day in range(self.n_days):
            # Seasonal multiplier (higher prices in summer)
            season_multiplier = 0.8 + 0.4 * np.cos(2 * np.pi * (day - 172) / 365)
            
            # Weekend discount
            is_weekend = (day % 7) in [5, 6]
            weekend_discount = 0.85 if is_weekend else 1.0
            
            for hour in range(24):
                hour_idx = day * 24 + hour
                
                # TOU periods
                if 0 <= hour <= 6:  # Night off-peak
                    base_price = off_peak
                elif 7 <= hour <= 9:  # Morning peak
                    base_price = on_peak
                elif 10 <= hour <= 16:  # Midday
                    base_price = mid_peak
                elif 17 <= hour <= 21:  # Evening peak
                    base_price = on_peak
                else:  # Late evening
                    base_price = mid_peak
                
                # Add price volatility (real-time pricing component)
                volatility = np.random.uniform(0.95, 1.15)
                
                # Combine factors
                price = base_price * season_multiplier * weekend_discount * volatility
                prices[hour_idx] = np.clip(price, 0.05, 0.60)
        
        return prices
    
    def generate_grid_constraints(self):
        """
        Generate grid capacity and constraint data.
        """
        print("  → Generating grid constraints...")
        
        # Transformer capacity (kW)
        transformer_capacity = self.n_agents * 5  # 5 kW per household average
        
        # Grid capacity varies by time (maintenance, other loads)
        grid_capacity = np.ones(self.n_hours) * transformer_capacity
        
        # Occasional maintenance periods (reduced capacity)
        n_maintenance_periods = self.n_days // 30
        for _ in range(n_maintenance_periods):
            start_hour = np.random.randint(0, self.n_hours - 24)
            duration = np.random.randint(4, 12)
            grid_capacity[start_hour:start_hour+duration] *= 0.7
        
        return grid_capacity
    
    def generate_metadata(self):
        """Generate metadata about agents and system."""
        print("  → Generating metadata...")
        
        metadata = {
            'n_agents': self.n_agents,
            'n_days': self.n_days,
            'agent_info': []
        }
        
        for agent_id in range(self.n_agents):
            agent_meta = {
                'agent_id': agent_id,
                'battery_capacity': np.random.uniform(8, 15),  # kWh
                'battery_efficiency': np.random.uniform(0.90, 0.95),
                'solar_capacity': np.random.uniform(3, 8),  # kW
                'has_ev': agent_id < self.n_agents * 0.2,
                'household_type': np.random.choice(['single', 'couple', 'family', 'work_from_home'])
            }
            metadata['agent_info'].append(agent_meta)
        
        return metadata
    
    def save_dataset(self, data, filepath='smart_grid_dataset.npz'):
        """Save dataset to disk."""
        print(f"\nSaving dataset to {filepath}...")
        
        np.savez_compressed(
            filepath,
            solar=data['solar'],
            consumption=data['consumption'],
            prices=data['prices'],
            grid_capacity=data['grid'],
            metadata=json.dumps(data['metadata'])
        )
        
        # Also save metadata as JSON
        with open(filepath.replace('.npz', '_metadata.json'), 'w') as f:
            json.dump(data['metadata'], f, indent=2)
        
        print(f"✓ Dataset saved ({filepath})")
        
        # Print statistics
        print("\n=== Dataset Statistics ===")
        print(f"Solar generation: mean={data['solar'].mean():.2f} kW, max={data['solar'].max():.2f} kW")
        print(f"Consumption: mean={data['consumption'].mean():.2f} kW, max={data['consumption'].max():.2f} kW")
        print(f"Prices: mean=${data['prices'].mean():.3f}/kWh, max=${data['prices'].max():.3f}/kWh")
        print(f"Grid capacity: mean={data['grid'].mean():.1f} kW")
    
    def visualize_dataset(self, data, days_to_plot=7):
        """Visualize sample of generated data."""
        print(f"\nGenerating visualizations for first {days_to_plot} days...")
        
        hours = np.arange(days_to_plot * 24)
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        # 1. Solar generation (sample agents)
        ax1 = axes[0]
        for i in range(min(5, self.n_agents)):
            ax1.plot(hours, data['solar'][i, :days_to_plot*24], 
                    alpha=0.7, label=f'Agent {i}')
        ax1.set_ylabel('Solar Generation (kW)')
        ax1.set_title('Solar Generation Profiles (First 5 Agents)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Consumption (sample agents)
        ax2 = axes[1]
        for i in range(min(5, self.n_agents)):
            ax2.plot(hours, data['consumption'][i, :days_to_plot*24], 
                    alpha=0.7, label=f'Agent {i}')
        ax2.set_ylabel('Consumption (kW)')
        ax2.set_title('Household Consumption Profiles (First 5 Agents)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Electricity prices
        ax3 = axes[2]
        ax3.plot(hours, data['prices'][:days_to_plot*24], color='red', linewidth=2)
        ax3.set_ylabel('Price ($/kWh)')
        ax3.set_title('Electricity Pricing (Time-of-Use)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Aggregate demand vs grid capacity
        ax4 = axes[3]
        total_consumption = data['consumption'][:, :days_to_plot*24].sum(axis=0)
        ax4.plot(hours, total_consumption, label='Total Demand', color='blue', linewidth=2)
        ax4.plot(hours, data['grid'][:days_to_plot*24], 
                label='Grid Capacity', color='red', linestyle='--', linewidth=2)
        ax4.fill_between(hours, 0, data['grid'][:days_to_plot*24], alpha=0.2, color='red')
        ax4.set_xlabel('Hour')
        ax4.set_ylabel('Power (kW)')
        ax4.set_title('Total Demand vs Grid Capacity')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('dataset_visualization.png', dpi=300, bbox_inches='tight')
        print("✓ Visualization saved (dataset_visualization.png)")
        plt.show()


# Example usage
if __name__ == '__main__':
    # Generate dataset
    generator = SmartGridDataGenerator(
        n_agents=20,  
        n_days=365,
        seed=42
    )

    data = generator.generate_full_dataset()
    generator.save_dataset(data, 'smart_grid_dataset_20agents_365days.npz')
    generator.visualize_dataset(data, days_to_plot=7)
    
    print("\n" + "="*60)
    print("Dataset generation complete!")
    print("="*60)
    print("\nTo use this dataset in training:")
    print("  data = np.load('smart_grid_dataset_20agents_365days.npz')")
    print("  solar = data['solar']")
    print("  consumption = data['consumption']")
    print("  prices = data['prices']")