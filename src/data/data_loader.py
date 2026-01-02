import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import json

class RealDatasetLoader:
    """
    Load and preprocess real-world datasets for smart grid RL.
    
    Supports:
    - Pecan Street Dataport
    - UK Power Networks Low Carbon London
    - NREL Solar Data
    - Custom CSV files
    """
    
    def __init__(self, output_path='processed_real_dataset.npz'):
        self.output_path = output_path
    
    def load_from_csv(self, 
                     consumption_csv,
                     solar_csv=None,
                     price_csv=None,
                     n_agents=None):
        """
        Load from custom CSV files.
        
        Expected formats:
        - consumption_csv: columns=['timestamp', 'agent_0', 'agent_1', ...]
        - solar_csv: columns=['timestamp', 'agent_0', 'agent_1', ...]
        - price_csv: columns=['timestamp', 'price']
        """
        print("Loading consumption data...")
        consumption_df = pd.read_csv(consumption_csv, parse_dates=['timestamp'])
        consumption_df = consumption_df.set_index('timestamp')
        
        # Determine number of agents
        if n_agents is None:
            n_agents = len([col for col in consumption_df.columns if col.startswith('agent_')])
        
        print(f"Found {n_agents} agents in consumption data")
        
        # Load solar data
        if solar_csv:
            print("Loading solar data...")
            solar_df = pd.read_csv(solar_csv, parse_dates=['timestamp'])
            solar_df = solar_df.set_index('timestamp')
        else:
            print("No solar data provided, generating synthetic...")
            solar_df = self._generate_solar_for_timestamps(consumption_df.index, n_agents)
        
        # Load price data
        if price_csv:
            print("Loading price data...")
            price_df = pd.read_csv(price_csv, parse_dates=['timestamp'])
            price_df = price_df.set_index('timestamp')
        else:
            print("No price data provided, generating synthetic...")
            price_df = self._generate_prices_for_timestamps(consumption_df.index)
        
        # Align timestamps
        print("Aligning timestamps...")
        common_index = consumption_df.index.intersection(solar_df.index).intersection(price_df.index)
        
        consumption_df = consumption_df.loc[common_index]
        solar_df = solar_df.loc[common_index]
        price_df = price_df.loc[common_index]
        
        # Convert to numpy arrays
        consumption_data = consumption_df[[f'agent_{i}' for i in range(n_agents)]].values.T
        solar_data = solar_df[[f'agent_{i}' for i in range(n_agents)]].values.T
        price_data = price_df['price'].values
        
        # Generate grid capacity
        grid_capacity = np.ones(len(common_index)) * n_agents * 5
        
        # Create metadata
        metadata = {
            'n_agents': n_agents,
            'n_hours': len(common_index),
            'n_days': len(common_index) // 24,
            'start_date': str(common_index[0]),
            'end_date': str(common_index[-1]),
            'source': 'custom_csv',
            'agent_info': [self._generate_agent_metadata(i) for i in range(n_agents)]
        }
        
        return {
            'solar': solar_data,
            'consumption': consumption_data,
            'prices': price_data,
            'grid_capacity': grid_capacity,
            'metadata': metadata
        }
    
    def load_pecan_street(self, 
                         csv_path,
                         n_agents=5,
                         start_date=None,
                         end_date=None):
        """
        Load Pecan Street Dataport data.
        
        Format: columns=['localminute', 'dataid', 'grid', 'solar', ...]
        """
        print("Loading Pecan Street data...")
        df = pd.read_csv(csv_path, parse_dates=['localminute'])
        
        # Filter by date range
        if start_date:
            df = df[df['localminute'] >= start_date]
        if end_date:
            df = df[df['localminute'] <= end_date]
        
        # Get unique households
        unique_homes = df['dataid'].unique()[:n_agents]
        print(f"Found {len(unique_homes)} households, using {n_agents}")
        
        # Resample to hourly
        consumption_data = []
        solar_data = []
        
        for home_id in unique_homes[:n_agents]:
            home_df = df[df['dataid'] == home_id].set_index('localminute')
            
            # Resample to hourly
            hourly = home_df.resample('1H').mean()
            
            # Extract consumption (grid + appliances)
            consumption = hourly['grid'].fillna(0).values
            consumption_data.append(consumption)
            
            # Extract solar (if available)
            if 'solar' in hourly.columns:
                solar = hourly['solar'].fillna(0).values
            else:
                solar = np.zeros_like(consumption)
            solar_data.append(solar)
        
        # Align all to same length
        min_length = min(len(c) for c in consumption_data)
        consumption_data = np.array([c[:min_length] for c in consumption_data])
        solar_data = np.array([s[:min_length] for s in solar_data])
        
        # Generate prices
        n_hours = min_length
        price_data = self._generate_realistic_prices(n_hours)
        
        # Grid capacity
        grid_capacity = np.ones(n_hours) * n_agents * 5
        
        metadata = {
            'n_agents': n_agents,
            'n_hours': n_hours,
            'n_days': n_hours // 24,
            'source': 'pecan_street',
            'agent_info': [self._generate_agent_metadata(i) for i in range(n_agents)]
        }
        
        return {
            'solar': solar_data,
            'consumption': consumption_data,
            'prices': price_data,
            'grid_capacity': grid_capacity,
            'metadata': metadata
        }
    
    def load_uk_low_carbon_london(self,
                                  csv_path,
                                  n_agents=5):
        """
        Load UK Power Networks Low Carbon London data.
        
        Format: columns=['LCLid', 'tstp', 'energy(kWh/hh)']
        where hh = half-hourly
        """
        print("Loading UK Low Carbon London data...")
        df = pd.read_csv(csv_path, parse_dates=['tstp'])
        
        # Get unique households
        unique_homes = df['LCLid'].unique()[:n_agents]
        print(f"Using {len(unique_homes)} households")
        
        consumption_data = []
        
        for home_id in unique_homes[:n_agents]:
            home_df = df[df['LCLid'] == home_id].set_index('tstp')
            
            # Convert half-hourly to hourly (sum two consecutive readings)
            hourly = home_df['energy(kWh/hh)'].resample('1H').sum()
            consumption = hourly.fillna(0).values
            consumption_data.append(consumption)
        
        # Align all to same length
        min_length = min(len(c) for c in consumption_data)
        consumption_data = np.array([c[:min_length] for c in consumption_data])
        
        # Generate solar (UK data doesn't include solar)
        solar_data = self._generate_solar_profile(n_agents, min_length)
        
        # Generate prices (UK TOU rates)
        price_data = self._generate_uk_prices(min_length)
        
        grid_capacity = np.ones(min_length) * n_agents * 5
        
        metadata = {
            'n_agents': n_agents,
            'n_hours': min_length,
            'n_days': min_length // 24,
            'source': 'uk_low_carbon_london',
            'agent_info': [self._generate_agent_metadata(i) for i in range(n_agents)]
        }
        
        return {
            'solar': solar_data,
            'consumption': consumption_data,
            'prices': price_data,
            'grid_capacity': grid_capacity,
            'metadata': metadata
        }
    
    def download_nrel_solar_data(self,
                                 latitude=40.7128,
                                 longitude=-74.0060,
                                 year=2020,
                                 api_key=None):
        """
        Download solar irradiance data from NREL NSRDB API.
        
        Get free API key from: https://developer.nrel.gov/signup/
        """
        if api_key is None:
            raise ValueError("NREL API key required. Get one at https://developer.nrel.gov/signup/")
        
        print(f"Downloading NREL solar data for ({latitude}, {longitude}) in {year}...")
        
        url = "https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-5min-download.csv"
        params = {
            'api_key': api_key,
            'wkt': f'POINT({longitude} {latitude})',
            'names': year,
            'attributes': 'ghi,dni,dhi,air_temperature',
            'leap_day': 'false',
            'interval': '60',  # hourly
            'utc': 'false',
            'email': 'your.email@example.com'
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            # Save to file
            filename = f'nrel_solar_{year}.csv'
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"✓ Saved to {filename}")
            return filename
        else:
            print(f"✗ Error: {response.status_code}")
            return None
    
    def _generate_solar_profile(self, n_agents, n_hours):
        """Generate synthetic solar profiles."""
        solar_data = np.zeros((n_agents, n_hours))
        
        for agent in range(n_agents):
            capacity = np.random.uniform(3, 8)
            
            for hour in range(n_hours):
                hour_of_day = hour % 24
                day = hour // 24
                
                # Seasonal variation
                season_factor = 0.7 + 0.3 * np.cos(2 * np.pi * (day - 172) / 365)
                
                # Daily pattern
                sun_angle = np.sin((hour_of_day - 6) * np.pi / 12)
                base = max(0, sun_angle)
                
                solar_data[agent, hour] = base * season_factor * capacity * np.random.uniform(0.8, 1.2)
        
        return solar_data
    
    def _generate_realistic_prices(self, n_hours):
        """Generate realistic TOU prices."""
        prices = np.zeros(n_hours)
        
        for hour in range(n_hours):
            hour_of_day = hour % 24
            day = hour // 24
            
            # Seasonal variation
            season_multiplier = 0.8 + 0.4 * np.cos(2 * np.pi * (day - 172) / 365)
            
            # TOU structure
            if 0 <= hour_of_day <= 6:
                base_price = 0.10
            elif 7 <= hour_of_day <= 9 or 17 <= hour_of_day <= 21:
                base_price = 0.35
            else:
                base_price = 0.20
            
            prices[hour] = base_price * season_multiplier * np.random.uniform(0.95, 1.05)
        
        return prices
    
    def _generate_uk_prices(self, n_hours):
        """Generate UK-style TOU prices (£/kWh)."""
        prices = np.zeros(n_hours)
        
        for hour in range(n_hours):
            hour_of_day = hour % 24
            
            # UK Economy 7 tariff structure
            if 0 <= hour_of_day <= 7:  # Night rate
                prices[hour] = 0.09
            else:  # Day rate
                prices[hour] = 0.22
        
        return prices
    
    def _generate_solar_for_timestamps(self, timestamps, n_agents):
        """Generate solar data matching timestamps."""
        df = pd.DataFrame(index=timestamps)
        
        for agent in range(n_agents):
            capacity = np.random.uniform(3, 8)
            solar = []
            
            for ts in timestamps:
                hour = ts.hour
                day_of_year = ts.timetuple().tm_yday
                
                season_factor = 0.7 + 0.3 * np.cos(2 * np.pi * (day_of_year - 172) / 365)
                sun_angle = np.sin((hour - 6) * np.pi / 12)
                base = max(0, sun_angle)
                
                solar.append(base * season_factor * capacity * np.random.uniform(0.8, 1.2))
            
            df[f'agent_{agent}'] = solar
        
        return df
    
    def _generate_prices_for_timestamps(self, timestamps):
        """Generate prices matching timestamps."""
        df = pd.DataFrame(index=timestamps)
        prices = []
        
        for ts in timestamps:
            hour = ts.hour
            day_of_year = ts.timetuple().tm_yday
            
            season_multiplier = 0.8 + 0.4 * np.cos(2 * np.pi * (day_of_year - 172) / 365)
            
            if 0 <= hour <= 6:
                base_price = 0.10
            elif 7 <= hour <= 9 or 17 <= hour <= 21:
                base_price = 0.35
            else:
                base_price = 0.20
            
            prices.append(base_price * season_multiplier * np.random.uniform(0.95, 1.05))
        
        df['price'] = prices
        return df
    
    def _generate_agent_metadata(self, agent_id):
        """Generate metadata for one agent."""
        return {
            'agent_id': agent_id,
            'battery_capacity': np.random.uniform(8, 15),
            'battery_efficiency': np.random.uniform(0.90, 0.95),
            'solar_capacity': np.random.uniform(3, 8),
            'has_ev': agent_id < 4,  # 20% have EVs
            'household_type': np.random.choice(['single', 'couple', 'family', 'work_from_home'])
        }
    
    def save_dataset(self, data):
        """Save processed dataset."""
        print(f"\nSaving processed dataset to {self.output_path}...")
        
        np.savez_compressed(
            self.output_path,
            solar=data['solar'],
            consumption=data['consumption'],
            prices=data['prices'],
            grid_capacity=data['grid_capacity'],
            metadata=json.dumps(data['metadata'])
        )
        
        print(f"✓ Dataset saved!")
        print(f"\n=== Dataset Info ===")
        print(f"Source: {data['metadata']['source']}")
        print(f"Agents: {data['metadata']['n_agents']}")
        print(f"Days: {data['metadata']['n_days']}")
        print(f"Hours: {data['metadata']['n_hours']}")


# Example usage
if __name__ == '__main__':
    loader = RealDatasetLoader(output_path='processed_real_dataset.npz')
    
    print("="*60)
    print("Real-World Dataset Loader")
    print("="*60)
    print("\nSupported datasets:")
    print("1. Custom CSV files")
    print("2. Pecan Street Dataport")
    print("3. UK Low Carbon London")
    print("4. NREL Solar Data (requires API key)")
    print("\nExample: Loading from custom CSVs")
    print("-"*60)
    
    # Example: Load from custom CSV
    # data = loader.load_from_csv(
    #     consumption_csv='consumption_data.csv',
    #     solar_csv='solar_data.csv',
    #     price_csv='price_data.csv',
    #     n_agents=20
    # )
    # loader.save_dataset(data)
    
    print("\nTo use real data:")
    print("1. Download dataset from sources listed in documentation")
    print("2. Call appropriate load function:")
    print("   - loader.load_from_csv(...)")
    print("   - loader.load_pecan_street(...)")
    print("   - loader.load_uk_low_carbon_london(...)")
    print("3. loader.save_dataset(data)")
    print("4. Use with DatasetDrivenSmartGridEnv")