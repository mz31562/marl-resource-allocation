# Smart Grid Environment Specification

## Overview

The SmartGridEnv is a multi-agent gymnasium environment that simulates a residential community with distributed energy resources. Each agent represents a household with solar generation, battery storage, and grid connectivity.

## State Space

Each agent observes:

| Component | Range | Description |
|-----------|-------|-------------|
| Battery Level | [0, 1] | Normalized state of charge (0 = empty, 1 = full) |
| Solar Generation | [0, 1] | Current normalized solar output |
| Grid Price | [0.1, 0.5] | Electricity price in $/kWh |
| Time of Day | [0, 23] | Current hour (0-23) |
| Neighbor Actions | [-1, 1]^(2n-2) | Previous actions of other agents |

**Total observation dimension**: 4 + 2×(n_agents - 1)

## Action Space

Each agent controls two continuous actions:

| Action | Range | Description |
|--------|-------|-------------|
| Charge Rate | [-1, 1] | Battery charge/discharge rate<br>-1 = max discharge, +1 = max charge |
| Grid Interaction | [-1, 1] | Grid buy/sell<br>-1 = max sell, +1 = max buy |

Actions are clipped to [-1, 1] before environment execution.

## Dynamics

The environment evolves according to:


battery(t+1) = clip(battery(t) + Δbattery, 0, 1)

where:
  Δbattery = (solar(t) - consumption(t) + charge_rate(t) × 0.2 + grid(t) × 0.5) / battery_capacity


**Key parameters**:
- Solar generation: Sinusoidal profile peaking at noon
- Consumption: 0.3 + uniform(0, 0.2) kW per hour
- Battery capacity: 10 kWh per agent
- Max charge/discharge rate: 2 kW (20% of capacity)
- Max grid interaction: 0.5 kW per agent

## Reward Structure

### Individual Reward

Each agent receives:


individual_reward = -grid_cost + grid_revenue + battery_penalty

where:
  grid_cost = price × max(0, grid_interaction)
  grid_revenue = price × 0.8 × |min(0, grid_interaction)|
  battery_penalty = -0.1 if battery < 0.2 else 0


The 0.8 coefficient on grid revenue reflects realistic feed-in tariffs where selling price is lower than buying price.

### Collective Penalty

To encourage coordination, all agents share a grid capacity constraint:

grid_penalty = -10.0 if sum(max(0, grid_i)) > grid_capacity/10 else 0.0


This penalty is distributed equally among all agents, incentivizing them to avoid simultaneous high grid usage.

### Total Reward


total_reward_i = individual_reward_i + grid_penalty / n_agents

## Episode Structure

- **Episode length**: 24 timesteps (representing 24 hours)
- **Reset**: Battery levels initialized randomly in [0.3, 0.7]
- **Termination**: Episode ends after 24 steps
- **Truncation**: Not used (episodes always run to completion)

## Price Profile

Electricity prices vary throughout the day:

## python
hour = [0, 1, 2, ..., 23]
morning_peak = exp(-((hour - 8)^2) / 8)
evening_peak = exp(-((hour - 19)^2) / 8)
price = 0.2 + 0.2 × (morning_peak + evening_peak)


This creates realistic price dynamics with:
- Low prices: 0.2 $/kWh (night)
- Peak prices: 0.5 $/kWh (morning 8am, evening 7pm)

## Solar Profile

Solar generation follows a sinusoidal pattern:

## python
hour = [0, 1, 2, ..., 23]
solar = max(0, sin((hour - 6) × π / 12)) × random_factor

where random_factor ~ Uniform(0.7, 1.3)


This produces:
- Zero generation: 6pm - 6am
- Peak generation: noon
- Daily variability through random scaling

## Observation Construction

For agent i, the observation vector is:

## python
obs_i = [
    battery_levels[i],           # Own battery
    solar_generation,             # Current solar (shared)
    grid_price,                   # Current price (shared)
    current_hour / 24.0,          # Time of day (normalized)
    prev_actions[0][0],           # Neighbor 0 charge_rate
    prev_actions[0][1],           # Neighbor 0 grid
    ...
    prev_actions[j][0],           # Neighbor j charge_rate (j ≠ i)
    prev_actions[j][1]            # Neighbor j grid
]


This provides partial observability while enabling implicit communication through action observation.

## Implementation Details

**Environment class**: `SmartGridEnv(gym.Env)`

**Key methods**:
- `reset()`: Initialize episode
- `step(actions)`: Execute one timestep
- `render()`: Display current state (optional)

**Returns**:
- Observations: `dict{agent_id: np.ndarray}`
- Rewards: `dict{agent_id: float}`
- Dones: `dict{agent_id: bool}` + `{'__all__': bool}`
- Info: `dict` with auxiliary information

## Configuration

Default configuration in `src/config/default_config.yaml`:

yaml
environment:
  n_agents: 5
  grid_capacity: 50.0  # kW
  battery_capacity: 10.0  # kWh
  episode_length: 24  # hours


## Usage Example

python
from src.environments.grid_env import SmartGridEnv

env = SmartGridEnv(n_agents=5)
obs, info = env.reset()

for step in range(24):
    actions = {i: env.action_space.sample() for i in range(5)}
    obs, rewards, dones, truncated, info = env.step(actions)
    
    if dones['__all__']:
        break


## Design Rationale

1. **Continuous actions**: Real energy systems require fine-grained control
2. **Partial observability**: Agents don't have perfect information about others' states
3. **Implicit communication**: Observing neighbors' actions enables coordination
4. **Mixed rewards**: Balances individual rationality with collective welfare
5. **Realistic dynamics**: Solar profiles, price patterns, and battery constraints mirror real systems

## Validation

The environment has been validated for:
- Action space bounds respected
- Battery levels remain in [0, 1]
- Energy conservation (no free energy)
- Realistic reward magnitudes
- Stable dynamics (no divergence)

## Future Extensions

Potential enhancements:
1. Stochastic solar/demand forecasts
2. Time-varying grid capacity
3. Battery degradation over episodes
4. Electric vehicle charging
5. Heterogeneous agent types (commercial, industrial)