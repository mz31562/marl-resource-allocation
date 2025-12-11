# docs/environment_spec.md

## State Space
- Battery level: [0, 1] (normalized)
- Solar generation: [0, 1] (normalized)
- Grid price: [0.1, 0.5] ($/kWh)
- Time: [0, 23] (hour of day)
- Neighbor actions: [n_neighbors x action_dim]

## Action Space
- charge_rate: [-1, 1] (negative = discharge)
- grid_interaction: [-1, 1] (negative = sell, positive = buy)

## Dynamics
battery(t+1) = battery(t) + solar(t) - consumption(t) + charge_rate(t)

## Rewards
individual_reward = -(price * grid_buy) + (price * 0.8 * grid_sell)
grid_penalty = -10 if sum(all_agents_grid_load) > capacity else 0
total_reward = individual_reward + grid_penalty