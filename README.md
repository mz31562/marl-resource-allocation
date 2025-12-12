# MARL Resource Allocation

Multi-agent reinforcement learning for smart grid energy management. Households with solar panels and batteries learn to coordinate their energy usage while maximizing individual profits and maintaining grid stability.

## Overview

This project implements MADDPG and PPO algorithms to solve distributed energy management in smart grids. Multiple agents learn to balance self-consumption, storage, and grid interaction without centralized control.

**Problem**: Can autonomous household agents learn to coordinate energy resources efficiently while respecting shared grid constraints?

**Result**: MADDPG agents achieve 165% improvement over greedy heuristics and learn emergent coordination behaviors.

## Quick Start

```bash
# Clone and install
git clone https://github.com/mz31562/marl-resource-allocation.git
cd marl-resource-allocation
pip install -r requirements.txt

# Train agents
python experiments/train_multi_agent.py

# Run evaluation
python experiments/evaluate.py

# Test environment
python notebooks/environment_test.py
```

## Features

- **Multi-agent environment**: Gymnasium-based smart grid simulator
- **MADDPG implementation**: Centralized training, decentralized execution
- **Comprehensive baselines**: Random, greedy, PPO, and optimal policies
- **Ablation studies**: Agent scaling, reward structures, communication
- **Full visualization**: Training curves, policy analysis, performance metrics

## Environment

**State**: Battery level, solar generation, grid price, time, neighbor actions  
**Action**: Battery charge rate, grid buy/sell  
**Reward**: Electricity cost/revenue - grid overload penalty

See [docs/environment_spec.md](docs/environment_spec.md) for details.

## Algorithms

### MADDPG
Multi-Agent DDPG with centralized critics and decentralized actors. Uses experience replay and target networks for stable training.

### PPO
Single-agent Proximal Policy Optimization baseline for comparison.

## Results

| Method | Mean Reward | Grid Violations |
|--------|-------------|-----------------|
| Random | -15.3 ± 2.1 | 45% |
| Greedy | -5.2 ± 1.8 | 28% |
| **MADDPG** | **+8.5 ± 1.2** | **8%** |
| Optimal | +12.3 ± 0.9 | 2% |

MADDPG achieves 69% of optimal performance while using only local observations and decentralized execution.

## Key Findings

- Agents learn time-of-use optimization without explicit programming
- Coordination emerges through implicit communication (observing neighbor actions)
- Performance scales gracefully from 2 to 20 agents
- Learned policies maintain grid stability 92% of the time

See [docs/technical_report.md](docs/technical_report.md) for full analysis.

## Training

```bash
# Multi-agent MADDPG (recommended)
python experiments/train_multi_agent.py --n_agents 5 --n_episodes 2000

# Single-agent PPO baseline
python experiments/train_single_agent.py --n_episodes 1000

# Run ablation studies
python experiments/run_ablation.py

# Compare baselines
python experiments/train_baseline.py
```

## Configuration

Default settings in `src/config/default_config.yaml`:

```yaml
training:
  n_episodes: 2000
  batch_size: 64
  learning_rate: 0.001

environment:
  n_agents: 5
  battery_capacity: 10.0
  episode_length: 24
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Gymnasium 0.28+
- CUDA (optional, for GPU acceleration)

See `requirements.txt` for complete dependencies.

## References

**Foundational Work**:
- Mnih et al. (2013) - Playing Atari with Deep RL
- Schulman et al. (2017) - Proximal Policy Optimization
- Lowe et al. (2017) - Multi-Agent Actor-Critic (MADDPG)

**Advanced Methods**:
- Rashid et al. (2018) - QMIX
- Yu et al. (2022) - PPO in Cooperative Multi-Agent Games
- Bansal et al. (2018) - Emergent Complexity via Competition

See References for complete citations.

## License

MIT License - see LICENSE file for details.

## Contact

**Mohammed Zaid**  
email: mohazaid2001@gmail.com  
GitHub: (https://github.com/mz31562/marl-resource-allocation)

---
## References 

1. **Mnih, V., Kavukcuoglu, K., Silver, D., et al.** (2013). Playing Atari with Deep Reinforcement Learning. *arXiv preprint arXiv:1312.5602*.

2. **Schulman, J., Wolski, F., Dhariwal, P., et al.** (2017). Proximal Policy Optimization Algorithms. *arXiv preprint arXiv:1707.06347*.

3. **Lowe, R., Wu, Y., Tamar, A., et al.** (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments. *Advances in Neural Information Processing Systems*, 30.

4. **Rashid, T., Samvelyan, M., Schroeder, C., et al.** (2018). QMIX: Monotonic Value Function Factorisation for Decentralized Multi-Agent Reinforcement Learning. *ICML*, 80, 4295-4304.

5. **Yu, C., Velu, A., Vinitsky, E., et al.** (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games. *NeurIPS*, 35, 24611-24624.

6. **Bansal, T., Pachocki, J., Sidor, S., et al.** (2018). Emergent Complexity via Multi-Agent Competition. *ICLR*.
