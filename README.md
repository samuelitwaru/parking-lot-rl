# Parking Lot Reinforcement Learning Environment

A comprehensive parking lot simulation with reinforcement learning training capabilities and pygame visualization.

## Overview

This project implements a grid-based parking lot environment where:
- Cars randomly arrive and need to find parking spots
- An RL agent learns to select optimal parking spots to minimize walking distance from the entry
- Pygame provides real-time visualization of the environment
- Deep Q-Network (DQN) is used for training the agent

## Features

### Environment Features
- **Grid-based parking lot**: Customizable width and height
- **Dynamic car management**: Cars arrive randomly and leave after random time periods
- **Reward system**: 
  - Positive rewards for selecting empty spots
  - Negative rewards for selecting occupied spots
  - Distance-based bonus rewards (closer to entry = higher reward)
- **Entry point**: Designated entry position for calculating distance bonuses

### Visualization Features
- **Real-time pygame visualization**: Interactive grid display
- **Car tracking**: Shows car IDs and remaining parking time
- **Statistics panel**: Live statistics including success rate, occupancy rate
- **Interactive controls**: Click to select parking spots, reset environment
- **Auto mode**: Watch trained agent make decisions automatically

### Training Features
- **Deep Q-Network (DQN)**: Neural network-based RL agent
- **Experience replay**: Efficient learning from past experiences
- **Epsilon-greedy exploration**: Balanced exploration and exploitation
- **Target network**: Stable training with periodic target updates
- **Model persistence**: Save and load trained models

## Installation

### Requirements
```bash
pip install pygame numpy torch matplotlib
```

### Files Structure
```
parking-lot/
├── parking_environment.py    # Core environment implementation
├── pygame_visualizer.py      # Pygame visualization
├── rl_trainer.py            # DQN agent and training logic
├── main.py                  # Main entry point with CLI
└── README.md               # This file
```

## Usage

### 1. Interactive Simulation
Run the basic interactive simulation:
```bash
python main.py --mode sim
```

**Controls:**
- Click on grid cells to select parking spots
- Press `R` or click "Reset" to reset environment
- Press `Space` or click "Random Step" for random action
- Close window to exit

### 2. Train RL Agent
Train a new DQN agent:
```bash
python main.py --mode train
```

This will:
- Train for 1000 episodes
- Save the model as `parking_dqn_trained.pth`
- Display training progress plots
- Run test episodes and show results

### 3. Demo with Trained Agent
Run simulation with a trained agent:
```bash
python main.py --mode demo
```

**Additional Controls:**
- Press `A` or click "Auto" button to toggle automatic agent decisions
- All previous controls still work for manual override

### 4. Direct Module Usage

You can also use the components directly in your code:

```python
from parking_environment import ParkingLotEnvironment
from rl_trainer import ParkingLotTrainer
from pygame_visualizer import ParkingLotVisualizer

# Create environment
env = ParkingLotEnvironment(width=10, height=8, entry_position=(0, 4))

# For training
trainer = ParkingLotTrainer(env)
trainer.train(episodes=500)

# For visualization
visualizer = ParkingLotVisualizer(env)
visualizer.run()
```

## Environment Details

### State Representation
- **Grid state**: Flattened array of grid cell states (empty/occupied/entry)
- **Queue information**: Normalized number of cars waiting to park
- **State size**: `grid_width * grid_height + 1`

### Action Space
- **Actions**: Select any grid cell as parking spot
- **Action size**: `grid_width * grid_height`
- **Valid actions**: Only empty cells (not occupied or entry)

### Reward Function
```python
reward = base_reward + distance_bonus
```
- **Positive reward**: +10 for selecting empty spot
- **Negative reward**: -5 for selecting occupied spot
- **Distance bonus**: Up to +5 based on proximity to entry
- **Invalid action**: -5 for selecting entry or out-of-bounds

### Environment Parameters
- **Grid size**: Configurable width and height
- **Entry position**: Configurable entry point
- **Car arrival rate**: 0-3 cars per step (random)
- **Parking duration**: 5-20 time steps (random per car)

## Training Details

### DQN Architecture
- **Input layer**: State size (grid + queue info)
- **Hidden layers**: 3 layers with 128 neurons each
- **Output layer**: Action size (all grid positions)
- **Activation**: ReLU for hidden layers

### Training Parameters
- **Learning rate**: 0.001
- **Discount factor (γ)**: 0.99
- **Epsilon decay**: 0.995 (exploration → exploitation)
- **Batch size**: 32
- **Memory size**: 10,000 experiences
- **Target update frequency**: Every 100 episodes

### Training Process
1. **Experience Collection**: Agent interacts with environment
2. **Experience Replay**: Train on random batches from memory
3. **Target Network Updates**: Periodic updates for stability
4. **Action Masking**: Only consider valid actions
5. **Progress Tracking**: Monitor rewards and success rates

## Customization

### Environment Customization
```python
# Create custom environment
env = ParkingLotEnvironment(
    width=15,           # Wider parking lot
    height=10,          # Taller parking lot
    entry_position=(7, 0)  # Entry at top center
)

# Adjust rewards
env.positive_reward = 15.0    # Higher reward for success
env.negative_reward = -10.0   # Higher penalty for failure
env.max_distance_bonus = 8.0  # Higher distance bonus
```

### Training Customization
```python
# Create custom agent
trainer = ParkingLotTrainer(env)
agent = trainer.agent

# Adjust hyperparameters
agent.learning_rate = 0.0005   # Slower learning
agent.epsilon_decay = 0.999    # Slower exploration decay
agent.gamma = 0.95             # Less future-focused

# Custom training
trainer.train(
    episodes=2000,              # More episodes
    max_steps_per_episode=200   # Longer episodes
)
```

## Performance Metrics

The system tracks several performance metrics:

- **Success Rate**: Percentage of successful parking attempts
- **Occupancy Rate**: Current parking lot utilization
- **Average Reward**: Mean reward per episode during training
- **Queue Length**: Number of cars waiting to park
- **Training Loss**: DQN training loss over time

## Troubleshooting

### Common Issues

1. **Pygame not displaying**: Install pygame with `pip install pygame`
2. **CUDA errors**: The code automatically falls back to CPU if CUDA unavailable
3. **Memory issues**: Reduce batch size or memory size in DQN parameters
4. **Slow training**: Reduce episode count or use GPU acceleration

### Performance Tips

1. **Faster training**: Use smaller grid sizes for initial experiments
2. **Better convergence**: Increase training episodes and adjust learning rate
3. **Smoother visualization**: Reduce FPS in pygame loop for slower observation

## Future Enhancements

Potential improvements and extensions:

1. **Multi-agent environment**: Multiple cars making decisions simultaneously
2. **Dynamic pricing**: Variable parking costs based on demand
3. **Reserved spots**: VIP or handicap parking areas
4. **Traffic flow**: More realistic car movement patterns
5. **Advanced RL algorithms**: PPO, A3C, or other state-of-the-art methods

## License

This project is provided as-is for educational purposes. Feel free to modify and extend for your own research or learning needs.