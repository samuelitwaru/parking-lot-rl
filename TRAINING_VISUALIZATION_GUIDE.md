# 🎮 Training with Pygame Visualization

## New Feature: `--simulate` Flag

You can now watch your AI agent learn in real-time using pygame visualization during the training process!

## Usage Examples

### 1. Training with Visualization (Recommended)
```bash
# Train with real-time pygame window showing the learning process
python main.py --mode train --simulate
```

### 2. Training without Visualization (Faster)
```bash
# Traditional training without visualization (faster)
python main.py --mode train
```

### 3. Other Modes (Unchanged)
```bash
# Interactive simulation
python main.py --mode sim

# Demo with trained agent
python main.py --mode demo
```

## What You'll See During Training

### 🖥️ Real-time Visualization Window
- **Live Grid**: Watch the parking lot state update in real-time
- **Training Stats**: Episode number, current score, agent epsilon
- **Action Feedback**: See which action the AI just took and its reward
- **Environment Statistics**: Success rate, occupancy, total cars parked
- **Car Queue**: Live view of cars waiting to park

### 🎯 Visual Highlights
- **Green cells**: Entry points
- **Red cells**: Occupied parking spots with car ID and time remaining
- **Gray cells**: Empty parking spots with coordinates
- **Yellow highlight**: Last action taken by the AI agent

### ⌨️ Interactive Controls
- **ESC key**: Stop training early
- **Close window**: Stop training early
- **Space**: Pause (can be extended for debugging)

## Technical Details

### Performance Impact
- **With `--simulate`**: ~20 FPS visualization, slightly slower training
- **Without `--simulate`**: Full speed training, no visual overhead

### Memory Usage
- Pygame window adds ~10-20MB memory usage
- Training data structures remain the same

### Training Behavior
- **Agent learns normally**: Visualization doesn't affect learning
- **Same episodes**: Default 1000 episodes regardless of visualization
- **Early stopping**: You can stop training by closing the window

## Benefits of Visual Training

### 🔍 **Understanding Learning Process**
- Watch how the agent's behavior improves over episodes
- See which parking strategies it develops
- Observe how it handles different queue lengths

### 🐛 **Debugging Training**
- Spot if the agent gets stuck in local optima
- Identify if the environment is working correctly
- Monitor if rewards are being calculated properly

### 📊 **Real-time Monitoring**
- Track success rate improvements live
- See occupancy patterns develop
- Monitor epsilon decay (exploration vs exploitation)

### 🎓 **Educational Value**
- Perfect for presentations and demonstrations
- Help others understand reinforcement learning
- Visual learning for complex RL concepts

## Advanced Usage

### Custom Training Parameters
```bash
# Train for more episodes with visualization
python main.py --mode train --simulate

# Then edit main.py to change episodes:
# scores = trainer.train(episodes=2000, max_steps_per_episode=50, visualize=simulate)
```

### Faster Visualization
The visualization runs at 20 FPS by default. For faster training with occasional visual updates, you can modify the `update_delay` in the `TrainingVisualizer` class.

## Troubleshooting

### Window Not Appearing
- Make sure pygame is installed: `pip install pygame>=2.1.0`
- Check if running in a headless environment (no display)

### Training Too Slow
- Use `--mode train` without `--simulate` for maximum speed
- The visualization adds slight overhead but shouldn't dramatically slow training

### Memory Issues
- Close other applications if running low on memory
- The pygame window uses minimal additional resources

## Example Output

```bash
$ python main.py --mode train --simulate

Parking Lot Reinforcement Learning Environment
==================================================
Running RL training...
With pygame visualization enabled
Starting RL Training...
Starting training for 1000 episodes...
State size: 80, Action size: 80
Pygame visualization enabled during training

Episode 100/1000, Average Score: 12.45, Epsilon: 0.366
  Success Rate: 67.2%, Occupancy: 43.1%

Episode 200/1000, Average Score: 18.32, Epsilon: 0.134
  Success Rate: 78.5%, Occupancy: 52.3%

[Training continues with live pygame window...]
```

Enjoy watching your AI learn to park! 🚗🤖