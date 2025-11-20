# Simulation & Training: Environment Setup and Optimization

## Environment Setup

### Simulation Architecture

The parking lot simulation environment is built using a modular architecture that separates concerns and enables flexible experimentation:

```
ParkingLotEnvironment (Core Logic)
├── Grid Management: 10×8 parking grid with entry points
├── Car Management: Dynamic car generation, parking, and departure
├── State Representation: Flattened grid + queue + statistics
├── Reward Calculation: Multi-objective reward function
└── Action Validation: Constraint checking and available actions

PygameVisualizer (Visualization)
├── Real-time Rendering: Visual grid updates and car tracking
├── Interactive Controls: Manual parking spot selection
├── Statistics Display: Live performance metrics
└── Training Visualization: Real-time learning progress

TrainingVisualizer (Training-specific)
├── Episode Tracking: Progress monitoring during training
├── Action Highlighting: Visual feedback for AI decisions
├── Performance Metrics: Live success rate and occupancy
└── Early Stopping: User-controlled training termination
```

### Core Environment Parameters

**Grid Configuration:**
```python
width = 10              # Parking lot width (columns)
height = 8              # Parking lot height (rows)  
entry_position = (0, 4) # Entry point coordinates (x, y)
total_spots = 79        # Available parking spots (80 - 1 entry)
```

**Car Generation Parameters:**
```python
arrival_rate = random.randint(0, 3)      # New cars per time step
parking_duration = random.randint(25, 50) # How long cars stay parked
queue_capacity = unlimited               # No artificial queue limits
```

**Reward Structure:**
```python
positive_reward = 10.0      # Successful parking bonus
negative_reward = -5.0      # Invalid action penalty
max_distance_bonus = 5.0    # Maximum distance-based reward
distance_formula = euclidean # √((x₁-x₂)² + (y₁-y₂)²)
```

### State Space Engineering

**Grid Representation:**
- **Flattened Vector**: 80-dimensional vector representing grid state
- **Encoding**: 0=empty, 1=occupied, 2=entry
- **Spatial Information**: Preserves relative positions for spatial learning

**Temporal Features:**
- **Car Timers**: Remaining parking time for each occupied spot
- **Queue Information**: Number of waiting cars and their durations
- **System Statistics**: Historical performance metrics

**Feature Engineering:**
```python
def get_state():
    grid_features = flatten(occupancy_grid)      # 80 features
    queue_features = [len(queue), avg_duration]  # 2 features  
    stats_features = [occupancy_rate, success_rate] # 2 features
    return concatenate([grid_features, queue_features, stats_features])
```

### Environment Dynamics

**Time Step Process:**
1. **Action Execution**: Agent selects parking spot for queued car
2. **Validity Check**: Ensure selected spot is empty and available
3. **Car Placement**: If valid, place car with timer set to parking duration
4. **Time Progression**: Decrement all parked car timers by 1
5. **Car Departure**: Remove cars with timer ≤ 0
6. **New Arrivals**: Generate 0-3 new cars with random parking durations
7. **Queue Update**: Add new cars to waiting queue
8. **Reward Calculation**: Compute reward based on action outcome
9. **State Update**: Generate new state representation

**Stochastic Elements:**
- **Arrival Patterns**: Random number of cars per time step
- **Parking Durations**: Variable stay times create dynamic occupancy
- **Initial Conditions**: Random starting configurations for generalization

## Hyperparameter Tuning

### Critical Hyperparameters

**1. Learning Rate (α)**
```python
learning_rate = 0.001  # Current setting
```
**Range Tested**: [0.0001, 0.001, 0.01, 0.1]
**Selection Criteria**: Balance between learning speed and stability
**Optimal Value**: 0.001 (best convergence without overshooting)

**Tuning Strategy:**
- **Grid Search**: Tested logarithmic scale values
- **Performance Metric**: Average reward over 100 episodes
- **Stability Check**: Variance in performance across multiple runs

**2. Discount Factor (γ)**
```python
gamma = 0.99  # High discount for long-term planning
```
**Range Tested**: [0.9, 0.95, 0.99, 0.999]
**Selection Criteria**: Balance immediate vs future rewards
**Optimal Value**: 0.99 (encourages forward-thinking parking decisions)

**3. Exploration Parameters**
```python
epsilon_start = 1.0     # Begin with full exploration
epsilon_min = 0.01      # Maintain 1% exploration permanently  
epsilon_decay = 0.995   # Gradual transition to exploitation
```
**Tuning Methodology:**
- **Epsilon Start**: Always 1.0 for comprehensive initial exploration
- **Epsilon Min**: Tested [0.01, 0.05, 0.1] - 0.01 prevents complete exploitation
- **Decay Rate**: Tested [0.99, 0.995, 0.999] - 0.995 provides good balance

**4. Neural Network Architecture**
```python
hidden_size = 128       # Neurons per hidden layer
num_layers = 3         # Hidden layers (plus input/output)
activation = ReLU      # Activation function
```
**Architecture Search:**
- **Hidden Sizes**: [64, 128, 256, 512] - 128 optimal for problem complexity
- **Depth**: [2, 3, 4, 5] layers - 3 layers sufficient for pattern learning
- **Activation**: ReLU outperformed Tanh and Sigmoid

**5. Experience Replay Parameters**
```python
memory_size = 10000    # Experience buffer capacity
batch_size = 32       # Training batch size
```
**Memory Size Tuning:**
- **Range**: [1000, 5000, 10000, 20000]
- **Selection**: 10000 provides good diversity without memory issues
- **Batch Size**: [16, 32, 64] - 32 balances learning stability and speed

### Hyperparameter Tuning Strategy

**Phase 1: Coarse Grid Search**
```python
# Parameters tested in combination
learning_rates = [0.0001, 0.001, 0.01]
gamma_values = [0.9, 0.95, 0.99]
hidden_sizes = [64, 128, 256]
```
**Evaluation**: 200 episodes per configuration, average final performance

**Phase 2: Fine-Tuning**
```python
# Refined search around best configurations
learning_rates = [0.0005, 0.001, 0.002]
epsilon_decays = [0.99, 0.995, 0.999]
batch_sizes = [16, 32, 64]
```
**Evaluation**: 500 episodes per configuration, convergence analysis

**Phase 3: Validation**
```python
# Best configuration tested extensively
final_config = {
    'learning_rate': 0.001,
    'gamma': 0.99,
    'epsilon_decay': 0.995,
    'hidden_size': 128,
    'batch_size': 32
}
```
**Validation**: 10 independent runs, 1000 episodes each

### Hyperparameter Sensitivity Analysis

**High Sensitivity:**
- **Learning Rate**: 10x change causes instability or slow learning
- **Epsilon Decay**: Affects exploration-exploitation balance significantly
- **Reward Function**: Small changes in reward structure impact behavior

**Moderate Sensitivity:**
- **Hidden Size**: 64-256 range works well, 128 is sweet spot
- **Batch Size**: 16-64 range acceptable, 32 provides best stability
- **Memory Size**: 5K-20K range effective, 10K chosen for memory efficiency

**Low Sensitivity:**
- **Target Update Frequency**: 50-200 episodes work similarly
- **Network Depth**: 2-4 hidden layers show similar performance
- **Activation Function**: ReLU vs LeakyReLU minimal difference

## Evaluation Metrics

### Primary Performance Metrics

**1. Success Rate**
```python
success_rate = successful_parks / total_parking_attempts
```
**Definition**: Percentage of cars successfully assigned parking spots
**Target**: >80% success rate indicates effective policy
**Measurement**: Calculated over rolling 100-episode window

**2. Average Queue Length**
```python
avg_queue_length = sum(queue_lengths) / num_time_steps
```
**Definition**: Mean number of cars waiting to park per time step
**Target**: <5 cars indicates efficient processing
**Measurement**: Tracked continuously during episodes

**3. Occupancy Rate**
```python
occupancy_rate = occupied_spots / total_available_spots
```
**Definition**: Percentage of parking spots occupied at any time
**Target**: 60-80% indicates efficient space utilization
**Measurement**: Sampled every time step, averaged per episode

**4. Average Episode Reward**
```python
episode_reward = sum(step_rewards) / episode_length
```
**Definition**: Mean reward earned per time step in an episode
**Target**: Increasing trend indicates learning progress
**Measurement**: Tracked for every episode during training

### Secondary Performance Metrics

**5. Distance Efficiency**
```python
avg_distance = sum(parking_distances) / successful_parks
```
**Definition**: Average distance from entry point for parked cars
**Target**: Lower values indicate better user convenience
**Measurement**: Calculated for each successful parking action

**6. Parking Duration Utilization**
```python
utilization = actual_occupancy_time / theoretical_max_occupancy
```
**Definition**: How well the system maximizes parking spot usage over time
**Target**: >0.7 indicates good temporal efficiency
**Measurement**: Analyzed post-episode using parking history

**7. Learning Stability**
```python
stability = 1 - (std_dev(recent_rewards) / mean(recent_rewards))
```
**Definition**: Consistency of performance over recent episodes
**Target**: >0.8 indicates converged, stable policy
**Measurement**: Calculated over sliding window of 50 episodes

### Training Progress Evaluation

**Convergence Criteria:**
1. **Performance Plateau**: Success rate stable for 200+ episodes
2. **Exploration Reduction**: Epsilon < 0.1 with consistent performance
3. **Reward Stability**: Episode rewards within 10% variance over 100 episodes

**Early Stopping Conditions:**
1. **No Improvement**: Success rate hasn't improved for 500 episodes
2. **Performance Degradation**: Success rate drops >20% from peak
3. **Training Timeout**: Maximum 2000 episodes reached

**Performance Benchmarks:**
```python
# Minimum acceptable performance
min_success_rate = 0.70      # 70% of cars parked successfully
max_avg_queue_length = 6     # At most 6 cars waiting on average
min_occupancy_rate = 0.50    # At least 50% spots occupied

# Target performance (well-trained agent)
target_success_rate = 0.85   # 85% success rate
target_avg_queue_length = 3  # Average 3 cars in queue
target_occupancy_rate = 0.70 # 70% occupancy rate
```

### Evaluation Methodology

**Training Evaluation:**
- **Frequency**: Metrics calculated every 100 episodes
- **Window**: Rolling average over last 100 episodes for stability
- **Logging**: All metrics saved to training log for analysis

**Testing Evaluation:**
```python
def evaluate_agent(agent, episodes=100):
    agent.epsilon = 0.0  # Pure exploitation
    test_metrics = []
    
    for episode in range(episodes):
        episode_metrics = run_episode(agent)
        test_metrics.append(episode_metrics)
    
    return aggregate_metrics(test_metrics)
```

**Statistical Significance:**
- **Multiple Runs**: 5 independent training runs for robust evaluation
- **Confidence Intervals**: 95% confidence intervals for all reported metrics
- **Hypothesis Testing**: T-tests for comparing different configurations

This comprehensive evaluation framework ensures that the DQN agent not only learns to solve the parking lot management problem but does so in a stable, efficient, and measurable manner that can be validated and compared across different implementations and improvements.