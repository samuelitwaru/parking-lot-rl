# Problem Formulation: Intelligent Parking Lot Management

## Real-world Problem

### Problem Statement
Urban parking lot management presents a complex optimization challenge that directly impacts city efficiency, environmental sustainability, and user satisfaction. Traditional parking systems operate on a first-come-first-served basis without intelligent allocation, leading to:

- **Inefficient space utilization**: Suboptimal parking spot assignments result in underutilized areas
- **Increased traffic congestion**: Cars circling to find parking spots contribute to urban congestion
- **Environmental impact**: Extended search times increase fuel consumption and emissions
- **Economic losses**: Poor parking management reduces revenue for parking operators
- **User frustration**: Long wait times and difficulty finding spots degrade user experience

### Real-world Applications
This parking lot management problem is representative of broader resource allocation challenges in:

1. **Smart Cities**: Optimizing public parking resources in urban areas
2. **Airport Parking**: Managing multi-level parking structures with varying demand
3. **Shopping Centers**: Balancing customer convenience with space utilization
4. **Hospital Parking**: Critical resource allocation where parking availability can impact emergency response
5. **University Campuses**: Managing limited parking during peak hours

### Why Reinforcement Learning?
Parking lot management is ideally suited for RL because:

- **Dynamic Environment**: Continuous arrival and departure of vehicles creates a constantly changing state
- **Sequential Decision Making**: Each parking assignment affects future options and overall system performance
- **Delayed Rewards**: The quality of parking decisions becomes apparent over time through occupancy rates and user satisfaction
- **Complex State Space**: Multiple factors (queue length, occupancy patterns, time remaining) must be considered simultaneously
- **Adaptability**: RL agents can learn optimal policies that adapt to different demand patterns and scenarios

## Markov Decision Process (MDP) Definition

### State Space (S)

The state space represents the complete information needed to make optimal parking decisions at any given time.

**State Components:**
1. **Grid Occupancy Matrix**: `G ∈ {0, 1, 2}^(H×W)`
   - 0: Empty parking spot
   - 1: Occupied parking spot
   - 2: Entry point
   
2. **Car Time Information**: `T ∈ ℕ^(H×W)`
   - For occupied spots: remaining parking time
   - For empty spots: 0
   
3. **Queue State**: `Q = (q_length, q_cars)`
   - `q_length`: Number of cars waiting to park
   - `q_cars`: List of waiting cars with their intended parking durations

4. **System Statistics**: `Stats = (total_parked, successful_parks, failed_attempts)`

**Formal State Definition:**
```
S = {(G, T, Q, Stats) | G ∈ {0,1,2}^(H×W), T ∈ ℕ^(H×W), Q ∈ Queue_Space, Stats ∈ ℕ³}
```

**State Encoding:**
- **Flattened Grid**: 80-dimensional vector (10×8 grid)
- **Additional Features**: Queue length, occupancy rate, time-based features
- **Total State Dimension**: Approximately 80-100 features

### Action Space (A)

The action space represents all possible parking spot assignments the agent can make.

**Action Definition:**
- Each action `a ∈ A` corresponds to selecting a parking spot position
- Action space size: `|A| = W × H = 80` (for 10×8 grid)
- Action `a = i` represents assigning the next queued car to position `(i // W, i % W)`

**Action Constraints:**
- Only empty spots (grid value = 0) are valid actions
- Entry points cannot be selected as parking spots
- Agent receives feedback for invalid actions (negative reward)

**Available Actions Function:**
```python
def get_available_actions():
    return [i for i in range(W*H) if grid[i//W][i%W] == 0 and (i//W, i%W) != entry_position]
```

### Reward Function (R)

The reward function balances multiple objectives to encourage efficient parking lot management.

**Reward Components:**

1. **Base Parking Reward**: `R_base`
   - +10 for successful parking assignment
   - -5 for invalid action (trying to park in occupied/entry spot)

2. **Distance-based Bonus**: `R_distance`
   - Encourages parking closer to entry point
   - `R_distance = max_bonus × (1 - normalized_distance)`
   - `max_bonus = 5.0`
   - `distance = √((x₁-x₂)² + (y₁-y₂)²)`

3. **Efficiency Reward**: `R_efficiency`
   - Bonus for maintaining high occupancy rates
   - Penalty for leaving cars in queue too long

**Total Reward Function:**
```
R(s, a, s') = R_base(a) + R_distance(s, a) + R_efficiency(s, s')

Where:
- R_base(a) = +10 if valid_action(a), -5 otherwise
- R_distance(s, a) = 5.0 × (1 - distance_to_entry(a) / max_distance)
- R_efficiency(s, s') = occupancy_bonus + queue_penalty
```

**Reward Range**: [-5, +15] per step

### Transition Dynamics (P)

The transition dynamics define how the environment evolves in response to actions and time.

**Deterministic Components:**
1. **Car Placement**: If action is valid, car is placed at selected position
2. **Time Progression**: All parked cars' remaining time decreases by 1
3. **Car Departure**: Cars with time_remaining ≤ 0 leave the parking lot

**Stochastic Components:**
1. **New Car Arrivals**: 
   - `num_new_cars ~ Uniform(0, 3)` per time step
   - Each car's parking duration: `parking_time ~ Uniform(25, 50)`

2. **Queue Management**:
   - New cars added to queue with random parking durations
   - First car in queue is assigned when valid action is taken

**Transition Function:**
```
P(s' | s, a) = P_deterministic(s' | s, a) × P_arrivals(new_cars)

Where:
- P_deterministic handles car placement, departures, time updates
- P_arrivals models stochastic car generation
```

**State Evolution Process:**
1. Agent selects action `a`
2. If valid: place queued car at position `a`
3. Update all car timers (-1 for each parked car)
4. Remove cars with timer ≤ 0
5. Generate new arriving cars
6. Add new cars to queue
7. Compute reward and return new state

### Terminal Conditions

**Episode Termination:**
- **Max Steps Reached**: Episode ends after predetermined number of steps (typically 50-100)
- **No Terminal State**: Parking lot management is a continuing task
- **Early Termination**: Optional termination if queue becomes too large (overflow condition)

### MDP Properties

**Markov Property**: ✅ Current state contains all necessary information for decision making
- Grid occupancy and timers capture current parking situation
- Queue state represents immediate future demand
- No need for historical information beyond current state

**Finite State Space**: ✅ Although large, state space is finite
- Bounded grid size (10×8)
- Finite queue capacity
- Discrete time values

**Discrete Time**: ✅ Decisions made at discrete time steps
- Natural for parking management scenarios
- Allows for clear action-outcome relationships

This MDP formulation captures the essential aspects of parking lot management while remaining computationally tractable for reinforcement learning algorithms.