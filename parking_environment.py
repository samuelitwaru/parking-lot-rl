import numpy as np
import random
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

class GridState(Enum):
    EMPTY = 0
    OCCUPIED = 1
    ENTRY = 2

@dataclass
class Car:
    id: int
    position: Tuple[int, int]
    parking_time: int
    time_remaining: int

class ParkingLotEnvironment:
    """
    Parking lot environment for reinforcement learning with pygame visualization.
    """
    PARKING_TIME_RANGE = (25, 50)
    
    def __init__(self, width: int = 10, height: int = 8, entry_position: Tuple[int, int] = (0, 4)):
        self.width = width
        self.height = height
        self.entry_position = entry_position
        
        # Initialize grid
        self.grid = np.zeros((height, width), dtype=int)
        self.grid[entry_position[1], entry_position[0]] = GridState.ENTRY.value
        
        # Car management
        self.cars: Dict[Tuple[int, int], Car] = {}
        self.car_counter = 0
        self.car_queue: List[Car] = []
        
        # Rewards
        self.positive_reward = 10.0
        self.negative_reward = -5.0
        self.max_distance_bonus = 5.0
        
        # Statistics
        self.total_cars_parked = 0
        self.successful_parks = 0
        self.failed_attempts = 0
        
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state."""
        self.grid = np.zeros((self.height, self.width), dtype=int)
        self.grid[self.entry_position[1], self.entry_position[0]] = GridState.ENTRY.value
        self.cars.clear()
        self.car_counter = 0
        self.car_queue.clear()
        self.total_cars_parked = 0
        self.successful_parks = 0
        self.failed_attempts = 0
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """Get current state of the parking lot."""
        # Create state representation combining grid and car queue info
        state = self.grid.copy().flatten()
        queue_size = min(len(self.car_queue), 5)  # Limit queue info to 5 cars
        queue_info = np.array([queue_size / 5.0])  # Normalize queue size
        return np.concatenate([state, queue_info])
    
    def generate_new_cars(self) -> None:
        """Generate new cars arriving at the parking lot."""
        # Random number of new cars (0-3 per step)
        num_new_cars = random.randint(0, 3)
        
        for _ in range(num_new_cars):
            parking_time = random.randint(self.PARKING_TIME_RANGE[0], self.PARKING_TIME_RANGE[1])  # Random parking duration
            new_car = Car(
                id=self.car_counter,
                position=(-1, -1),  # Not yet parked
                parking_time=parking_time,
                time_remaining=parking_time
            )
            self.car_queue.append(new_car)
            self.car_counter += 1
    
    def update_parked_cars(self) -> None:
        """Update time for parked cars and remove those whose time is up."""
        positions_to_remove = []
        
        for position, car in self.cars.items():
            car.time_remaining -= 1
            if car.time_remaining <= 0:
                positions_to_remove.append(position)
        
        # Remove cars whose parking time is up
        for position in positions_to_remove:
            del self.cars[position]
            x, y = position  # position is (x, y)
            self.grid[y, x] = GridState.EMPTY.value  # grid is indexed as [row, col] = [y, x]
    
    def calculate_distance_bonus(self, position: Tuple[int, int]) -> float:
        """Calculate bonus reward based on distance from entry."""
        entry_x, entry_y = self.entry_position
        pos_x, pos_y = position
        
        # Calculate Manhattan distance
        distance = abs(entry_x - pos_x) + abs(entry_y - pos_y)
        max_distance = self.width + self.height  # Maximum possible distance
        
        # Closer positions get higher bonus
        normalized_distance = distance / max_distance
        bonus = self.max_distance_bonus * (1.0 - normalized_distance)
        return bonus
    
    def is_valid_position(self, position: Tuple[int, int]) -> bool:
        """Check if position is valid (within bounds and not entry)."""
        x, y = position
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        if (x, y) == self.entry_position:
            return False
        return True
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Grid position as single integer (row * width + col)
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Convert action to grid coordinates
        row = action // self.width
        col = action % self.width
        position = (col, row)  # (x, y) format
        
        reward = 0.0
        info = {'successful_park': False, 'invalid_action': False}
        
        # Check if position is valid
        if not self.is_valid_position(position):
            reward = self.negative_reward
            info['invalid_action'] = True
        else:
            # Check if position is occupied
            if self.grid[row, col] == GridState.OCCUPIED.value:
                # Negative reward for choosing occupied spot
                reward = self.negative_reward
                self.failed_attempts += 1
            else:
                # Positive reward for choosing empty spot
                reward = self.positive_reward
                
                # Add distance bonus
                distance_bonus = self.calculate_distance_bonus(position)
                reward += distance_bonus
                
                # Park a car if there's one in queue
                if self.car_queue:
                    car = self.car_queue.pop(0)
                    car.position = position
                    self.cars[position] = car
                    self.grid[row, col] = GridState.OCCUPIED.value
                    self.successful_parks += 1
                    self.total_cars_parked += 1
                    info['successful_park'] = True
        
        # Update environment
        self.update_parked_cars()
        self.generate_new_cars()
        
        # Check if done (for episodic training)
        done = False  # Can be modified based on specific training criteria
        
        next_state = self.get_state()
        return next_state, reward, done, info
    
    def get_available_actions(self) -> List[int]:
        """Get list of available (empty) parking spots as action indices."""
        available_actions = []
        for row in range(self.height):
            for col in range(self.width):
                if (col, row) != self.entry_position and self.grid[row, col] == GridState.EMPTY.value:
                    action = row * self.width + col
                    available_actions.append(action)
        return available_actions
    
    def get_occupancy_rate(self) -> float:
        """Calculate current occupancy rate of the parking lot."""
        total_spots = self.width * self.height - 1  # Exclude entry
        occupied_spots = len(self.cars)
        return occupied_spots / total_spots if total_spots > 0 else 0.0
    
    def get_stats(self) -> Dict:
        """Get current statistics."""
        return {
            'total_cars_parked': self.total_cars_parked,
            'successful_parks': self.successful_parks,
            'failed_attempts': self.failed_attempts,
            'occupancy_rate': self.get_occupancy_rate(),
            'cars_in_queue': len(self.car_queue),
            'success_rate': self.successful_parks / max(1, self.successful_parks + self.failed_attempts)
        }