#!/usr/bin/env python3
"""
Simple example demonstrating the parking lot environment functionality.
This script shows basic usage without requiring pygame or training.
"""

import sys
import random
from parking_environment import ParkingLotEnvironment

def print_grid(env):
    """Print a text representation of the parking lot grid."""
    print("\nParking Lot State:")
    print("=" * (env.width * 3 + 1))
    
    for row in range(env.height):
        print("|", end="")
        for col in range(env.width):
            if (col, row) == env.entry_position:
                symbol = "E"  # Entry
            elif env.grid[row, col] == 1:  # Occupied
                if (col, row) in env.cars:
                    car = env.cars[(col, row)]
                    symbol = str(car.time_remaining)[:1]  # Show remaining time (first digit)
                else:
                    symbol = "X"
            else:
                symbol = " "  # Empty
            
            print(f"{symbol:2}|", end="")
        print()
    
    print("=" * (env.width * 3 + 1))
    print("Legend: E=Entry, Numbers=Car(time remaining), Space=Empty")

def print_stats(env):
    """Print environment statistics."""
    stats = env.get_stats()
    print(f"\nStatistics:")
    print(f"  Cars parked: {stats['total_cars_parked']}")
    print(f"  Success rate: {stats['success_rate']:.1%}")
    print(f"  Occupancy: {stats['occupancy_rate']:.1%}")
    print(f"  Queue length: {stats['cars_in_queue']}")

def basic_demo():
    """Demonstrate basic environment functionality."""
    print("Parking Lot Environment Demo")
    print("=" * 40)
    
    # Create environment
    env = ParkingLotEnvironment(width=8, height=6, entry_position=(0, 2))
    
    print("Initial state:")
    print_grid(env)
    print_stats(env)
    
    # Run simulation for several steps
    for step in range(10):
        print(f"\n--- Step {step + 1} ---")
        
        # Get available actions
        available_actions = env.get_available_actions()
        
        if available_actions:
            # Choose random action
            action = random.choice(available_actions)
            
            # Convert action to coordinates for display
            row = action // env.width
            col = action % env.width
            
            # Take action
            state, reward, done, info = env.step(action)
            
            print(f"Action: Park at ({col}, {row})")
            print(f"Reward: {reward:.1f}")
            print(f"Info: {info}")
        else:
            print("No available parking spots!")
            # Still need to step to update cars and generate new ones
            env.step(0)  # Invalid action, but updates environment
        
        print_grid(env)
        print_stats(env)
        
        # Stop if no cars in queue and lot is full
        if len(env.car_queue) == 0 and len(available_actions) == 0:
            print("\nStopping: No cars waiting and no available spots")
            break

def interactive_demo():
    """Interactive demo where user can choose actions."""
    print("Interactive Parking Lot Demo")
    print("=" * 40)
    print("Commands: 'help', 'quit', 'reset', 'auto', or grid coordinates like '2,3'")
    
    env = ParkingLotEnvironment(width=6, height=5, entry_position=(0, 2))
    
    while True:
        print_grid(env)
        print_stats(env)
        
        available_actions = env.get_available_actions()
        if available_actions:
            print(f"\nAvailable spots: {len(available_actions)}")
            # Show available coordinates
            coords = []
            for action in available_actions:
                row = action // env.width
                col = action % env.width
                coords.append(f"({col},{row})")
            print(f"Coordinates: {', '.join(coords)}")
        else:
            print("\nNo available spots!")
        
        user_input = input("\nEnter command: ").strip().lower()
        
        if user_input == 'quit':
            break
        elif user_input == 'help':
            print("\nCommands:")
            print("  x,y     - Park at coordinates (x,y)")
            print("  auto    - Take random action")
            print("  reset   - Reset environment")
            print("  quit    - Exit demo")
        elif user_input == 'reset':
            env.reset()
            print("Environment reset!")
        elif user_input == 'auto':
            if available_actions:
                action = random.choice(available_actions)
                row = action // env.width
                col = action % env.width
                state, reward, done, info = env.step(action)
                print(f"Auto-selected: ({col},{row}), Reward: {reward:.1f}")
            else:
                print("No available actions for auto mode")
        else:
            # Try to parse coordinates
            try:
                parts = user_input.split(',')
                if len(parts) == 2:
                    col, row = int(parts[0].strip()), int(parts[1].strip())
                    action = row * env.width + col
                    
                    state, reward, done, info = env.step(action)
                    print(f"Result: Reward={reward:.1f}, Info={info}")
                else:
                    print("Invalid input. Use format 'x,y' or type 'help'")
            except ValueError:
                print("Invalid coordinates. Use numbers like '2,3' or type 'help'")

def main():
    """Main function with demo selection."""
    print("Parking Lot Environment Examples")
    print("Choose demo type:")
    print("1. Basic automated demo")
    print("2. Interactive demo")
    
    try:
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == '1':
            basic_demo()
        elif choice == '2':
            interactive_demo()
        else:
            print("Invalid choice. Running basic demo...")
            basic_demo()
            
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()