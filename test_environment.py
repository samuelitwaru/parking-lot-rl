#!/usr/bin/env python3
"""
Test script to verify the parking lot environment components work correctly.
"""

import sys
import traceback
from parking_environment import ParkingLotEnvironment, GridState, Car

def test_environment_basic():
    """Test basic environment functionality."""
    print("Testing basic environment functionality...")
    
    try:
        # Create environment
        env = ParkingLotEnvironment(width=5, height=4, entry_position=(0, 1))
        
        # Test reset
        state = env.reset()
        assert len(state) > 0, "State should not be empty"
        
        # Test state representation
        expected_state_size = 5 * 4 + 1  # grid + queue info
        assert len(state) == expected_state_size, f"Expected state size {expected_state_size}, got {len(state)}"
        
        # Test entry position is set correctly
        entry_x, entry_y = env.entry_position
        assert env.grid[entry_y, entry_x] == GridState.ENTRY.value, "Entry position not set correctly"
        
        # Test available actions
        available_actions = env.get_available_actions()
        assert len(available_actions) > 0, "Should have available actions initially"
        
        # Test taking an action
        action = available_actions[0]
        next_state, reward, done, info = env.step(action)
        
        assert len(next_state) == len(state), "State size should remain consistent"
        assert isinstance(reward, (int, float)), "Reward should be a number"
        assert isinstance(info, dict), "Info should be a dictionary"
        
        print("✓ Basic environment functionality test passed")
        return True
        
    except Exception as e:
        print(f"✗ Basic environment test failed: {e}")
        traceback.print_exc()
        return False

def test_car_management():
    """Test car arrival and departure functionality."""
    print("Testing car management...")
    
    try:
        env = ParkingLotEnvironment(width=4, height=3, entry_position=(0, 1))
        env.reset()
        
        # Manually add cars to test
        initial_queue_size = len(env.car_queue)
        env.generate_new_cars()
        
        # Queue might grow (random generation)
        assert len(env.car_queue) >= initial_queue_size, "Queue size should not decrease after generation"
        
        # Test parking a car
        if env.car_queue:
            available_actions = env.get_available_actions()
            if available_actions:
                action = available_actions[0]
                row = action // env.width
                col = action % env.width
                
                state, reward, done, info = env.step(action)
                
                # Check if car was parked
                if info.get('successful_park', False):
                    assert env.grid[row, col] == GridState.OCCUPIED.value, "Grid should show occupied after parking"
                    assert (col, row) in env.cars, "Car should be in cars dictionary"
        
        # Test car time updates
        original_cars = dict(env.cars)
        env.update_parked_cars()
        
        # Cars should have updated time (this is hard to test deterministically due to randomness)
        print("✓ Car management test passed")
        return True
        
    except Exception as e:
        print(f"✗ Car management test failed: {e}")
        traceback.print_exc()
        return False

def test_reward_calculation():
    """Test reward calculation system."""
    print("Testing reward calculation...")
    
    try:
        env = ParkingLotEnvironment(width=4, height=3, entry_position=(0, 1))
        env.reset()
        
        # Test distance bonus calculation
        close_position = (1, 1)  # Close to entry at (0, 1)
        far_position = (3, 2)    # Far from entry
        
        close_bonus = env.calculate_distance_bonus(close_position)
        far_bonus = env.calculate_distance_bonus(far_position)
        
        assert close_bonus > far_bonus, "Closer position should have higher bonus"
        assert close_bonus >= 0, "Distance bonus should be non-negative"
        assert far_bonus >= 0, "Distance bonus should be non-negative"
        
        # Test position validity
        assert env.is_valid_position((1, 1)), "Normal position should be valid"
        assert not env.is_valid_position((-1, 1)), "Out of bounds position should be invalid"
        assert not env.is_valid_position(env.entry_position), "Entry position should be invalid for parking"
        
        print("✓ Reward calculation test passed")
        return True
        
    except Exception as e:
        print(f"✗ Reward calculation test failed: {e}")
        traceback.print_exc()
        return False

def test_occupancy_and_stats():
    """Test occupancy rate and statistics calculation."""
    print("Testing occupancy and statistics...")
    
    try:
        env = ParkingLotEnvironment(width=3, height=3, entry_position=(1, 1))
        env.reset()
        
        # Initial occupancy should be 0
        initial_occupancy = env.get_occupancy_rate()
        assert initial_occupancy == 0.0, f"Initial occupancy should be 0, got {initial_occupancy}"
        
        # Get initial stats
        stats = env.get_stats()
        required_keys = ['total_cars_parked', 'successful_parks', 'failed_attempts', 
                        'occupancy_rate', 'cars_in_queue', 'success_rate']
        
        for key in required_keys:
            assert key in stats, f"Missing required stat: {key}"
        
        # Test that stats are reasonable
        assert stats['occupancy_rate'] >= 0.0, "Occupancy rate should be non-negative"
        assert stats['success_rate'] >= 0.0, "Success rate should be non-negative"
        assert stats['success_rate'] <= 1.0, "Success rate should not exceed 1.0"
        
        print("✓ Occupancy and statistics test passed")
        return True
        
    except Exception as e:
        print(f"✗ Occupancy and statistics test failed: {e}")
        traceback.print_exc()
        return False

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test core environment
        from parking_environment import ParkingLotEnvironment, GridState, Car
        
        # Test pygame visualizer (might fail if pygame not installed)
        try:
            from pygame_visualizer import ParkingLotVisualizer
            pygame_available = True
        except ImportError:
            pygame_available = False
            print("  Note: pygame not available, visualization will not work")
        
        # Test RL trainer (might fail if torch not installed)
        try:
            from rl_trainer import ParkingLotTrainer, DQNAgent
            torch_available = True
        except ImportError:
            torch_available = False
            print("  Note: torch not available, RL training will not work")
        
        print("✓ Import test passed")
        return True
        
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests."""
    print("Parking Lot Environment Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_environment_basic,
        test_car_management,
        test_reward_calculation,
        test_occupancy_and_stats
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()  # Empty line between tests
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! ✓")
        return True
    else:
        print("Some tests failed! ✗")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)