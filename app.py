from flask import Flask, render_template, request, jsonify, session
import json
import numpy as np
import threading
import time
import uuid
from parking_environment import ParkingLotEnvironment, GridState
from rl_trainer import ParkingLotTrainer
import os
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'parking-lot-rl-secret-key'

# Global storage for environments and trainers
environments = {}
trainers = {}
training_threads = {}

# Global status tracking for AJAX polling
environment_updates = {}  # Store latest updates for each environment
training_status = {}      # Store training progress
auto_mode_status = {}     # Store auto mode progress

class WebParkingEnvironment:
    """Wrapper for parking environment with web-specific features."""
    
    def __init__(self, env_id, width=10, height=8, entry_position=(0, 4)):
        self.env_id = env_id
        self.env = ParkingLotEnvironment(width, height, entry_position)
        self.trainer = None
        self.auto_mode = False
        self.auto_thread = None
        self.is_training = False
        
    def get_state_dict(self):
        """Get current state as dictionary for JSON serialization."""
        grid_state = []
        for row in range(self.env.height):
            grid_row = []
            for col in range(self.env.width):
                cell_info = {
                    'type': 'empty',
                    'car_id': None,
                    'time_remaining': None
                }
                
                if (col, row) == self.env.entry_position:
                    cell_info['type'] = 'entry'
                elif self.env.grid[row, col] == GridState.OCCUPIED.value:
                    cell_info['type'] = 'occupied'
                    if (col, row) in self.env.cars:
                        car = self.env.cars[(col, row)]
                        cell_info['car_id'] = car.id
                        cell_info['time_remaining'] = car.time_remaining
                
                grid_row.append(cell_info)
            grid_state.append(grid_row)
        
        stats = self.env.get_stats()
        
        return {
            'grid': grid_state,
            'stats': stats,
            'queue_length': len(self.env.car_queue),
            'queue_cars': [{'id': car.id, 'parking_time': car.parking_time} 
                          for car in self.env.car_queue[:5]],
            'auto_mode': self.auto_mode,
            'is_training': self.is_training,
            'width': self.env.width,
            'height': self.env.height,
            'entry_position': self.env.entry_position
        }
    
    def step_action(self, action):
        """Take an action and return result."""
        available_actions = self.env.get_available_actions()
        
        if action not in available_actions:
            return {
                'success': False,
                'message': 'Invalid action: spot not available',
                'reward': self.env.negative_reward
            }
        
        state, reward, done, info = self.env.step(action)
        
        return {
            'success': True,
            'reward': reward,
            'info': info,
            'message': f"Reward: {reward:.1f}"
        }
    
    def reset_environment(self):
        """Reset the environment."""
        self.env.reset()
        self.stop_auto_mode()
        return self.get_state_dict()
    
    def start_auto_mode(self):
        """Start automatic mode."""
        if not self.auto_mode:
            self.auto_mode = True
            self.auto_thread = threading.Thread(target=self._auto_mode_loop)
            self.auto_thread.daemon = True
            self.auto_thread.start()
    
    def stop_auto_mode(self):
        """Stop automatic mode."""
        self.auto_mode = False
        if self.auto_thread:
            self.auto_thread = None
        # Clear auto mode status
        if self.env_id in auto_mode_status:
            auto_mode_status[self.env_id]['active'] = False
    
    def _auto_mode_loop(self):
        """Auto mode loop for automatic actions."""
        step_count = 0
        while self.auto_mode:
            try:
                step_count += 1
                print(f"Auto mode step {step_count} for environment {self.env_id}")
                
                available_actions = self.env.get_available_actions()
                if available_actions:
                    if self.trainer and self.trainer.agent:
                        # Use trained agent
                        state = self.env.get_state()
                        action = self.trainer.agent.act(state, available_actions)
                        action_type = "Agent"
                    else:
                        # Random action
                        action = np.random.choice(available_actions)
                        action_type = "Random"
                    
                    result = self.step_action(action)
                    print(f"Auto mode - Env: {self.env_id}, Step: {step_count}, Action: {action}, Result: {result}")
                    
                    # Create update data
                    update_data = {
                        'env_id': self.env_id,
                        'state': self.get_state_dict(),
                        'action_result': {
                            'action': int(action),
                            'type': action_type,
                            'result': result,
                            'step': step_count
                        }
                    }
                    
                    # Store updates for AJAX polling
                    print(f"Storing auto mode update for environment: {self.env_id}")
                    environment_updates[self.env_id] = {
                        'timestamp': time.time(),
                        'type': 'environment_update',
                        'data': update_data
                    }
                    
                    # Store auto mode heartbeat
                    auto_mode_status[self.env_id] = {
                        'step': step_count,
                        'timestamp': time.time(),
                        'active': True
                    }
                else:
                    print(f"No available actions for environment {self.env_id}")
                
                time.sleep(2)  # Wait 2 seconds between actions
            except Exception as e:
                print(f"Error in auto mode: {e}")
                break

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/create_environment', methods=['POST'])
def create_environment():
    """Create a new parking lot environment."""
    data = request.json
    width = data.get('width', 10)
    height = data.get('height', 8)
    entry_x = data.get('entry_x', 0)
    entry_y = data.get('entry_y', 4)
    
    env_id = str(uuid.uuid4())
    environments[env_id] = WebParkingEnvironment(
        env_id, width, height, (entry_x, entry_y)
    )
    
    # Store environment creation event for AJAX polling
    environment_updates[env_id] = {
        'timestamp': time.time(),
        'type': 'environment_created',
        'data': {
            'env_id': env_id,
            'message': f'Environment {env_id} created successfully'
        }
    }
    
    print(f"Created environment {env_id}")
    
    return jsonify({
        'success': True,
        'env_id': env_id,
        'state': environments[env_id].get_state_dict()
    })

@app.route('/get_environment/<env_id>')
def get_environment(env_id):
    """Get current environment state."""
    if env_id not in environments:
        return jsonify({'success': False, 'message': 'Environment not found'})
    
    return jsonify({
        'success': True,
        'state': environments[env_id].get_state_dict()
    })

@app.route('/action/<env_id>', methods=['POST'])
def take_action(env_id):
    """Take an action in the environment."""
    if env_id not in environments:
        return jsonify({'success': False, 'message': 'Environment not found'})
    
    data = request.json
    action = data.get('action')
    
    if action is None:
        return jsonify({'success': False, 'message': 'Action required'})
    
    result = environments[env_id].step_action(action)
    
    return jsonify({
        'success': True,
        'action_result': result,
        'state': environments[env_id].get_state_dict()
    })

@app.route('/reset/<env_id>', methods=['POST'])
def reset_environment(env_id):
    """Reset an environment."""
    if env_id not in environments:
        return jsonify({'success': False, 'message': 'Environment not found'})
    
    state = environments[env_id].reset_environment()
    
    return jsonify({
        'success': True,
        'state': state
    })

@app.route('/auto_mode/<env_id>', methods=['POST'])
def toggle_auto_mode(env_id):
    """Toggle automatic mode."""
    if env_id not in environments:
        return jsonify({'success': False, 'message': 'Environment not found'})
    
    env = environments[env_id]
    
    if env.auto_mode:
        env.stop_auto_mode()
        message = "Auto mode stopped"
    else:
        env.start_auto_mode()
        message = "Auto mode started"
    
    return jsonify({
        'success': True,
        'message': message,
        'auto_mode': env.auto_mode,
        'state': env.get_state_dict()
    })

@app.route('/train/<env_id>', methods=['POST'])
def start_training(env_id):
    """Start training for an environment."""
    if env_id not in environments:
        return jsonify({'success': False, 'message': 'Environment not found'})
    
    data = request.json
    episodes = data.get('episodes', 100)
    max_steps = data.get('max_steps_per_episode', 50)
    
    env = environments[env_id]
    
    if env.is_training:
        return jsonify({'success': False, 'message': 'Training already in progress'})
    
    # Create trainer
    env.trainer = ParkingLotTrainer(env.env)
    env.is_training = True
    
    # Start training in background thread
    def training_thread():
        try:
            # Store training started status
            print(f"Training started for environment {env_id}")
            training_status[env_id] = {
                'status': 'started',
                'episodes': episodes,
                'max_steps': max_steps,
                'current_episode': 0,
                'timestamp': time.time()
            }
            
            scores = env.trainer.train(episodes=episodes, max_steps_per_episode=max_steps)
            
            # Save trained model
            model_path = f'models/parking_model_{env_id}.pth'
            os.makedirs('models', exist_ok=True)
            env.trainer.agent.save_model(model_path)
            
            # Store training completion status
            print(f"Training completed for environment {env_id}")
            training_status[env_id] = {
                'status': 'completed',
                'success': True,
                'average_score': np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores),
                'total_episodes': len(scores),
                'model_path': model_path,
                'timestamp': time.time()
            }
            
        except Exception as e:
            print(f"Training failed for environment {env_id}: {str(e)}")
            training_status[env_id] = {
                'status': 'failed',
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            }
        finally:
            env.is_training = False
    
    thread = threading.Thread(target=training_thread)
    thread.daemon = True
    thread.start()
    training_threads[env_id] = thread
    
    return jsonify({
        'success': True,
        'message': f'Training started for {episodes} episodes',
        'state': env.get_state_dict()
    })

@app.route('/load_model/<env_id>', methods=['POST'])
def load_model(env_id):
    """Load a trained model."""
    if env_id not in environments:
        return jsonify({'success': False, 'message': 'Environment not found'})
    
    data = request.json
    model_path = data.get('model_path', f'models/parking_model_{env_id}.pth')
    
    env = environments[env_id]
    
    try:
        if not env.trainer:
            env.trainer = ParkingLotTrainer(env.env)
        
        env.trainer.agent.load_model(model_path)
        env.trainer.agent.epsilon = 0.0  # No exploration for demo
        
        return jsonify({
            'success': True,
            'message': 'Model loaded successfully',
            'state': env.get_state_dict()
        })
    
    except FileNotFoundError:
        return jsonify({
            'success': False,
            'message': 'Model file not found'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error loading model: {str(e)}'
        })

@app.route('/api/poll/<env_id>')
def poll_updates(env_id):
    """Poll for environment updates via AJAX."""
    if env_id not in environments:
        return jsonify({'success': False, 'message': 'Environment not found'})
    
    updates = []
    
    # Check for environment updates
    if env_id in environment_updates:
        updates.append(environment_updates[env_id])
        # Remove update after sending (or keep for multiple clients)
        
    # Check for training status
    if env_id in training_status:
        updates.append({
            'timestamp': training_status[env_id]['timestamp'],
            'type': 'training_status',
            'data': training_status[env_id]
        })
    
    # Check for auto mode status
    if env_id in auto_mode_status:
        updates.append({
            'timestamp': auto_mode_status[env_id]['timestamp'],
            'type': 'auto_mode_heartbeat',
            'data': auto_mode_status[env_id]
        })
    
    return jsonify({
        'success': True,
        'updates': updates,
        'current_state': environments[env_id].get_state_dict(),
        'timestamp': time.time()
    })

@app.route('/api/test_connection')
def test_connection():
    """Test API connection."""
    return jsonify({
        'success': True,
        'message': 'API connection working!',
        'timestamp': time.time()
    })

if __name__ == '__main__':
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)