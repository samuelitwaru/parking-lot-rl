from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import subprocess
import threading
import time
import json
import os
import signal
import psutil
from parking_environment import ParkingLotEnvironment
from rl_trainer import ParkingLotTrainer
from main import EnhancedParkingLotVisualizer
import pygame
import sys

app = Flask(__name__)
app.config['SECRET_KEY'] = 'parking-demo-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
demo_process = None
demo_thread = None
is_demo_running = False
demo_env = None
demo_trainer = None

class WebDemo:
    """Web-compatible demo runner that mimics the pygame demo."""
    
    def __init__(self):
        self.env = ParkingLotEnvironment(width=10, height=8, entry_position=(0, 4))
        self.trainer = None
        self.auto_mode = False
        self.is_running = False
        self.demo_thread = None
        
    def setup_trainer(self):
        """Set up the trainer and try to load a model."""
        try:
            self.trainer = ParkingLotTrainer(self.env)
            
            # Try to load a trained model
            model_files = [
                'parking_dqn_trained.pth',
                'parking_dqn_final.pth',
                'models/parking_dqn_trained.pth'
            ]
            
            for model_file in model_files:
                if os.path.exists(model_file):
                    try:
                        self.trainer.agent.load_model(model_file)
                        self.trainer.agent.epsilon = 0.0  # No exploration for demo
                        print(f"Loaded model: {model_file}")
                        return True
                    except Exception as e:
                        print(f"Failed to load {model_file}: {e}")
                        continue
            
            # If no model found, create a simple one
            print("No trained model found. Training a quick model...")
            self.trainer.train(episodes=50, max_steps_per_episode=20)
            self.trainer.agent.epsilon = 0.0
            return True
            
        except Exception as e:
            print(f"Error setting up trainer: {e}")
            return False
    
    def get_state_dict(self):
        """Get current state as dictionary."""
        grid_state = []
        for row in range(self.env.height):
            grid_row = []
            for col in range(self.env.width):
                cell_info = {
                    'type': 'empty',
                    'car_id': None,
                    'time_remaining': None,
                    'coordinates': f'{col},{row}'
                }
                
                if (col, row) == self.env.entry_position:
                    cell_info['type'] = 'entry'
                elif self.env.grid[row, col] == 1:  # Occupied
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
            'has_trainer': self.trainer is not None,
            'width': self.env.width,
            'height': self.env.height,
            'entry_position': self.env.entry_position
        }
    
    def take_action(self, action):
        """Take an action in the environment."""
        try:
            available_actions = self.env.get_available_actions()
            
            if action not in available_actions:
                return {
                    'success': False,
                    'message': 'Invalid action: spot not available or occupied',
                    'reward': -5.0
                }
            
            state, reward, done, info = self.env.step(action)
            
            return {
                'success': True,
                'reward': reward,
                'info': info,
                'message': f"Reward: {reward:.1f}, {'Successful park!' if info.get('successful_park') else 'Action taken'}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Error taking action: {str(e)}',
                'reward': 0
            }
    
    def auto_step(self):
        """Take an automatic step using the agent or random action."""
        try:
            available_actions = self.env.get_available_actions()
            if not available_actions:
                return None
            
            if self.trainer and self.trainer.agent:
                # Use trained agent
                state = self.env.get_state()
                action = self.trainer.agent.act(state, available_actions)
                action_type = "AI Agent"
            else:
                # Random action
                import random
                action = random.choice(available_actions)
                action_type = "Random"
            
            result = self.take_action(action)
            result['action'] = action
            result['action_type'] = action_type
            
            # Convert action to coordinates for display
            row = action // self.env.width
            col = action % self.env.width
            result['coordinates'] = f"({col},{row})"
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Error in auto step: {str(e)}',
                'action_type': 'Error'
            }
    
    def reset_environment(self):
        """Reset the environment."""
        try:
            self.env.reset()
            return True
        except Exception as e:
            print(f"Error resetting environment: {e}")
            return False
    
    def start_auto_mode(self):
        """Start automatic demo mode."""
        if not self.auto_mode and not self.is_running:
            self.auto_mode = True
            self.is_running = True
            self.demo_thread = threading.Thread(target=self._auto_demo_loop)
            self.demo_thread.daemon = True
            self.demo_thread.start()
            return True
        return False
    
    def stop_auto_mode(self):
        """Stop automatic demo mode."""
        self.auto_mode = False
        self.is_running = False
        if self.demo_thread:
            self.demo_thread = None
        return True
    
    def _auto_demo_loop(self):
        """Main auto demo loop."""
        while self.is_running and self.auto_mode:
            try:
                result = self.auto_step()
                if result:
                    # Emit update to all clients
                    socketio.emit('demo_update', {
                        'state': self.get_state_dict(),
                        'action_result': result
                    })
                
                time.sleep(2)  # Wait 2 seconds between actions
                
            except Exception as e:
                print(f"Error in auto demo loop: {e}")
                break
        
        self.is_running = False

# Global demo instance
web_demo = WebDemo()

@app.route('/')
def index():
    """Main demo page."""
    return render_template('demo.html')

@app.route('/start_demo')
def start_demo():
    """Initialize and start the demo."""
    global web_demo
    
    try:
        # Setup trainer
        trainer_ready = web_demo.setup_trainer()
        
        return jsonify({
            'success': True,
            'message': 'Demo initialized successfully!',
            'trainer_ready': trainer_ready,
            'state': web_demo.get_state_dict()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error starting demo: {str(e)}'
        })

@app.route('/get_state')
def get_state():
    """Get current demo state."""
    try:
        return jsonify({
            'success': True,
            'state': web_demo.get_state_dict()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error getting state: {str(e)}'
        })

@app.route('/action/<int:action>')
def take_action(action):
    """Take a manual action."""
    try:
        result = web_demo.take_action(action)
        return jsonify({
            'success': True,
            'action_result': result,
            'state': web_demo.get_state_dict()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error taking action: {str(e)}'
        })

@app.route('/auto_step')
def auto_step():
    """Take one automatic step."""
    try:
        result = web_demo.auto_step()
        return jsonify({
            'success': True,
            'action_result': result,
            'state': web_demo.get_state_dict()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error in auto step: {str(e)}'
        })

@app.route('/toggle_auto_mode')
def toggle_auto_mode():
    """Toggle automatic demo mode."""
    try:
        if web_demo.auto_mode:
            success = web_demo.stop_auto_mode()
            message = "Auto mode stopped"
        else:
            success = web_demo.start_auto_mode()
            message = "Auto mode started"
        
        return jsonify({
            'success': success,
            'message': message,
            'auto_mode': web_demo.auto_mode,
            'state': web_demo.get_state_dict()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error toggling auto mode: {str(e)}'
        })

@app.route('/reset')
def reset_demo():
    """Reset the demo environment."""
    try:
        # Stop auto mode first
        web_demo.stop_auto_mode()
        
        # Reset environment
        success = web_demo.reset_environment()
        
        return jsonify({
            'success': success,
            'message': 'Demo reset successfully!' if success else 'Failed to reset demo',
            'state': web_demo.get_state_dict()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error resetting demo: {str(e)}'
        })

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print(f'Client connected: {request.sid}')
    emit('connected', {
        'message': 'Connected to Parking Lot Demo',
        'state': web_demo.get_state_dict()
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print(f'Client disconnected: {request.sid}')

def cleanup():
    """Cleanup function to stop demo when app shuts down."""
    global web_demo
    if web_demo:
        web_demo.stop_auto_mode()

import atexit
atexit.register(cleanup)

if __name__ == '__main__':
    print("🚗 Starting Parking Lot Demo Web Application...")
    print("📱 Access the demo at: http://localhost:5000")
    print("🎮 Features:")
    print("   - Interactive parking lot grid")
    print("   - AI agent demonstration")
    print("   - Manual parking spot selection")
    print("   - Real-time statistics")
    print("   - Auto mode with trained agent")
    
    try:
        socketio.run(app, debug=False, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
        cleanup()
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        cleanup()