import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
from typing import List, Tuple
import matplotlib.pyplot as plt
from parking_environment import ParkingLotEnvironment

# Experience replay memory
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class TrainingVisualizer:
    """Pygame visualizer for training process."""
    
    def __init__(self, environment, agent):
        import pygame
        self.env = environment
        self.agent = agent
        self.pygame = pygame
        
        # Initialize pygame
        pygame.init()
        
        # Display settings
        self.cell_size = 60
        self.grid_width = self.env.width * self.cell_size
        self.grid_height = self.env.height * self.cell_size
        self.info_width = 400
        self.window_width = self.grid_width + self.info_width
        self.window_height = max(self.grid_height, 600)
        
        # Create display
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Parking Lot RL Training")
        
        # Colors
        self.colors = {
            'background': (240, 240, 240),
            'empty': (220, 220, 220),
            'occupied': (255, 100, 100),
            'entry': (100, 255, 100),
            'grid_line': (100, 100, 100),
            'text': (50, 50, 50),
            'highlight': (255, 255, 100)
        }
        
        # Fonts
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Training stats
        self.current_episode = 0
        self.current_score = 0
        self.last_action = None
        self.last_reward = 0
        self.last_info = ""
        
        # Timing
        self.clock = pygame.time.Clock()
        self.update_delay = 50  # milliseconds between updates
        
    def reset_episode(self, episode):
        """Reset for new episode."""
        self.current_episode = episode
        self.current_score = 0
        self.last_action = None
        self.last_reward = 0
        self.last_info = ""
        
    def handle_events(self):
        """Handle pygame events. Returns False if should quit."""
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                return False
            elif event.type == self.pygame.KEYDOWN:
                if event.key == self.pygame.K_ESCAPE:
                    return False
                elif event.key == self.pygame.K_SPACE:
                    # Pause/unpause (could be extended)
                    pass
        return True
        
    def update(self, action, reward, info, episode, score):
        """Update visualization with latest training step."""
        self.current_episode = episode
        self.current_score = score
        self.last_action = action
        self.last_reward = reward
        self.last_info = str(info)
        
        # Draw everything
        self.draw()
        
        # Control frame rate
        self.clock.tick(20)  # 20 FPS for training visualization
        
    def draw(self):
        """Draw the current state."""
        # Clear screen
        self.screen.fill(self.colors['background'])
        
        # Draw grid
        self.draw_grid()
        
        # Draw info panel
        self.draw_info_panel()
        
        # Update display
        self.pygame.display.flip()
        
    def draw_grid(self):
        """Draw the parking lot grid."""
        for row in range(self.env.height):
            for col in range(self.env.width):
                x = col * self.cell_size
                y = row * self.cell_size
                rect = self.pygame.Rect(x, y, self.cell_size, self.cell_size)
                
                # Determine cell color based on state
                if (row, col) == (self.env.entry_position[1], self.env.entry_position[0]):
                    color = self.colors['entry']
                    text = "ENTRY"
                elif (row, col) in self.env.cars:
                    car = self.env.cars[(row, col)]
                    color = self.colors['occupied']
                    text = f"C{car.id}\\nT:{car.time_remaining}"
                else:
                    color = self.colors['empty']
                    text = f"{col},{row}"
                
                # Highlight last action
                if self.last_action is not None:
                    action_row, action_col = divmod(self.last_action, self.env.width)
                    if row == action_row and col == action_col:
                        color = self.colors['highlight']
                
                # Draw cell
                self.pygame.draw.rect(self.screen, color, rect)
                self.pygame.draw.rect(self.screen, self.colors['grid_line'], rect, 2)
                
                # Draw text
                lines = text.split('\\n')
                for i, line in enumerate(lines):
                    text_surface = self.small_font.render(line, True, self.colors['text'])
                    text_rect = text_surface.get_rect()
                    text_rect.centerx = rect.centerx
                    text_rect.centery = rect.centery + (i - len(lines)/2 + 0.5) * 12
                    self.screen.blit(text_surface, text_rect)
    
    def draw_info_panel(self):
        """Draw the information panel."""
        x_offset = self.grid_width + 20
        y_offset = 20
        line_height = 25
        
        info_items = [
            f"Episode: {self.current_episode}",
            f"Score: {self.current_score:.1f}",
            f"Epsilon: {self.agent.epsilon:.3f}",
            f"Last Action: {self.last_action}",
            f"Last Reward: {self.last_reward:.2f}",
            f"Action Info: {self.last_info}",
            "",
            "Environment Stats:",
        ]
        
        # Add environment statistics
        try:
            stats = self.env.get_stats()
            info_items.extend([
                f"Total Cars: {stats['total_cars_parked']}",
                f"Successful: {stats['successful_parks']}",
                f"Success Rate: {stats['success_rate']:.1%}",
                f"Occupancy: {stats['occupancy_rate']:.1%}",
            ])
        except:
            info_items.append("Stats unavailable")
        
        info_items.extend([
            "",
            "Queue:",
            f"Cars waiting: {len(self.env.car_queue)}",
        ])
        
        # Show first few cars in queue
        for i, car in enumerate(self.env.car_queue[:5]):
            info_items.append(f"  Car {car.id}: {car.parking_time}min")
        
        if len(self.env.car_queue) > 5:
            info_items.append(f"  ... +{len(self.env.car_queue) - 5} more")
            
        info_items.extend([
            "",
            "Controls:",
            "ESC - Stop training",
            "Close window - Stop training"
        ])
        
        # Draw all info items
        for i, item in enumerate(info_items):
            if item.startswith("Environment Stats:") or item.startswith("Queue:") or item.startswith("Controls:"):
                color = (100, 100, 150)  # Header color
                font = self.font
            else:
                color = self.colors['text']
                font = self.small_font
                
            text_surface = font.render(item, True, color)
            self.screen.blit(text_surface, (x_offset, y_offset + i * line_height))
    
    def cleanup(self):
        """Cleanup pygame resources."""
        self.pygame.quit()

class DQN(nn.Module):
    """Deep Q-Network for parking lot environment."""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class DQNAgent:
    """DQN Agent for parking lot reinforcement learning."""
    
    def __init__(self, state_size: int, action_size: int, 
                 learning_rate: float = 0.001, 
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 memory_size: int = 10000,
                 batch_size: int = 32):
        
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Neural networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQN(state_size, action_size).to(self.device)
        self.target_network = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay
        self.memory = deque(maxlen=memory_size)
        
        # Training statistics
        self.training_scores = []
        self.training_losses = []
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        experience = Experience(state, action, reward, next_state, done)
        self.memory.append(experience)
    
    def act(self, state, available_actions: List[int] = None):
        """Choose action using epsilon-greedy policy."""
        if available_actions is None:
            available_actions = list(range(self.action_size))
        
        if np.random.random() <= self.epsilon:
            # Random action from available actions
            return random.choice(available_actions)
        
        # Q-value based action selection
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        
        # Mask unavailable actions
        masked_q_values = q_values.clone()
        unavailable_actions = [i for i in range(self.action_size) if i not in available_actions]
        if unavailable_actions:
            masked_q_values[0, unavailable_actions] = float('-inf')
        
        return masked_q_values.argmax().item()
    
    def replay(self):
        """Train the model on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.array([e.state for e in batch])).to(self.device)
        actions = torch.LongTensor(np.array([e.action for e in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([e.reward for e in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in batch])).to(self.device)
        dones = torch.BoolTensor(np.array([e.done for e in batch])).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.training_losses.append(loss.item())
    
    def update_target_network(self):
        """Update target network with main network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_scores': self.training_scores,
            'training_losses': self.training_losses
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_scores = checkpoint.get('training_scores', [])
        self.training_losses = checkpoint.get('training_losses', [])

class ParkingLotTrainer:
    """Trainer class for the parking lot RL environment."""
    
    def __init__(self, env: ParkingLotEnvironment):
        self.env = env
        
        # Calculate state and action sizes
        sample_state = env.get_state()
        self.state_size = len(sample_state)
        self.action_size = env.width * env.height
        
        # Create agent
        self.agent = DQNAgent(self.state_size, self.action_size)
        
        # Training parameters
        self.target_update_frequency = 100
        self.save_frequency = 500
        
    def train(self, episodes: int = 2000, max_steps_per_episode: int = 100, visualize: bool = False):
        """Train the DQN agent."""
        print(f"Starting training for {episodes} episodes...")
        print(f"State size: {self.state_size}, Action size: {self.action_size}")
        
        # Initialize pygame visualization if requested
        visualizer = None
        if visualize:
            from pygame_visualizer import ParkingLotVisualizer
            visualizer = TrainingVisualizer(self.env, self.agent)
            print("Pygame visualization enabled during training")
        
        scores_window = deque(maxlen=100)
        
        for episode in range(1, episodes + 1):
            state = self.env.reset()
            score = 0
            
            # Reset visualizer for new episode
            if visualizer:
                visualizer.reset_episode(episode)
            
            for step in range(max_steps_per_episode):
                # Handle pygame events if visualizing
                if visualizer and not visualizer.handle_events():
                    print("Training stopped by user")
                    return self.agent.training_scores
                
                # Get available actions
                available_actions = self.env.get_available_actions()
                if not available_actions:
                    # No available actions, skip this step
                    continue
                
                # Choose action
                action = self.agent.act(state, available_actions)
                
                # Take action
                next_state, reward, done, info = self.env.step(action)
                
                # Store experience
                self.agent.remember(state, action, reward, next_state, done)
                
                # Train agent
                self.agent.replay()
                
                # Update visualization
                if visualizer:
                    visualizer.update(action, reward, info, episode, score)
                
                state = next_state
                score += reward
                
                if done:
                    break
            
            # Update target network
            if episode % self.target_update_frequency == 0:
                self.agent.update_target_network()
            
            # Save model
            if episode % self.save_frequency == 0:
                self.agent.save_model(f'parking_dqn_episode_{episode}.pth')
            
            # Update statistics
            scores_window.append(score)
            self.agent.training_scores.append(score)
            
            # Print progress
            if episode % 100 == 0:
                avg_score = np.mean(scores_window)
                print(f"Episode {episode}/{episodes}, "
                      f"Average Score: {avg_score:.2f}, "
                      f"Epsilon: {self.agent.epsilon:.3f}")
                
                # Print environment statistics
                stats = self.env.get_stats()
                print(f"  Success Rate: {stats['success_rate']:.2%}, "
                      f"Occupancy: {stats['occupancy_rate']:.2%}")
        
        # Clean up visualization
        if visualizer:
            visualizer.cleanup()
            
        print("Training completed!")
        return self.agent.training_scores
    
    def test(self, episodes: int = 100, render: bool = False):
        """Test the trained agent."""
        print(f"Testing agent for {episodes} episodes...")
        
        # Set epsilon to 0 for testing (no exploration)
        old_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.0
        
        test_scores = []
        
        for episode in range(episodes):
            state = self.env.reset()
            score = 0
            
            for step in range(100):  # Max 100 steps per episode
                available_actions = self.env.get_available_actions()
                if not available_actions:
                    continue
                
                action = self.agent.act(state, available_actions)
                next_state, reward, done, info = self.env.step(action)
                
                state = next_state
                score += reward
                
                if done:
                    break
            
            test_scores.append(score)
        
        # Restore epsilon
        self.agent.epsilon = old_epsilon
        
        avg_score = np.mean(test_scores)
        print(f"Test completed. Average score: {avg_score:.2f}")
        
        return test_scores
    
    def plot_training_progress(self):
        """Plot training progress."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot scores
        ax1.plot(self.agent.training_scores)
        ax1.set_title('Training Scores')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Score')
        ax1.grid(True)
        
        # Plot moving average
        if len(self.agent.training_scores) >= 100:
            moving_avg = []
            for i in range(99, len(self.agent.training_scores)):
                moving_avg.append(np.mean(self.agent.training_scores[i-99:i+1]))
            ax1.plot(range(99, len(self.agent.training_scores)), moving_avg, 'r-', label='100-episode average')
            ax1.legend()
        
        # Plot losses
        if self.agent.training_losses:
            ax2.plot(self.agent.training_losses)
            ax2.set_title('Training Losses')
            ax2.set_xlabel('Training Step')
            ax2.set_ylabel('Loss')
            ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

def main():
    """Main function for training the parking lot RL agent."""
    # Create environment
    env = ParkingLotEnvironment(width=10, height=8, entry_position=(0, 4))
    
    # Create trainer
    trainer = ParkingLotTrainer(env)
    
    print("Parking Lot RL Training")
    print("=" * 50)
    
    # Train the agent
    scores = trainer.train(episodes=1000, max_steps_per_episode=50)
    
    # Save final model
    trainer.agent.save_model('parking_dqn_final.pth')
    
    # Test the agent
    test_scores = trainer.test(episodes=50)
    
    # Plot results
    trainer.plot_training_progress()
    
    print("\nTraining Summary:")
    print(f"Final average training score (last 100 episodes): {np.mean(scores[-100:]):.2f}")
    print(f"Average test score: {np.mean(test_scores):.2f}")

if __name__ == "__main__":
    main()