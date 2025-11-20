import pygame
import random
from parking_environment import ParkingLotEnvironment
from pygame_visualizer import ParkingLotVisualizer
from rl_trainer import ParkingLotTrainer
import sys
import argparse

def run_simulation():
    """Run the interactive pygame simulation."""
    print("Starting Parking Lot Simulation...")
    env = ParkingLotEnvironment(width=10, height=8, entry_position=(0, 4))
    visualizer = ParkingLotVisualizer(env)
    visualizer.run()

def run_training(simulate=False):
    """Run the reinforcement learning training."""
    print("Starting RL Training...")
    env = ParkingLotEnvironment(width=10, height=8, entry_position=(0, 4))
    trainer = ParkingLotTrainer(env)
    
    # Train the agent
    scores = trainer.train(episodes=1000, max_steps_per_episode=50, visualize=simulate)
    
    # Save the model
    trainer.agent.save_model('parking_dqn_trained.pth')
    
    # Test the agent
    test_scores = trainer.test(episodes=100)
    
    # Plot training progress
    trainer.plot_training_progress()
    
    print(f"\nTraining completed!")
    print(f"Average training score (last 100 episodes): {scores[-100:] if len(scores) >= 100 else scores}")
    print(f"Average test score: {sum(test_scores)/len(test_scores):.2f}")

def run_demo_with_trained_agent():
    """Run simulation with a trained agent."""
    try:
        print("Loading trained agent...")
        env = ParkingLotEnvironment(width=10, height=8, entry_position=(0, 4))
        trainer = ParkingLotTrainer(env)
        
        # Try to load a trained model
        try:
            trainer.agent.load_model('parking_dqn_trained.pth')
            print("Loaded trained model successfully!")
        except FileNotFoundError:
            print("No trained model found. Training a quick model first...")
            trainer.train(episodes=200, max_steps_per_episode=30)
            trainer.agent.save_model('parking_dqn_trained.pth')
        
        # Set epsilon to 0 for demonstration (no exploration)
        trainer.agent.epsilon = 0.0
        
        # Create enhanced visualizer that can use the trained agent
        visualizer = EnhancedParkingLotVisualizer(env, trainer.agent)
        visualizer.run()
        
    except Exception as e:
        print(f"Error in demo: {e}")
        print("Falling back to basic simulation...")
        run_simulation()

class EnhancedParkingLotVisualizer(ParkingLotVisualizer):
    """Enhanced visualizer that can use a trained RL agent."""
    
    def __init__(self, environment, agent=None):
        super().__init__(environment)
        self.agent = agent
        self.auto_mode = False
        
        # Add auto mode button
        self.auto_button = pygame.Rect(self.grid_width + 250, 
                                      self.window_height - 80, 120, 30)
    
    def draw_buttons(self):
        """Draw control buttons including auto mode."""
        super().draw_buttons()
        
        # Auto mode button
        color = self.colors['button_hover'] if self.auto_mode else self.colors['button']
        pygame.draw.rect(self.screen, color, self.auto_button)
        pygame.draw.rect(self.screen, self.colors['grid_line'], self.auto_button, 2)
        
        auto_text = "Auto: ON" if self.auto_mode else "Auto: OFF"
        text_surface = self.small_font.render(auto_text, True, self.colors['text'])
        text_rect = text_surface.get_rect(center=self.auto_button.center)
        self.screen.blit(text_surface, text_rect)
    
    def handle_events(self):
        """Handle pygame events with auto mode support."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    mouse_pos = pygame.mouse.get_pos()
                    
                    # Check auto button click
                    if self.auto_button.collidepoint(mouse_pos):
                        self.auto_mode = not self.auto_mode
                        print(f"Auto mode: {'ON' if self.auto_mode else 'OFF'}")
                    
                    # Handle other button clicks
                    elif self.reset_button.collidepoint(mouse_pos):
                        self.env.reset()
                        print("Environment reset")
                    
                    elif self.step_button.collidepoint(mouse_pos):
                        self.take_action()
                    
                    elif not self.auto_mode:
                        # Manual grid clicks only when not in auto mode
                        action = self.get_clicked_cell(mouse_pos)
                        if action is not None:
                            state, reward, done, info = self.env.step(action)
                            print(f"Manual action: {action}, Reward: {reward:.2f}, Info: {info}")
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.env.reset()
                    print("Environment reset (R key)")
                elif event.key == pygame.K_SPACE:
                    self.take_action()
                elif event.key == pygame.K_a:
                    self.auto_mode = not self.auto_mode
                    print(f"Auto mode: {'ON' if self.auto_mode else 'OFF'}")
        
        # Auto mode actions
        if self.auto_mode and pygame.time.get_ticks() % 1000 < 50:  # Every ~1 second
            self.take_action()
        
        return True
    
    def take_action(self):
        """Take an action using the agent or randomly."""
        available_actions = self.env.get_available_actions()
        if not available_actions:
            return
        
        if self.agent:
            # Use trained agent
            state = self.env.get_state()
            action = self.agent.act(state, available_actions)
            action_type = "Agent"
        else:
            # Random action
            action = random.choice(available_actions)
            action_type = "Random"
        
        state, reward, done, info = self.env.step(action)
        print(f"{action_type} action: {action}, Reward: {reward:.2f}, Info: {info}")

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Parking Lot RL Simulation')
    parser.add_argument('--mode', choices=['sim', 'train', 'demo'], default='sim',
                       help='Mode to run: sim (simulation), train (training), demo (trained agent demo)')
    parser.add_argument('--simulate', action='store_true',
                       help='Show pygame simulation during training (only works with --mode train)')
    
    args = parser.parse_args()
    
    print("Parking Lot Reinforcement Learning Environment")
    print("=" * 50)
    
    if args.mode == 'sim':
        print("Running interactive simulation...")
        run_simulation()
    elif args.mode == 'train':
        print("Running RL training...")
        if args.simulate:
            print("With pygame visualization enabled")
        run_training(simulate=args.simulate)
    elif args.mode == 'demo':
        print("Running demo with trained agent...")
        run_demo_with_trained_agent()

if __name__ == "__main__":
    main()