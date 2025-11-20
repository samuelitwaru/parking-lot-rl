import pygame
import numpy as np
from typing import Tuple, Optional
from parking_environment import ParkingLotEnvironment, GridState
import sys

class ParkingLotVisualizer:
    """
    Pygame visualization for the parking lot environment.
    """
    
    def __init__(self, environment: ParkingLotEnvironment, 
                 window_width: int = 800, 
                 window_height: int = 600):
        
        self.env = environment
        self.window_width = window_width
        self.window_height = window_height
        
        # Calculate grid dimensions
        self.grid_width = self.window_width // 2
        self.grid_height = self.window_height - 100  # Leave space for UI
        self.cell_width = self.grid_width // self.env.width
        self.cell_height = self.grid_height // self.env.height
        
        # Colors
        self.colors = {
            'background': (240, 240, 240),
            'grid_line': (200, 200, 200),
            'empty': (255, 255, 255),
            'occupied': (220, 20, 20),
            'entry': (20, 220, 20),
            'text': (50, 50, 50),
            'panel_bg': (250, 250, 250),
            'button': (100, 150, 200),
            'button_hover': (120, 170, 220)
        }
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Parking Lot RL Environment")
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # UI elements
        self.info_panel_rect = pygame.Rect(self.grid_width + 10, 10, 
                                          self.window_width - self.grid_width - 20, 
                                          self.window_height - 20)
        
        # Control buttons
        self.reset_button = pygame.Rect(self.grid_width + 20, 
                                       self.window_height - 80, 100, 30)
        self.step_button = pygame.Rect(self.grid_width + 130, 
                                      self.window_height - 80, 100, 30)
        
        self.clock = pygame.time.Clock()
        self.running = True
        
    def draw_grid(self):
        """Draw the parking lot grid."""
        # Fill background
        grid_rect = pygame.Rect(0, 0, self.grid_width, self.grid_height)
        pygame.draw.rect(self.screen, self.colors['background'], grid_rect)
        
        # Draw cells
        for row in range(self.env.height):
            for col in range(self.env.width):
                x = col * self.cell_width
                y = row * self.cell_height
                cell_rect = pygame.Rect(x, y, self.cell_width, self.cell_height)
                
                # Determine cell color
                if (col, row) == self.env.entry_position:
                    color = self.colors['entry']
                elif self.env.grid[row, col] == GridState.OCCUPIED.value:
                    color = self.colors['occupied']
                else:
                    color = self.colors['empty']
                
                # Draw cell
                pygame.draw.rect(self.screen, color, cell_rect)
                pygame.draw.rect(self.screen, self.colors['grid_line'], cell_rect, 1)
                
                # Add labels for special cells
                if (col, row) == self.env.entry_position:
                    text = self.small_font.render("ENTRY", True, self.colors['text'])
                    text_rect = text.get_rect(center=cell_rect.center)
                    self.screen.blit(text, text_rect)
                elif self.env.grid[row, col] == GridState.OCCUPIED.value:
                    # Show car info
                    if (col, row) in self.env.cars:
                        car = self.env.cars[(col, row)]
                        car_text = f"C{car.id}"
                        time_text = f"T:{car.time_remaining}"
                        
                        car_surface = self.small_font.render(car_text, True, (255, 255, 255))
                        time_surface = self.small_font.render(time_text, True, (255, 255, 255))
                        
                        car_rect = car_surface.get_rect(center=(cell_rect.centerx, cell_rect.centery - 5))
                        time_rect = time_surface.get_rect(center=(cell_rect.centerx, cell_rect.centery + 8))
                        
                        self.screen.blit(car_surface, car_rect)
                        self.screen.blit(time_surface, time_rect)
    
    def draw_info_panel(self):
        """Draw the information panel."""
        pygame.draw.rect(self.screen, self.colors['panel_bg'], self.info_panel_rect)
        pygame.draw.rect(self.screen, self.colors['grid_line'], self.info_panel_rect, 2)
        
        stats = self.env.get_stats()
        
        # Title
        title_text = self.font.render("Parking Lot Statistics", True, self.colors['text'])
        self.screen.blit(title_text, (self.info_panel_rect.x + 10, self.info_panel_rect.y + 10))
        
        # Statistics
        y_offset = 50
        line_height = 25
        
        info_lines = [
            f"Total Cars Parked: {stats['total_cars_parked']}",
            f"Successful Parks: {stats['successful_parks']}",
            f"Failed Attempts: {stats['failed_attempts']}",
            f"Success Rate: {stats['success_rate']:.2%}",
            f"Occupancy Rate: {stats['occupancy_rate']:.2%}",
            f"Cars in Queue: {stats['cars_in_queue']}"
        ]
        
        for i, line in enumerate(info_lines):
            text = self.small_font.render(line, True, self.colors['text'])
            self.screen.blit(text, (self.info_panel_rect.x + 10, 
                                   self.info_panel_rect.y + y_offset + i * line_height))
        
        # Queue information
        queue_y = y_offset + len(info_lines) * line_height + 20
        queue_title = self.font.render("Car Queue:", True, self.colors['text'])
        self.screen.blit(queue_title, (self.info_panel_rect.x + 10, 
                                      self.info_panel_rect.y + queue_y))
        
        # Show first few cars in queue
        for i, car in enumerate(self.env.car_queue[:5]):
            car_info = f"Car {car.id}: {car.parking_time}min"
            text = self.small_font.render(car_info, True, self.colors['text'])
            self.screen.blit(text, (self.info_panel_rect.x + 20, 
                                   self.info_panel_rect.y + queue_y + 25 + i * 18))
        
        if len(self.env.car_queue) > 5:
            more_text = f"... and {len(self.env.car_queue) - 5} more"
            text = self.small_font.render(more_text, True, self.colors['text'])
            self.screen.blit(text, (self.info_panel_rect.x + 20, 
                                   self.info_panel_rect.y + queue_y + 25 + 5 * 18))
    
    def draw_buttons(self):
        """Draw control buttons."""
        # Reset button
        pygame.draw.rect(self.screen, self.colors['button'], self.reset_button)
        pygame.draw.rect(self.screen, self.colors['grid_line'], self.reset_button, 2)
        reset_text = self.small_font.render("Reset", True, self.colors['text'])
        reset_rect = reset_text.get_rect(center=self.reset_button.center)
        self.screen.blit(reset_text, reset_rect)
        
        # Step button
        pygame.draw.rect(self.screen, self.colors['button'], self.step_button)
        pygame.draw.rect(self.screen, self.colors['grid_line'], self.step_button, 2)
        step_text = self.small_font.render("Random Step", True, self.colors['text'])
        step_rect = step_text.get_rect(center=self.step_button.center)
        self.screen.blit(step_text, step_rect)
    
    def get_clicked_cell(self, mouse_pos: Tuple[int, int]) -> Optional[int]:
        """Convert mouse position to grid cell action."""
        x, y = mouse_pos
        
        if x < 0 or x >= self.grid_width or y < 0 or y >= self.grid_height:
            return None
        
        col = x // self.cell_width
        row = y // self.cell_height
        
        if col >= self.env.width or row >= self.env.height:
            return None
        
        return row * self.env.width + col
    
    def handle_events(self) -> bool:
        """Handle pygame events. Returns False if should quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    mouse_pos = pygame.mouse.get_pos()
                    
                    # Check button clicks
                    if self.reset_button.collidepoint(mouse_pos):
                        self.env.reset()
                        print("Environment reset")
                    
                    elif self.step_button.collidepoint(mouse_pos):
                        available_actions = self.env.get_available_actions()
                        if available_actions:
                            action = np.random.choice(available_actions)
                            state, reward, done, info = self.env.step(action)
                            print(f"Random action: {action}, Reward: {reward:.2f}, Info: {info}")
                    
                    else:
                        # Check grid clicks
                        action = self.get_clicked_cell(mouse_pos)
                        if action is not None:
                            state, reward, done, info = self.env.step(action)
                            print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.env.reset()
                    print("Environment reset (R key)")
                elif event.key == pygame.K_SPACE:
                    available_actions = self.env.get_available_actions()
                    if available_actions:
                        action = np.random.choice(available_actions)
                        state, reward, done, info = self.env.step(action)
                        print(f"Random action: {action}, Reward: {reward:.2f}, Info: {info}")
        
        return True
    
    def run(self):
        """Main visualization loop."""
        print("Parking Lot Environment Controls:")
        print("- Click on grid cells to select parking spots")
        print("- Click 'Reset' button or press 'R' to reset environment")
        print("- Click 'Random Step' button or press 'Space' for random action")
        print("- Close window or press ESC to quit")
        
        while self.running:
            self.running = self.handle_events()
            
            # Clear screen
            self.screen.fill(self.colors['background'])
            
            # Draw components
            self.draw_grid()
            self.draw_info_panel()
            self.draw_buttons()
            
            # Update display
            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS
        
        pygame.quit()

def main():
    """Main function to run the visualization."""
    # Create environment
    env = ParkingLotEnvironment(width=10, height=8, entry_position=(0, 4))
    
    # Create and run visualizer
    visualizer = ParkingLotVisualizer(env)
    visualizer.run()

if __name__ == "__main__":
    main()