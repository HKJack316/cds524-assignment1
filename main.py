"""
CDS524 Assignment 1 - GridWorld Treasure Hunt with Q-Learning
Author: [ZHAN Ze peng]
Student ID: [3161439]
Date: March 2, 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import time
import os
import pandas as pd

# ==================== Game Environment Class ====================
class GridWorld:
    """
    Grid World Game Environment
    State Space: 5x5 grid positions (0-4, 0-4)
    Action Space: 0:up, 1:down, 2:left, 3:right
    """
    
    def __init__(self, size=5):
        """
        Initialize the game environment
        Args:
            size: grid size (default 5x5)
        """
        self.size = size
        
        # Define special positions
        self.start_pos = (0, 0)           # Starting position
        self.treasure_pos = (4, 4)         # Treasure location
        self.trap_positions = [(2, 2), (1, 3), (3, 1)]   # Trap locations
        self.wall_positions = [(1, 1), (3, 3)]           # Wall locations
        
        # Action mapping
        self.actions = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1)    # right
        }
        
        # Initialize game state
        self.reset()
        
        print(f"Game environment initialized! Grid size: {size}x{size}")
        print(f"Treasure position: {self.treasure_pos}")
        print(f"Trap positions: {self.trap_positions}")
        print(f"Wall positions: {self.wall_positions}")
    
    def reset(self):
        """
        Reset the game to starting state
        Returns:
            initial agent position
        """
        self.agent_pos = self.start_pos
        self.steps = 0
        self.total_reward = 0
        self.done = False
        return self.agent_pos
    
    def get_reward(self, position):
        """
        Reward function design (positive + negative rewards)
        Args:
            position: current agent position
        Returns:
            reward value
        """
        if position == self.treasure_pos:
            return 10.0      # Positive reward: finding treasure
        elif position in self.trap_positions:
            return -10.0     # Negative reward: falling into trap
        elif position in self.wall_positions:
            return -1.0      # Negative reward: hitting wall
        else:
            return -0.1      # Small step penalty to encourage shortest path
    
    def is_valid_move(self, position):
        """
        Check if move is valid
        Args:
            position: target position
        Returns:
            boolean indicating if move is valid
        """
        x, y = position
        # Check boundaries
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return False
        # Check walls
        if position in self.wall_positions:
            return False
        return True
    
    def step(self, action):
        """
        Execute action and return next state, reward, done flag
        Args:
            action: 0=up, 1=down, 2=left, 3=right
        Returns:
            next_state, reward, done
        """
        self.steps += 1
        
        # Calculate new position
        dx, dy = self.actions[action]
        current_x, current_y = self.agent_pos
        new_pos = (current_x + dx, current_y + dy)
        
        # Check if move is valid
        if self.is_valid_move(new_pos):
            self.agent_pos = new_pos
        
        # Get reward
        reward = self.get_reward(self.agent_pos)
        self.total_reward += reward
        
        # Check if game is over
        if self.agent_pos == self.treasure_pos or self.agent_pos in self.trap_positions:
            self.done = True
        elif self.steps >= 50:  # Prevent infinite loops
            self.done = True
        
        return self.agent_pos, reward, self.done
    
    def render_text(self, episode=None):
        """
        Command line visualization
        Args:
            episode: current episode number
        """
        os.system('cls' if os.name == 'nt' else 'clear')
        print("=" * 40)
        if episode is not None:
            print(f"Episode: {episode}")
        print(f"Steps: {self.steps} | Total Reward: {self.total_reward:.1f}")
        print("=" * 40)
        
        # Draw grid
        for i in range(self.size):
            row = ""
            for j in range(self.size):
                pos = (i, j)
                if pos == self.agent_pos:
                    row += "🤖 "
                elif pos == self.treasure_pos:
                    row += "💰 "
                elif pos in self.trap_positions:
                    row += "💀 "
                elif pos in self.wall_positions:
                    row += "🧱 "
                else:
                    row += "· "
            print(row)
        print("=" * 40)


# ==================== Q-learning Agent Class ====================
class QLearningAgent:
    """
    Q-learning algorithm implementation
    Includes epsilon-greedy exploration strategy
    """
    
    def __init__(self, state_size, action_size, 
                 learning_rate=0.1,      # Learning rate α
                 discount_factor=0.95,    # Discount factor γ
                 epsilon=1.0,             # Initial exploration rate
                 epsilon_min=0.01,         # Minimum exploration rate
                 epsilon_decay=0.995):     # Exploration rate decay
        
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Initialize Q-table (using dictionary)
        self.q_table = defaultdict(lambda: np.zeros(action_size))
        
        # Training statistics
        self.stats = {
            'episodes': [],
            'total_rewards': [],
            'steps': [],
            'epsilon': [],
            'success_rate': []
        }
        
        print(f"Q-learning agent initialized!")
        print(f"Learning rate: {self.lr}")
        print(f"Discount factor: {self.gamma}")
        print(f"Initial epsilon: {self.epsilon}")
    
    def get_state_key(self, state):
        """Convert state to dictionary key"""
        if isinstance(state, tuple):
            x, y = state
            return x * 5 + y  # 5x5 grid -> 0-24
        return state
    
    def choose_action(self, state):
        """
        Epsilon-greedy action selection
        Args:
            state: current state
        Returns:
            selected action
        """
        state_key = self.get_state_key(state)
        
        # Exploration: random action
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        
        # Exploitation: choose action with max Q-value
        else:
            q_values = self.q_table[state_key]
            # If multiple actions have same max Q-value, choose randomly
            max_q = np.max(q_values)
            max_actions = np.where(q_values == max_q)[0]
            return np.random.choice(max_actions)
    
    def learn(self, state, action, reward, next_state, done):
        """
        Q-learning update formula:
        Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
        """
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        # Current Q-value
        current_q = self.q_table[state_key][action]
        
        # Calculate target Q-value
        if done:
            target = reward
        else:
            future_q = np.max(self.q_table[next_state_key])
            target = reward + self.gamma * future_q
        
        # Q-value update
        new_q = current_q + self.lr * (target - current_q)
        self.q_table[state_key][action] = new_q
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def train(self, env, episodes=300):
        """
        Train the agent
        Args:
            env: game environment
            episodes: number of training episodes
        Returns:
            training statistics
        """
        print(f"\nStarting training for {episodes} episodes...")
        print("-" * 50)
        
        success_count = 0
        
        for episode in range(1, episodes + 1):
            state = env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            # Single episode
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = env.step(action)
                self.learn(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                steps += 1
            
            # Statistics
            if env.agent_pos == env.treasure_pos:
                success_count += 1
            
            self.decay_epsilon()
            
            # Save stats
            self.stats['episodes'].append(episode)
            self.stats['total_rewards'].append(total_reward)
            self.stats['steps'].append(steps)
            self.stats['epsilon'].append(self.epsilon)
            self.stats['success_rate'].append(success_count / episode * 100)
            
            # Print progress every 50 episodes
            if episode % 50 == 0:
                avg_reward = np.mean(self.stats['total_rewards'][-50:])
                success_rate = success_count / episode * 100
                print(f"Episode {episode}/{episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Success Rate: {success_rate:.1f}% | "
                      f"Epsilon: {self.epsilon:.3f}")
        
        print(f"\nTraining completed! Final success rate: {success_count/episodes*100:.1f}%")
        return self.stats
    
    def test(self, env, episodes=5, render=True):
        """
        Test the trained agent
        Args:
            env: game environment
            episodes: number of test episodes
            render: whether to render visualization
        """
        print(f"\nTesting for {episodes} episodes...")
        print("-" * 50)
        
        # Turn off exploration
        original_epsilon = self.epsilon
        self.epsilon = 0
        
        for episode in range(1, episodes + 1):
            state = env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            print(f"\nTest Episode {episode}:")
            
            while not done:
                if render:
                    env.render_text(episode)
                    time.sleep(0.5)
                
                action = self.choose_action(state)
                next_state, reward, done = env.step(action)
                
                action_names = ['up', 'down', 'left', 'right']
                print(f"  Position {state} -> Action {action_names[action]} -> "
                      f"Reward {reward:.1f} -> New Position {next_state}")
                
                state = next_state
                total_reward += reward
                steps += 1
            
            if env.agent_pos == env.treasure_pos:
                print(f"  ✨ Successfully found treasure! Total reward: {total_reward:.1f}")
            else:
                print(f"  Game over! Total reward: {total_reward:.1f}")
        
        # Restore epsilon
        self.epsilon = original_epsilon
        
        return self.stats


# ==================== Visualization Functions ====================
def plot_learning_curves(stats):
    """
    Plot learning curves for report
    Args:
        stats: training statistics dictionary
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. Rewards per episode
    axes[0, 0].plot(stats['episodes'], stats['total_rewards'], 
                    alpha=0.6, linewidth=0.5, color='blue', label='Raw')
    # Moving average
    window = 20
    if len(stats['total_rewards']) >= window:
        moving_avg = np.convolve(stats['total_rewards'], 
                                np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window, len(stats['episodes'])+1), 
                       moving_avg, color='red', linewidth=2, label=f'{window}-episode MA')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Learning Curve: Total Reward per Episode')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0, 0].legend()
    
    # 2. Steps per episode
    axes[0, 1].plot(stats['episodes'], stats['steps'], 
                    alpha=0.6, linewidth=0.5, color='green', label='Raw')
    if len(stats['steps']) >= window:
        moving_avg_steps = np.convolve(stats['steps'], 
                                      np.ones(window)/window, mode='valid')
        axes[0, 1].plot(range(window, len(stats['episodes'])+1), 
                       moving_avg_steps, color='red', linewidth=2, label=f'{window}-episode MA')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].set_title('Learning Curve: Steps per Episode')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # 3. Epsilon decay
    axes[1, 0].plot(stats['episodes'], stats['epsilon'], 
                    color='purple', linewidth=2)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Epsilon (ε)')
    axes[1, 0].set_title('Exploration Rate Decay')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Success rate
    axes[1, 1].plot(stats['episodes'], stats['success_rate'], 
                    color='orange', linewidth=2)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Success Rate (%)')
    axes[1, 1].set_title('Training Success Rate')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print statistics summary
    print("\n📊 Training Statistics Summary:")
    print(f"Final success rate: {stats['success_rate'][-1]:.1f}%")
    print(f"Average reward (last 50 episodes): {np.mean(stats['total_rewards'][-50:]):.2f}")
    print(f"Average steps (last 50 episodes): {np.mean(stats['steps'][-50:]):.1f}")


def visualize_policy(agent, env):
    """
    Visualize learned policy for report
    Args:
        agent: trained Q-learning agent
        env: game environment
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Action arrow mapping
    arrows = {0: '↑', 1: '↓', 2: '←', 3: '→'}
    
    # Draw optimal action for each cell
    for i in range(env.size):
        for j in range(env.size):
            pos = (i, j)
            
            # Mark special positions
            if pos == env.treasure_pos:
                ax.text(j+0.5, env.size-1-i+0.5, '💰', 
                       ha='center', va='center', fontsize=25, color='gold', weight='bold')
            elif pos in env.trap_positions:
                ax.text(j+0.5, env.size-1-i+0.5, '💀', 
                       ha='center', va='center', fontsize=25, color='red', weight='bold')
            elif pos in env.wall_positions:
                ax.add_patch(plt.Rectangle((j, env.size-1-i), 1, 1,
                                          facecolor='gray', alpha=0.5))
                ax.text(j+0.5, env.size-1-i+0.5, '🧱', 
                       ha='center', va='center', fontsize=20, color='black')
            else:
                # Get optimal action
                state_key = agent.get_state_key(pos)
                q_values = agent.q_table[state_key]
                best_action = np.argmax(q_values)
                
                # Draw arrow (larger and more visible)
                ax.text(j+0.5, env.size-1-i+0.5, arrows[best_action], 
                       ha='center', va='center', fontsize=35, color='blue', weight='bold')
                
                # Show Q-value
                ax.text(j+0.5, env.size-1-i+0.15, f'{np.max(q_values):.1f}', 
                       ha='center', va='center', fontsize=9, color='darkgreen', weight='bold')
    
    # Draw grid lines
    for i in range(env.size + 1):
        ax.axhline(i, color='black', linewidth=2)
        ax.axvline(i, color='black', linewidth=2)
    
    ax.set_xlim(0, env.size)
    ax.set_ylim(0, env.size)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    ax.set_title('Learned Policy (Arrows=Optimal Actions, Numbers=Q-values)', fontsize=14, weight='bold')
    
    plt.tight_layout()
    plt.savefig('learned_policy.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Policy visualization saved as 'learned_policy.png'")


def generate_game_screenshot(env, episode="Test"):
    """
    Generate a game screenshot for the report
    Args:
        env: game environment
        episode: episode name/number
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    # Draw special positions
    for i in range(env.size):
        for j in range(env.size):
            pos = (i, j)
            if pos == env.treasure_pos:
                ax.text(j+0.5, env.size-1-i+0.5, '💰', ha='center', va='center', fontsize=25)
            elif pos in env.trap_positions:
                ax.text(j+0.5, env.size-1-i+0.5, '💀', ha='center', va='center', fontsize=25)
            elif pos in env.wall_positions:
                ax.add_patch(plt.Rectangle((j, env.size-1-i), 1, 1,
                                          facecolor='gray', alpha=0.5))
    
    # Draw grid lines
    for i in range(env.size + 1):
        ax.axhline(i, color='black', linewidth=1)
        ax.axvline(i, color='black', linewidth=1)
    
    # Draw agent
    x, y = env.agent_pos
    circle = plt.Circle((y+0.5, env.size-1-x+0.5), 0.3, color='blue', alpha=0.8)
    ax.add_patch(circle)
    ax.text(y+0.5, env.size-1-x+0.5, '🤖', ha='center', va='center', fontsize=15)
    
    ax.set_xlim(0, env.size)
    ax.set_ylim(0, env.size)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    ax.set_title(f'Game State - {episode}', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'game_screenshot_{episode}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Game screenshot saved as 'game_screenshot_{episode}.png'")


# ==================== Main Program ====================
if __name__ == "__main__":
    print("=" * 60)
    print("CDS524 Assignment 1 - GridWorld Treasure Hunt")
    print("=" * 60)
    
    # 1. Create environment and agent
    env = GridWorld(size=5)
    agent = QLearningAgent(
        state_size=25,
        action_size=4,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    )
    
    # 2. Train the agent
    stats = agent.train(env, episodes=300)
    
    # 3. Plot learning curves
    plot_learning_curves(stats)
    
    # 4. Visualize learned policy
    visualize_policy(agent, env)
    
    # 5. Generate game screenshots for report
    print("\n📸 Generating game screenshots...")
    
    # Screenshot 1: Start position
    env.reset()
    generate_game_screenshot(env, "Start")
    
    # Screenshot 2: Middle of game
    env.agent_pos = (2, 2)  # Set to middle position
    generate_game_screenshot(env, "Middle")
    
    # Screenshot 3: Treasure found
    env.agent_pos = (4, 4)
    generate_game_screenshot(env, "Treasure")
    
    # 6. Test the agent
    agent.test(env, episodes=3, render=True)
    
    # 7. Save Q-table for report
    q_data = []
    for i in range(5):
        for j in range(5):
            state_key = i * 5 + j
            pos = (i, j)
            if pos not in env.trap_positions and pos not in env.wall_positions:
                q_values = agent.q_table[state_key]
                q_data.append({
                    'Position': f'({i},{j})',
                    'Q_up': f'{q_values[0]:.2f}',
                    'Q_down': f'{q_values[1]:.2f}',
                    'Q_left': f'{q_values[2]:.2f}',
                    'Q_right': f'{q_values[3]:.2f}',
                    'Best Action': ['↑', '↓', '←', '→'][np.argmax(q_values)]
                })
    
    df = pd.DataFrame(q_data)
    df.to_csv('q_table_results.csv', index=False)
    print("\n✅ Q-table saved to q_table_results.csv")
    
    # 8. Print summary of generated files
    print("\n" + "=" * 60)
    print("📁 GENERATED FILES SUMMARY")
    print("=" * 60)
    print("1. learning_curves.png     - Training performance charts")
    print("2. learned_policy.png      - Visualized optimal policy")
    print("3. game_screenshot_Start.png  - Game at start position")
    print("4. game_screenshot_Middle.png - Game in middle")
    print("5. game_screenshot_Treasure.png - Game at treasure")
    print("6. q_table_results.csv     - Complete Q-table data")
    print("=" * 60)
    print("\n🎉 Project completed! All files generated successfully!")
    print("Use these files for your report and video presentation.")