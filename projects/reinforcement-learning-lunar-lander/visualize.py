import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import json
import logging
from datetime import datetime
import imageio
from typing import Dict, List, Optional, Tuple, Any
from matplotlib import animation

from environment import LunarLanderEnvironment
from agent import DQNAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/visualization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def visualize_episode(agent, env: LunarLanderEnvironment, save_path: str = None, render: bool = True):
    """
    Visualize a single episode with the given agent.
    
    Args:
        agent: The trained agent to use
        env: The environment to run in
        save_path: Path to save the visualization (if None, display it)
        render: Whether to render during execution
    
    Returns:
        Tuple of (total_reward, frames)
    """
    frames = []
    observation = env.reset()
    total_reward = 0
    done = False
    
    # Run episode
    while not done:
        if render:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        
        # Get action from the agent
        action = agent.predict(observation)
        
        # Execute action
        observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
    
    # Get the final frame
    if render:
        frame = env.render()
        if frame is not None:
            frames.append(frame)
    
    # Save visualization if requested
    if save_path and frames:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save as GIF
        if save_path.endswith('.gif'):
            imageio.mimsave(save_path, frames, fps=30)
        # Save as MP4
        elif save_path.endswith('.mp4'):
            env.save_frames_as_mp4(frames, save_path)
    
    return total_reward, frames

def visualize_agent_behavior(agent, env: LunarLanderEnvironment, num_episodes: int = 5,
                            save_dir: str = "results/visualizations"):
    """
    Visualize the behavior of an agent across multiple episodes.
    
    Args:
        agent: The trained agent to use
        env: The environment to run in
        num_episodes: Number of episodes to visualize
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Visualizing agent behavior across {num_episodes} episodes")
    
    rewards = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for episode in range(1, num_episodes + 1):
        # Record episode
        save_path = os.path.join(save_dir, f"episode_{episode}_{timestamp}.gif")
        reward, _ = visualize_episode(agent, env, save_path=save_path)
        rewards.append(reward)
        
        logger.info(f"Episode {episode}: Reward = {reward:.2f}")
    
    # Plot rewards
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, num_episodes + 1), rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Rewards Across Episodes")
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig(os.path.join(save_dir, f"rewards_{timestamp}.png"))
    
    logger.info(f"Visualizations saved to {save_dir}")
    return rewards

def visualize_q_values(agent, env: LunarLanderEnvironment, num_states: int = 5,
                     save_dir: str = "results/q_values"):
    """
    Visualize the Q-values for different states.
    
    Args:
        agent: The trained agent to use
        env: The environment to run in
        num_states: Number of states to visualize
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Visualizing Q-values for {num_states} states")
    
    # Get some sample states
    states = []
    observation = env.reset()
    states.append(observation)
    
    # Collect diverse states
    done = False
    while len(states) < num_states and not done:
        action = agent.predict(observation)
        observation, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Add state if it's different enough
        if not done and all(np.linalg.norm(observation - s) > 1.0 for s in states):
            states.append(observation)
    
    # Fill up with random states if needed
    while len(states) < num_states:
        observation = env.reset()
        if all(np.linalg.norm(observation - s) > 1.0 for s in states):
            states.append(observation)
    
    # Compute Q-values for each state
    q_values = []
    for state in states:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.policy_net.fc1.weight.device)
            q_vals = agent.policy_net(state_tensor).cpu().numpy()[0]
            q_values.append(q_vals)
    
    # Create bar chart of Q-values for each state
    fig, axes = plt.subplots(num_states, 1, figsize=(10, 4 * num_states))
    if num_states == 1:
        axes = [axes]
    
    action_names = ["Do nothing", "Fire left engine", "Fire main engine", "Fire right engine"]
    
    for i, (state, q_vals) in enumerate(zip(states, q_values)):
        ax = axes[i]
        bars = ax.bar(action_names, q_vals)
        
        # Color the highest value bar
        best_action = np.argmax(q_vals)
        bars[best_action].set_color('green')
        
        # Add value labels on the bars
        for bar, val in zip(bars, q_vals):
            ax.text(bar.get_x() + bar.get_width()/2, 
                   val + 0.1 * (max(q_vals) - min(q_vals)),
                   f'{val:.2f}', 
                   ha='center')
        
        ax.set_title(f"State {i+1}: {env.get_state_description(state)}")
        ax.set_ylabel("Q-Value")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "q_values.png"))
    
    logger.info(f"Q-value visualization saved to {save_dir}")

def visualize_training_progress(training_data_path: str, save_dir: str = "results/training_progress"):
    """
    Visualize the training progress from a saved training data file.
    
    Args:
        training_data_path: Path to the training data JSON file
        save_dir: Directory to save the visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Visualizing training progress from {training_data_path}")
    
    # Load training data
    with open(training_data_path, 'r') as f:
        training_data = json.load(f)
    
    # Extract data
    episode_rewards = training_data.get('episode_rewards', [])
    losses = training_data.get('losses', [])
    eval_rewards = training_data.get('eval_rewards', [])
    
    # Create rewards plot
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards, alpha=0.3, label='Episode Rewards')
    
    # Add moving average if we have enough episodes
    if len(episode_rewards) >= 100:
        window_size = 100
        weights = np.ones(window_size) / window_size
        moving_avg = np.convolve(episode_rewards, weights, mode='valid')
        plt.plot(range(window_size-1, len(episode_rewards)), moving_avg, 
                label=f'{window_size}-Episode Moving Avg', linewidth=2)
    
    # Add solving threshold
    plt.axhline(y=200, color='r', linestyle='--', label='Solving Threshold')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "training_rewards.png"))
    
    # Create loss plot if available
    if losses:
        plt.figure(figsize=(12, 6))
        plt.plot(losses, label='Training Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, "training_loss.png"))
    
    # Create evaluation rewards plot if available
    if eval_rewards:
        if isinstance(eval_rewards[0], list):  # Handle list of lists format
            episodes = [item[0] for item in eval_rewards]
            rewards = [item[1] for item in eval_rewards]
        else:  # Handle dict format
            episodes = range(len(eval_rewards))
            rewards = eval_rewards
        
        plt.figure(figsize=(12, 6))
        plt.plot(episodes, rewards, 'o-', label='Evaluation Rewards')
        plt.axhline(y=200, color='r', linestyle='--', label='Solving Threshold')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.title('Evaluation Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, "evaluation_rewards.png"))
    
    logger.info(f"Training visualizations saved to {save_dir}")

def visualize_learning_curves_comparison(data_paths: List[str], labels: List[str], 
                                       save_path: str = "results/comparison/learning_curves.png"):
    """
    Compare learning curves from different training runs.
    
    Args:
        data_paths: List of paths to training data files
        labels: List of labels for each curve
        save_path: Path to save the comparison plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    for data_path, label in zip(data_paths, labels):
        # Load training data
        with open(data_path, 'r') as f:
            training_data = json.load(f)
        
        # Extract rewards
        episode_rewards = training_data.get('episode_rewards', [])
        
        # Plot raw rewards with low alpha
        plt.plot(episode_rewards, alpha=0.1)
        
        # Plot moving average
        if len(episode_rewards) >= 100:
            window_size = 100
            weights = np.ones(window_size) / window_size
            moving_avg = np.convolve(episode_rewards, weights, mode='valid')
            plt.plot(range(window_size-1, len(episode_rewards)), moving_avg, 
                    label=f'{label}', linewidth=2)
        else:
            plt.plot(episode_rewards, label=label, linewidth=2)
    
    # Add solving threshold
    plt.axhline(y=200, color='r', linestyle='--', label='Solving Threshold')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Learning Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_path)
    logger.info(f"Learning curves comparison saved to {save_path}")

def main():
    """Main function for the visualization module."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize RL agent performance")
    
    parser.add_argument("--model", type=str, required=True,
                      help="Path to the trained model file")
    parser.add_argument("--mode", type=str, choices=["episode", "behavior", "q_values", "training", "comparison"],
                      default="episode", help="Visualization mode")
    parser.add_argument("--episodes", type=int, default=1,
                      help="Number of episodes to visualize")
    parser.add_argument("--training_data", type=str,
                      help="Path to training data file (for training mode)")
    parser.add_argument("--comparison_models", type=str, nargs='+',
                      help="List of model paths to compare (for comparison mode)")
    parser.add_argument("--save_dir", type=str, default="results/visualizations",
                      help="Directory to save visualizations")
    parser.add_argument("--render", action="store_true",
                      help="Render environment during visualization")
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Handle different visualization modes
    if args.mode == "training":
        if not args.training_data:
            logger.error("Training data file must be specified for training mode")
            return
        
        visualize_training_progress(args.training_data, args.save_dir)
        return
    
    if args.mode == "comparison":
        if not args.comparison_models:
            logger.error("Comparison models must be specified for comparison mode")
            return
        
        # Assume training data files have the same basename but with _data.json
        data_paths = [path.replace(".pth", "_data.json") for path in args.comparison_models]
        labels = [os.path.basename(path).replace(".pth", "") for path in args.comparison_models]
        
        save_path = os.path.join(args.save_dir, "model_comparison.png")
        visualize_learning_curves_comparison(data_paths, labels, save_path)
        return
    
    # For other modes, we need a trained agent
    from evaluate import load_agent
    
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        return
    
    # Load the agent
    agent = load_agent(args.model)
    
    # Create environment
    render_mode = "human" if args.render else "rgb_array"
    env = LunarLanderEnvironment(render_mode=render_mode)
    
    # Run visualization based on mode
    if args.mode == "episode":
        # Visualize a single episode
        save_path = os.path.join(args.save_dir, "episode_visualization.gif")
        reward, _ = visualize_episode(agent, env, save_path)
        logger.info(f"Episode visualization completed. Reward: {reward:.2f}")
    
    elif args.mode == "behavior":
        # Visualize agent behavior across multiple episodes
        rewards = visualize_agent_behavior(agent, env, args.episodes, args.save_dir)
        logger.info(f"Agent behavior visualization completed. Average reward: {np.mean(rewards):.2f}")
    
    elif args.mode == "q_values":
        # Visualize Q-values
        visualize_q_values(agent, env, min(5, args.episodes), args.save_dir)
        logger.info("Q-value visualization completed")
    
    # Clean up
    env.close()

if __name__ == "__main__":
    main() 