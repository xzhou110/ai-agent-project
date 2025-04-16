#!/usr/bin/env python3
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Patch
import pandas as pd
import seaborn as sns
import imageio
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from matplotlib.animation import FuncAnimation, PillowWriter

from environment import LunarLanderEnv
from agent import DQNAgent, DoubleDQNAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Visualization")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize trained agents on LunarLander environment')
    
    parser.add_argument('--agent_type', type=str, default='dqn', choices=['dqn', 'double_dqn'],
                        help='Type of agent to visualize (dqn or double_dqn)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model file (.pth)')
    parser.add_argument('--continuous', action='store_true',
                        help='Use continuous action space version of LunarLander')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Visualization parameters
    parser.add_argument('--num_episodes', type=int, default=3,
                        help='Number of episodes to visualize')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='Maximum steps per episode')
    parser.add_argument('--delay', type=float, default=0.05,
                        help='Delay between frames when displaying (seconds)')
    
    # Network architecture (must match trained model)
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='Size of hidden layers in the neural network')
    parser.add_argument('--num_hidden_layers', type=int, default=2,
                        help='Number of hidden layers in the neural network')
    
    # Output options
    parser.add_argument('--save_video', action='store_true',
                        help='Save the visualization as a video file')
    parser.add_argument('--output_dir', type=str, default='results/videos',
                        help='Directory to save visualization results')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second for saved videos')
    parser.add_argument('--show_q_values', action='store_true',
                        help='Show Q-values for each action during visualization')
    parser.add_argument('--plot_state_values', action='store_true',
                        help='Plot state values after episode')
    
    return parser.parse_args()

def create_agent(args, env):
    """Create an agent based on the arguments and load the trained model."""
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()
    
    hidden_dims = [args.hidden_size] * args.num_hidden_layers
    
    # Create agent with the same architecture as during training
    if args.agent_type == 'dqn':
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            learning_rate=0.001,  # Not used during visualization
            gamma=0.99,           # Not used during visualization
            epsilon_start=0.0,    # No exploration during visualization
            epsilon_end=0.0,
            epsilon_decay=1.0,
            memory_size=1,        # Not used during visualization
            batch_size=1,         # Not used during visualization
            target_update_frequency=1  # Not used during visualization
        )
        logger.info("Created DQN Agent for visualization")
    else:  # double_dqn
        agent = DoubleDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            learning_rate=0.001,  # Not used during visualization
            gamma=0.99,           # Not used during visualization
            epsilon_start=0.0,    # No exploration during visualization
            epsilon_end=0.0,
            epsilon_decay=1.0,
            memory_size=1,        # Not used during visualization
            batch_size=1,         # Not used during visualization
            target_update_frequency=1  # Not used during visualization
        )
        logger.info("Created Double DQN Agent for visualization")
    
    # Load the trained model
    agent.load(args.model_path)
    logger.info(f"Loaded model from {args.model_path}")
    
    return agent

def record_episode(args, env, agent) -> Tuple[float, List[np.ndarray], List[np.ndarray], List[float]]:
    """
    Record a single episode from the agent.
    
    Args:
        args: Command line arguments
        env: Environment
        agent: Agent
        
    Returns:
        tuple: (episode_reward, frames, states, q_values)
    """
    # Reset environment and initialize variables
    state = env.reset()
    
    # Collect frames, states, and Q-values
    frames = []
    states = []
    q_values = []
    total_reward = 0
    
    # Run the episode
    for step in range(args.max_steps):
        # Render the current frame
        frame = env.render()
        frames.append(frame)
        
        # Get the action and Q-values
        action, q_value = agent.select_action(state, return_q_values=True, evaluate=True)
        
        # Take the action
        next_state, reward, done, info = env.step(action)
        
        # Store the state and Q-values
        states.append(state)
        q_values.append(q_value)
        
        # Update the total reward
        total_reward += reward
        
        # Move to the next state
        state = next_state
        
        # Check if the episode is done
        if done:
            break
    
    # Get the final frame
    frames.append(env.render())
    
    return total_reward, frames, states, q_values

def save_episode_video(frames: List[np.ndarray], q_values: List[np.ndarray], 
                       output_path: str, fps: int, show_q_values: bool = True):
    """
    Save the episode frames as a video.
    
    Args:
        frames: List of rendered frames
        q_values: List of Q-values for each step
        output_path: Path to save the video
        fps: Frames per second
        show_q_values: Whether to show Q-values on the video
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # If we're showing Q-values, add them to the frames
    if show_q_values and q_values:
        annotated_frames = []
        action_names = ["No-op", "Left", "Main", "Right"]
        
        for i, (frame, q_value) in enumerate(zip(frames[:-1], q_values)):
            # Create a figure with two subplots - one for the frame, one for the Q-values
            fig, (ax_frame, ax_q) = plt.subplots(1, 2, figsize=(12, 5), 
                                                 gridspec_kw={'width_ratios': [2, 1]})
            
            # Display the frame
            ax_frame.imshow(frame)
            ax_frame.set_title(f"Step {i}")
            ax_frame.axis('off')
            
            # Display the Q-values as a bar chart
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            bar_heights = q_value
            bars = ax_q.bar(action_names, bar_heights, color=colors)
            
            # Add a star to the chosen action
            chosen_action = np.argmax(q_value)
            ax_q.text(action_names[chosen_action], bar_heights[chosen_action], '★', 
                     horizontalalignment='center', verticalalignment='bottom',
                     fontsize=20, color='gold')
            
            ax_q.set_title("Q-Values")
            ax_q.set_ylim([min(q_value) - 0.1, max(q_value) + 0.1])
            
            # Adjust layout and convert to image
            plt.tight_layout()
            fig.canvas.draw()
            annotated_frame = np.array(fig.canvas.renderer.buffer_rgba())
            annotated_frames.append(annotated_frame)
            plt.close(fig)
        
        # Add the last frame
        fig, (ax_frame, ax_q) = plt.subplots(1, 2, figsize=(12, 5), 
                                             gridspec_kw={'width_ratios': [2, 1]})
        ax_frame.imshow(frames[-1])
        ax_frame.set_title(f"Final State")
        ax_frame.axis('off')
        ax_q.set_title("Episode Complete")
        ax_q.axis('off')
        plt.tight_layout()
        fig.canvas.draw()
        annotated_frame = np.array(fig.canvas.renderer.buffer_rgba())
        annotated_frames.append(annotated_frame)
        plt.close(fig)
        
        # Save the annotated frames
        frames_to_save = annotated_frames
    else:
        frames_to_save = frames
    
    # Save the video
    try:
        # Convert frames to uint8 if they aren't already
        uint8_frames = [frame.astype(np.uint8) if frame.dtype != np.uint8 else frame 
                        for frame in frames_to_save]
        
        # Save video using imageio
        imageio.mimsave(output_path, uint8_frames, fps=fps)
        logger.info(f"Saved video to {output_path}")
    except Exception as e:
        logger.error(f"Error saving video: {e}")

def plot_state_values(states: List[np.ndarray], output_path: str):
    """
    Plot the state values over time.
    
    Args:
        states: List of states
        output_path: Path to save the plot
    """
    # Convert to numpy array for easier manipulation
    states_array = np.array(states)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Define the state components
    state_names = ["X Position", "Y Position", "X Velocity", "Y Velocity", 
                   "Angle", "Angular Velocity", "Left Leg Contact", "Right Leg Contact"]
    
    # Create a plot for each state component
    for i, name in enumerate(state_names):
        plt.subplot(4, 2, i+1)
        plt.plot(states_array[:, i])
        plt.title(name)
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved state values plot to {output_path}")

def display_episode(frames: List[np.ndarray], delay: float):
    """
    Display the episode frames in real-time.
    
    Args:
        frames: List of frames
        delay: Delay between frames in seconds
    """
    plt.figure(figsize=(8, 6))
    
    for i, frame in enumerate(frames):
        plt.clf()
        plt.imshow(frame)
        plt.title(f"Step {i}")
        plt.axis('off')
        plt.pause(delay)
    
    plt.close()

def compare_agents(args1, args2, num_episodes=3):
    """
    Compare two different agents side by side.
    
    Args:
        args1: Arguments for the first agent
        args2: Arguments for the second agent
        num_episodes: Number of episodes to compare
    """
    # Setup environments and agents
    env1 = LunarLanderEnv(continuous=args1.continuous, render_mode="rgb_array", seed=args1.seed)
    env2 = LunarLanderEnv(continuous=args2.continuous, render_mode="rgb_array", seed=args2.seed)
    
    agent1 = create_agent(args1, env1)
    agent2 = create_agent(args2, env2)
    
    output_dir = os.path.join(args1.output_dir, "comparison")
    os.makedirs(output_dir, exist_ok=True)
    
    for episode in range(1, num_episodes + 1):
        # Record episodes
        reward1, frames1, states1, q_values1 = record_episode(args1, env1, agent1)
        reward2, frames2, states2, q_values2 = record_episode(args2, env2, agent2)
        
        # Create a side-by-side comparison video
        compare_frames = []
        max_frames = max(len(frames1), len(frames2))
        
        for i in range(max_frames):
            # Get frames from both agents (use the last frame if an episode ended early)
            frame1 = frames1[min(i, len(frames1) - 1)]
            frame2 = frames2[min(i, len(frames2) - 1)]
            
            # Create a figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Display the frames
            ax1.imshow(frame1)
            ax1.set_title(f"{args1.agent_type.upper()} - Reward: {reward1:.1f}")
            ax1.axis('off')
            
            ax2.imshow(frame2)
            ax2.set_title(f"{args2.agent_type.upper()} - Reward: {reward2:.1f}")
            ax2.axis('off')
            
            # Adjust layout and convert to image
            plt.tight_layout()
            fig.canvas.draw()
            compare_frame = np.array(fig.canvas.renderer.buffer_rgba())
            compare_frames.append(compare_frame)
            plt.close(fig)
        
        # Save the comparison video
        output_path = os.path.join(output_dir, f"comparison_episode_{episode}.mp4")
        uint8_frames = [frame.astype(np.uint8) if frame.dtype != np.uint8 else frame 
                         for frame in compare_frames]
        imageio.mimsave(output_path, uint8_frames, fps=args1.fps)
        logger.info(f"Saved comparison video to {output_path}")
    
    env1.close()
    env2.close()

def visualize_q_value_distribution(agent, env, args, num_samples=1000, num_episodes=5):
    """
    Visualize the distribution of Q-values for different actions.
    
    Args:
        agent: The agent to evaluate
        env: The environment
        args: Command line arguments
        num_samples: Number of random states to sample
        num_episodes: Number of episodes to run to collect states
    """
    # Collect states from random episodes
    states = []
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done and len(states) < num_samples:
            states.append(state)
            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, _ = env.step(action)
            state = next_state
    
    # Limit the number of states
    states = states[:num_samples]
    
    # Get Q-values for each state
    q_values = []
    for state in states:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_value = agent.q_network(state_tensor).cpu().numpy()[0]
            q_values.append(q_value)
    
    q_values = np.array(q_values)
    
    # Create a plot for the Q-value distributions
    plt.figure(figsize=(10, 6))
    
    # Action names for the discrete action space
    action_names = ["No-op", "Left Engine", "Main Engine", "Right Engine"]
    
    # Plot histogram for each action
    for i in range(env.get_action_dim()):
        sns.kdeplot(q_values[:, i], label=action_names[i], fill=True, alpha=0.3)
    
    plt.title("Q-Value Distribution by Action")
    plt.xlabel("Q-Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    output_dir = os.path.join(args.output_dir, "analysis")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "q_value_distribution.png")
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Saved Q-value distribution plot to {output_path}")

def plot_training_history(
    rewards_file,
    output_dir=None,
    window_size=100,
    plot_title="Training Progress",
    figsize=(12, 8)
):
    """
    Plot training history from saved rewards file.
    
    Args:
        rewards_file (str): Path to the saved rewards numpy file.
        output_dir (str): Directory to save plots (if None, plots are displayed).
        window_size (int): Window size for moving average.
        plot_title (str): Title for the plot.
        figsize (tuple): Figure size.
        
    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    # Load data
    rewards = np.load(rewards_file)
    episodes = np.arange(len(rewards))
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot raw rewards
    ax.plot(episodes, rewards, alpha=0.3, label="Episode Reward")
    
    # Plot moving average
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        ax.plot(np.arange(window_size-1, len(rewards)), moving_avg, 'r-', 
                linewidth=2, label=f"{window_size}-Episode Moving Average")
    
    # Add success threshold line
    ax.axhline(y=200, color='g', linestyle='--', label="Success Threshold (200)")
    
    # Add labels and title
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title(plot_title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save or display
    if output_dir:
        output_path = Path(output_dir) / "training_history.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Training history plot saved to {output_path}")
    else:
        plt.tight_layout()
        plt.show()
    
    return fig

def plot_reward_distribution(
    rewards_file,
    output_dir=None,
    plot_title="Reward Distribution",
    figsize=(10, 6)
):
    """
    Plot distribution of rewards from saved file.
    
    Args:
        rewards_file (str): Path to the saved rewards numpy file.
        output_dir (str): Directory to save plots (if None, plots are displayed).
        plot_title (str): Title for the plot.
        figsize (tuple): Figure size.
        
    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    # Load data
    rewards = np.load(rewards_file)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create distribution plot
    sns.histplot(rewards, kde=True, ax=ax)
    
    # Add statistics lines
    mean_reward = np.mean(rewards)
    median_reward = np.median(rewards)
    success_rate = (rewards >= 200).mean() * 100
    
    ax.axvline(mean_reward, color='r', linestyle='--', 
               label=f"Mean: {mean_reward:.2f}")
    ax.axvline(median_reward, color='g', linestyle='--', 
               label=f"Median: {median_reward:.2f}")
    
    # Add labels and title
    ax.set_xlabel("Reward")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{plot_title} (Success Rate: {success_rate:.2f}%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save or display
    if output_dir:
        output_path = Path(output_dir) / "reward_distribution.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Reward distribution plot saved to {output_path}")
    else:
        plt.tight_layout()
        plt.show()
    
    return fig

def compare_agents(
    rewards_files,
    agent_names,
    output_dir=None,
    window_size=100,
    plot_title="Agent Comparison",
    figsize=(12, 8)
):
    """
    Compare training progress of multiple agents.
    
    Args:
        rewards_files (list): List of paths to saved rewards numpy files.
        agent_names (list): List of agent names for legend.
        output_dir (str): Directory to save plots (if None, plots are displayed).
        window_size (int): Window size for moving average.
        plot_title (str): Title for the plot.
        figsize (tuple): Figure size.
        
    Returns:
        dict: Dictionary of generated figures.
    """
    if len(rewards_files) != len(agent_names):
        raise ValueError("Number of reward files must match number of agent names")
    
    # Load data
    all_rewards = []
    for file in rewards_files:
        rewards = np.load(file)
        all_rewards.append(rewards)
    
    # Create figures dictionary
    figures = {}
    
    # Plot moving averages for comparison
    fig_avg, ax_avg = plt.subplots(figsize=figsize)
    
    for i, rewards in enumerate(all_rewards):
        # Calculate moving average
        if len(rewards) >= window_size:
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            episodes = np.arange(window_size-1, len(rewards))
            ax_avg.plot(episodes, moving_avg, linewidth=2, label=agent_names[i])
    
    # Add success threshold line
    ax_avg.axhline(y=200, color='g', linestyle='--', label="Success Threshold (200)")
    
    # Add labels and title
    ax_avg.set_xlabel("Episode")
    ax_avg.set_ylabel(f"Average Reward (over {window_size} episodes)")
    ax_avg.set_title(plot_title)
    ax_avg.legend()
    ax_avg.grid(True, alpha=0.3)
    
    figures["average_comparison"] = fig_avg
    
    # Plot success rates over time
    fig_success, ax_success = plt.subplots(figsize=figsize)
    
    for i, rewards in enumerate(all_rewards):
        # Calculate success rate over time (using a sliding window)
        if len(rewards) >= window_size:
            success_rates = []
            for j in range(window_size, len(rewards) + 1):
                window_rewards = rewards[j - window_size:j]
                success_rate = (window_rewards >= 200).mean() * 100
                success_rates.append(success_rate)
            
            episodes = np.arange(window_size, len(rewards) + 1)
            ax_success.plot(episodes, success_rates, linewidth=2, label=agent_names[i])
    
    # Add labels and title
    ax_success.set_xlabel("Episode")
    ax_success.set_ylabel("Success Rate (%)")
    ax_success.set_title(f"Success Rate Over Time (Window: {window_size} episodes)")
    ax_success.legend()
    ax_success.grid(True, alpha=0.3)
    
    figures["success_comparison"] = fig_success
    
    # Plot final reward distributions
    fig_dist, ax_dist = plt.subplots(figsize=figsize)
    
    for i, rewards in enumerate(all_rewards):
        # Get last 100 episodes (or all if less than 100)
        final_rewards = rewards[-min(100, len(rewards)):]
        sns.kdeplot(final_rewards, label=agent_names[i], ax=ax_dist)
    
    # Add success threshold line
    ax_dist.axvline(x=200, color='g', linestyle='--', label="Success Threshold (200)")
    
    # Add labels and title
    ax_dist.set_xlabel("Reward")
    ax_dist.set_ylabel("Density")
    ax_dist.set_title("Final Reward Distributions")
    ax_dist.legend()
    ax_dist.grid(True, alpha=0.3)
    
    figures["distribution_comparison"] = fig_dist
    
    # Create success rate bar chart
    fig_bar, ax_bar = plt.subplots(figsize=figsize)
    
    success_rates = []
    for rewards in all_rewards:
        # Calculate overall success rate
        final_rewards = rewards[-min(100, len(rewards)):]
        success_rate = (final_rewards >= 200).mean() * 100
        success_rates.append(success_rate)
    
    ax_bar.bar(agent_names, success_rates, alpha=0.7)
    
    # Add labels and values
    for i, rate in enumerate(success_rates):
        ax_bar.text(i, rate + 1, f"{rate:.1f}%", ha='center')
    
    # Add labels and title
    ax_bar.set_xlabel("Agent")
    ax_bar.set_ylabel("Success Rate (%)")
    ax_bar.set_title("Success Rate Comparison (Last 100 Episodes)")
    ax_bar.grid(True, alpha=0.3, axis='y')
    
    figures["success_bar"] = fig_bar
    
    # Save or display figures
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for name, fig in figures.items():
            output_path = output_dir / f"{name}.png"
            fig.savefig(output_path)
            plt.close(fig)
        
        logger.info(f"Comparison plots saved to {output_dir}")
    else:
        plt.tight_layout()
        plt.show()
    
    return figures

def visualize_q_values(
    agent,
    environment,
    num_episodes=1,
    max_steps=200,
    output_dir=None,
    filename="q_values_visualization.gif",
    fps=10
):
    """
    Visualize Q-values of an agent during an episode.
    
    Args:
        agent: The trained agent (DQNAgent or DoubleDQNAgent).
        environment: The environment to run the agent in.
        num_episodes (int): Number of episodes to visualize.
        max_steps (int): Maximum steps per episode.
        output_dir (str): Directory to save the animation.
        filename (str): Filename for the output GIF.
        fps (int): Frames per second for the animation.
        
    Returns:
        str: Path to the saved animation file.
    """
    if not hasattr(agent, 'q_network'):
        raise ValueError("Agent must have a q_network attribute")
    
    # Set agent to evaluation mode
    agent.eval_mode()
    
    # Create output directory if needed
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename
    else:
        output_path = Path(filename)
    
    # Create figure and axes for visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Function to update the visualization
    frames = []
    
    for episode in range(num_episodes):
        # Reset environment
        state = environment.reset()
        done = False
        truncated = False
        step = 0
        
        while not (done or truncated) and step < max_steps:
            # Select action
            action = agent.select_action(state, evaluate=True)
            
            # Get Q-values for all actions
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                q_values = agent.q_network(state_tensor).squeeze().cpu().numpy()
            
            # Take action
            next_state, reward, done, truncated, _ = environment.step(action)
            
            # Create visualization frame
            axes[0].clear()
            axes[1].clear()
            
            # Render environment
            frame = environment.render()
            if frame is not None:
                axes[0].imshow(frame)
            axes[0].set_title(f"Episode {episode+1}, Step {step+1}, Reward: {reward:.2f}")
            axes[0].axis('off')
            
            # Visualize Q-values
            action_names = ["Left", "Nothing", "Right", "Main Engine"] if environment.action_space.n == 4 else [f"Action {i}" for i in range(environment.action_space.n)]
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            
            bars = axes[1].bar(action_names, q_values, color=[colors[i % len(colors)] for i in range(len(q_values))])
            axes[1].set_title("Q-Values for Each Action")
            axes[1].set_ylabel("Q-Value")
            axes[1].grid(True, alpha=0.3)
            
            # Highlight selected action
            bars[action].set_color('orange')
            
            # Add value labels
            for i, v in enumerate(q_values):
                axes[1].text(i, v + 0.01, f"{v:.2f}", ha='center')
            
            # Capture frame
            fig.tight_layout()
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(frame)
            
            # Update state
            state = next_state
            step += 1
    
    # Save animation
    ani = FuncAnimation(fig, lambda i: plt.imshow(frames[i]), frames=len(frames), blit=False)
    writer = PillowWriter(fps=fps)
    ani.save(output_path, writer=writer)
    
    plt.close(fig)
    logger.info(f"Q-values visualization saved to {output_path}")
    
    return str(output_path)

def visualize_agent_performance(
    agent,
    environment,
    num_episodes=3,
    max_steps=1000,
    output_dir=None,
    filename="agent_performance.gif",
    fps=30
):
    """
    Create an animation of the agent's performance in the environment.
    
    Args:
        agent: The trained agent.
        environment: The environment to run the agent in.
        num_episodes (int): Number of episodes to visualize.
        max_steps (int): Maximum steps per episode.
        output_dir (str): Directory to save the animation.
        filename (str): Filename for the output GIF.
        fps (int): Frames per second for the animation.
        
    Returns:
        str: Path to the saved animation file.
    """
    # Set agent to evaluation mode
    agent.eval_mode()
    
    # Create output directory if needed
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename
    else:
        output_path = Path(filename)
    
    # Create figure for visualization
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create frames
    frames = []
    episode_rewards = []
    
    for episode in range(num_episodes):
        # Reset environment
        state = environment.reset()
        done = False
        truncated = False
        step = 0
        episode_reward = 0
        
        while not (done or truncated) and step < max_steps:
            # Select action
            action = agent.select_action(state, evaluate=True)
            
            # Take action
            next_state, reward, done, truncated, _ = environment.step(action)
            
            # Render environment
            frame = environment.render()
            if frame is not None:
                # Clear axis
                ax.clear()
                
                # Display frame and add info
                ax.imshow(frame)
                ax.set_title(f"Episode {episode+1}/{num_episodes}, Step {step+1}, Reward: {episode_reward:.2f}")
                
                # Add action information
                action_text = ""
                if environment.action_space.n == 4:  # Lunar Lander discrete actions
                    action_names = ["Left Engine", "Do Nothing", "Right Engine", "Main Engine"]
                    action_text = f"Action: {action_names[action]}"
                
                ax.text(0.5, 0.02, action_text, transform=ax.transAxes, ha='center')
                
                # Remove axes
                ax.axis('off')
                
                # Capture frame
                fig.tight_layout()
                fig.canvas.draw()
                frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                frames.append(frame)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            step += 1
        
        episode_rewards.append(episode_reward)
        
        # Add final frame with episode summary
        ax.clear()
        if frame is not None:
            ax.imshow(frame)
        ax.set_title(f"Episode {episode+1} Complete")
        ax.text(0.5, 0.5, f"Episode Reward: {episode_reward:.2f}\nSteps: {step}", 
                transform=ax.transAxes, ha='center', va='center', fontsize=12)
        ax.axis('off')
        
        fig.tight_layout()
        fig.canvas.draw()
        summary_frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        summary_frame = summary_frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # Add summary frame multiple times to create a pause
        for _ in range(fps * 2):  # 2-second pause
            frames.append(summary_frame)
    
    # Save animation
    ani = FuncAnimation(fig, lambda i: plt.imshow(frames[i]), frames=len(frames), blit=False)
    writer = PillowWriter(fps=fps)
    ani.save(output_path, writer=writer)
    
    plt.close(fig)
    logger.info(f"Agent performance visualization saved to {output_path}")
    logger.info(f"Episode rewards: {episode_rewards}")
    
    return str(output_path)

def create_learning_progress_animation(
    rewards_file,
    q_networks_dir,
    environment,
    state_samples=10,
    output_dir=None,
    filename="learning_progress.gif",
    fps=5
):
    """
    Create an animation showing how the agent's Q-function changes during training.
    
    Args:
        rewards_file (str): Path to the saved rewards numpy file.
        q_networks_dir (str): Directory containing saved Q-network models at different epochs.
        environment: The environment for state space sampling.
        state_samples (int): Number of state samples to evaluate Q-values on.
        output_dir (str): Directory to save the animation.
        filename (str): Filename for the output GIF.
        fps (int): Frames per second for the animation.
        
    Returns:
        str: Path to the saved animation file.
    """
    # Load rewards data
    rewards = np.load(rewards_file)
    
    # Get list of Q-network model files
    q_networks_dir = Path(q_networks_dir)
    model_files = sorted([f for f in q_networks_dir.glob("*model_episode_*.pt")])
    
    if not model_files:
        raise FileNotFoundError(f"No model files found in {q_networks_dir}")
    
    # Create output directory if needed
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename
    else:
        output_path = Path(filename)
    
    # Sample states from environment
    states = []
    state_dim = environment.observation_space.shape[0]
    action_dim = environment.action_space.n
    
    # Generate baseline random agent for comparison
    base_agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    
    # Reset environment and collect states
    for _ in range(state_samples):
        state = environment.reset()
        for _ in range(np.random.randint(0, 10)):  # Take a few random steps
            action = environment.action_space.sample()
            next_state, _, done, truncated, _ = environment.step(action)
            if done or truncated:
                state = environment.reset()
            else:
                state = next_state
        states.append(state)
    
    # Create figure for visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create frames
    frames = []
    
    for i, model_file in enumerate(model_files):
        # Extract episode number from filename
        try:
            episode = int(model_file.stem.split("_episode_")[1])
        except:
            episode = i * 100  # Fallback if episode number can't be extracted
        
        # Load model
        agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
        agent.load_model(model_file)
        agent.eval_mode()
        
        # Evaluate Q-values for all states and actions
        all_q_values = []
        
        for state in states:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                q_values = agent.q_network(state_tensor).squeeze().cpu().numpy()
                all_q_values.append(q_values)
        
        # Calculate average Q-values for each action
        avg_q_values = np.mean(all_q_values, axis=0)
        max_q_values = np.max(all_q_values, axis=0)
        min_q_values = np.min(all_q_values, axis=0)
        
        # Clear plot
        ax.clear()
        
        # Plot rewards up to this episode
        episode_idx = min(episode, len(rewards) - 1)
        ax.plot(rewards[:episode_idx+1], 'b-', alpha=0.3, label="Episode Rewards")
        
        # Plot moving average if enough data
        if episode_idx >= 100:
            moving_avg = np.convolve(rewards[:episode_idx+1], np.ones(100)/100, mode='valid')
            ax.plot(np.arange(99, episode_idx + 1), moving_avg, 'r-', linewidth=2, label="100-Episode Moving Avg")
        
        # Plot success threshold
        ax.axhline(y=200, color='g', linestyle='--', label="Success Threshold (200)")
        
        # Plot current position
        ax.axvline(x=episode_idx, color='k', linestyle='-', label="Current Episode")
        ax.plot(episode_idx, rewards[episode_idx], 'ro', markersize=8)
        
        # Set labels and title
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.set_title(f"Training Progress (Episode {episode})")
        
        # Add Q-value information in the corner
        info_text = f"Avg Q-values:\n"
        
        if action_dim == 4:  # Lunar Lander discrete actions
            action_names = ["Left", "Nothing", "Right", "Main"]
        else:
            action_names = [f"A{i}" for i in range(action_dim)]
        
        for i, (action, q_value) in enumerate(zip(action_names, avg_q_values)):
            info_text += f"{action}: {q_value:.2f}\n"
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Add legend
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        # Ensure consistent y-axis scaling
        ax.set_ylim([min(-200, np.min(rewards[:episode_idx+1]) - 50), 
                     max(300, np.max(rewards[:episode_idx+1]) + 50)])
        
        # Capture frame
        fig.tight_layout()
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
    
    # Save animation
    ani = FuncAnimation(fig, lambda i: plt.imshow(frames[i]), frames=len(frames), blit=False)
    writer = PillowWriter(fps=fps)
    ani.save(output_path, writer=writer)
    
    plt.close(fig)
    logger.info(f"Learning progress animation saved to {output_path}")
    
    return str(output_path)

def main():
    """Main function."""
    args = parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create the environment
    env = LunarLanderEnv(
        continuous=args.continuous,
        render_mode="rgb_array",  # Use rgb_array for recording
        seed=args.seed,
        max_episode_steps=args.max_steps
    )
    
    # Create and load the agent
    agent = create_agent(args, env)
    
    # Get model name from the model path
    model_name = Path(args.model_path).stem
    
    # Record episodes
    for episode in range(1, args.num_episodes + 1):
        logger.info(f"Recording episode {episode}/{args.num_episodes}")
        
        # Record the episode
        reward, frames, states, q_values = record_episode(args, env, agent)
        
        logger.info(f"Episode {episode} completed with reward: {reward:.2f}")
        
        # Save the episode as a video if requested
        if args.save_video:
            output_path = os.path.join(args.output_dir, f"{model_name}_episode_{episode}.mp4")
            save_episode_video(frames, q_values, output_path, args.fps, args.show_q_values)
        
        # Plot state values if requested
        if args.plot_state_values:
            output_path = os.path.join(args.output_dir, f"{model_name}_episode_{episode}_states.png")
            plot_state_values(states, output_path)
        
        # Display the episode in real-time
        if not args.save_video:
            display_episode(frames, args.delay)
    
    # Visualize Q-value distribution
    if args.show_q_values:
        visualize_q_value_distribution(agent, env, args)
    
    # Close the environment
    env.close()
    
    logger.info("Visualization complete!")

if __name__ == "__main__":
    main() 