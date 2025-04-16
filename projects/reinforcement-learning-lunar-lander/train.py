#!/usr/bin/env python3
import os
import argparse
import numpy as np
import logging
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import random
import time
import json
from pathlib import Path
from tqdm import tqdm

from environment import LunarLanderEnvironment
from agent import DQNAgent, DoubleDQNAgent

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/train.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Training")

def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Set random seed to {seed}")

def create_agent(agent_type: str, env: LunarLanderEnvironment, **kwargs) -> Any:
    """
    Create an agent of the specified type.
    
    Args:
        agent_type: Type of agent ('dqn' or 'double_dqn')
        env: Environment instance
        **kwargs: Additional arguments for agent constructor
        
    Returns:
        The created agent instance
    """
    state_dim = env.state_dim
    action_dim = env.action_dim
    
    # Set default parameters if not provided
    if 'learning_rate' not in kwargs:
        kwargs['learning_rate'] = 0.001
    if 'hidden_dim' not in kwargs:
        kwargs['hidden_dim'] = [128, 128]
    if 'gamma' not in kwargs:
        kwargs['gamma'] = 0.99
    if 'epsilon_start' not in kwargs:
        kwargs['epsilon_start'] = 1.0
    if 'epsilon_end' not in kwargs:
        kwargs['epsilon_end'] = 0.01
    if 'epsilon_decay' not in kwargs:
        kwargs['epsilon_decay'] = 0.995
    if 'memory_size' not in kwargs:
        kwargs['memory_size'] = 100000
    if 'batch_size' not in kwargs:
        kwargs['batch_size'] = 64
    if 'target_update' not in kwargs:
        kwargs['target_update'] = 10
    
    # Create agent based on type
    if agent_type.lower() == 'dqn':
        logger.info(f"Creating DQNAgent with parameters: {kwargs}")
        return DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            **kwargs
        )
    elif agent_type.lower() == 'double_dqn':
        logger.info(f"Creating DoubleDQNAgent with parameters: {kwargs}")
        return DoubleDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

def train_agent(
    agent_type,
    env,
    num_episodes=1000,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.995,
    max_steps=1000,
    target_update_freq=10,
    update_start=1000,
    batch_size=64,
    gamma=0.99,
    learning_rate=0.001,
    device="cpu",
    render=False,
    save_dir="models",
    log_interval=10,
    eval_interval=100,
    verbose=True,
    save_rewards=True
):
    """
    Train a DQN or Double DQN agent on the Lunar Lander environment.
    
    Args:
        agent_type (str): Type of agent to train ('dqn' or 'double_dqn').
        env (LunarLanderEnvironment): The environment to train on.
        num_episodes (int): Number of episodes to train for.
        epsilon_start (float): Initial epsilon value for exploration.
        epsilon_end (float): Final epsilon value for exploration.
        epsilon_decay (float): Rate of epsilon decay per episode.
        max_steps (int): Maximum steps per episode.
        target_update_freq (int): Frequency of target network updates.
        update_start (int): Number of steps before starting updates.
        batch_size (int): Batch size for training.
        gamma (float): Discount factor.
        learning_rate (float): Learning rate for the optimizer.
        device (str): Device to train on ('cpu' or 'cuda').
        render (bool): Whether to render the environment during training.
        save_dir (str): Directory to save models and results.
        log_interval (int): Episode interval for logging.
        eval_interval (int): Episode interval for evaluation.
        verbose (bool): Whether to print training progress.
        save_rewards (bool): Whether to save rewards to a file.
        
    Returns:
        dict: Dictionary containing training results.
    """
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = Path(save_dir) / f"{agent_type}_{timestamp}"
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] if env.continuous else env.action_space.n
    
    logger.info(f"Initializing {agent_type} agent with state_dim={state_dim}, action_dim={action_dim}")
    
    # Create the agent
    if agent_type.lower() == "dqn":
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            buffer_size=100000,
            batch_size=batch_size,
            device=device
        )
    elif agent_type.lower() == "double_dqn":
        agent = DoubleDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            buffer_size=100000,
            batch_size=batch_size,
            device=device
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    logger.info(f"Starting training for {num_episodes} episodes")
    
    # Training variables
    episode_rewards = []
    episode_lengths = []
    avg_rewards = []
    eval_rewards = []
    
    # Progress bar
    pbar = tqdm(range(1, num_episodes + 1), desc="Training", disable=not verbose)
    
    # Training loop
    start_time = time.time()
    total_steps = 0
    
    for episode in pbar:
        state = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        
        for step in range(1, max_steps + 1):
            # Select action
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Learn from experience
            if total_steps >= update_start and len(agent.memory) >= batch_size:
                loss = agent.learn()
            
            # Update target network
            if total_steps % target_update_freq == 0 and total_steps >= update_start:
                agent.update_target_network()
            
            # Render if needed
            if render:
                env.render()
            
            # Update variables
            state = next_state
            episode_reward += reward
            total_steps += 1
            
            # Break if episode is done
            if done or truncated:
                break
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Track episode stats
        episode_rewards.append(episode_reward)
        episode_lengths.append(step)
        
        # Calculate average reward over last 100 episodes
        avg_reward = np.mean(episode_rewards[-100:])
        avg_rewards.append(avg_reward)
        
        # Update progress bar
        pbar.set_postfix({
            "reward": f"{episode_reward:.2f}",
            "avg_reward": f"{avg_reward:.2f}",
            "epsilon": f"{agent.epsilon:.3f}",
            "steps": step
        })
        
        # Log progress
        if episode % log_interval == 0:
            logger.info(
                f"Episode {episode}/{num_episodes}: "
                f"Reward: {episode_reward:.2f}, "
                f"Avg Reward: {avg_reward:.2f}, "
                f"Epsilon: {agent.epsilon:.3f}, "
                f"Steps: {step}"
            )
        
        # Evaluate agent
        if episode % eval_interval == 0 or episode == num_episodes:
            eval_reward = evaluate_agent(agent, env, num_episodes=5, render=False)
            eval_rewards.append(eval_reward)
            logger.info(f"Evaluation at episode {episode}: Average reward: {eval_reward:.2f}")
        
        # Save model periodically
        if episode % 100 == 0 or episode == num_episodes:
            model_path = save_path / f"{agent_type}_model_episode_{episode}.pt"
            agent.save_model(model_path)
            logger.info(f"Model saved to {model_path}")
    
    # Training completed
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f} seconds")
    
    # Save final model
    final_model_path = save_path / f"{agent_type}_final_model.pt"
    agent.save_model(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Save training results
    if save_rewards:
        # Save rewards
        rewards_path = save_path / "episode_rewards.npy"
        np.save(rewards_path, np.array(episode_rewards))
        
        # Save average rewards
        avg_rewards_path = save_path / "avg_rewards.npy"
        np.save(avg_rewards_path, np.array(avg_rewards))
        
        # Save evaluation rewards
        eval_rewards_path = save_path / "eval_rewards.npy"
        np.save(eval_rewards_path, np.array(eval_rewards))
        
        # Save episode lengths
        lengths_path = save_path / "episode_lengths.npy"
        np.save(lengths_path, np.array(episode_lengths))
        
        logger.info(f"Training results saved to {save_path}")
    
    # Plot training results
    plot_training_results(
        episode_rewards,
        avg_rewards,
        eval_rewards,
        episode_lengths,
        agent_type,
        save_path
    )
    
    # Return training results
    return {
        "episode_rewards": episode_rewards,
        "avg_rewards": avg_rewards,
        "eval_rewards": eval_rewards,
        "episode_lengths": episode_lengths,
        "final_model_path": str(final_model_path),
        "results_path": str(save_path)
    }

def evaluate_agent(agent, env, num_episodes=10, render=False, seed=None):
    """
    Evaluate an agent on the environment for a given number of episodes.
    
    Args:
        agent: The agent to evaluate.
        env: The environment to evaluate on.
        num_episodes (int): Number of episodes to evaluate for.
        render (bool): Whether to render the environment during evaluation.
        seed (int): Random seed for reproducibility.
        
    Returns:
        float: Average reward across episodes.
    """
    rewards = []
    
    # Set evaluation mode
    agent.eval_mode()
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # Select action (no exploration)
            action = agent.select_action(state, evaluate=True)
            
            # Take action
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Render if needed
            if render:
                env.render()
            
            # Update variables
            state = next_state
            episode_reward += reward
        
        rewards.append(episode_reward)
    
    # Set training mode
    agent.train_mode()
    
    # Return average reward
    return np.mean(rewards)

def plot_training_results(episode_rewards, avg_rewards, eval_rewards, episode_lengths, agent_type, save_path):
    """
    Plot training results and save figures.
    
    Args:
        episode_rewards (list): List of episode rewards.
        avg_rewards (list): List of average rewards over last 100 episodes.
        eval_rewards (list): List of evaluation rewards.
        episode_lengths (list): List of episode lengths.
        agent_type (str): Type of agent.
        save_path (Path): Directory to save figures.
    """
    # Create figure directory
    figures_path = save_path / "figures"
    figures_path.mkdir(exist_ok=True)
    
    # Plot episode rewards
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards)
    plt.plot(np.arange(len(episode_rewards)), avg_rewards, 'r-', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'{agent_type} Training Rewards')
    plt.legend(['Episode Reward', 'Average Reward (last 100)'])
    plt.grid(True)
    plt.savefig(figures_path / 'episode_rewards.png')
    plt.close()
    
    # Plot evaluation rewards
    eval_episodes = np.arange(100, len(episode_rewards) + 1, 100)
    if len(eval_rewards) < len(eval_episodes):
        eval_episodes = eval_episodes[:len(eval_rewards)]
    
    plt.figure(figsize=(10, 6))
    plt.plot(eval_episodes, eval_rewards, 'g-', linewidth=2, marker='o')
    plt.xlabel('Episode')
    plt.ylabel('Evaluation Reward')
    plt.title(f'{agent_type} Evaluation Rewards')
    plt.grid(True)
    plt.savefig(figures_path / 'eval_rewards.png')
    plt.close()
    
    # Plot episode lengths
    plt.figure(figsize=(10, 6))
    plt.plot(episode_lengths)
    plt.xlabel('Episode')
    plt.ylabel('Length')
    plt.title(f'{agent_type} Episode Lengths')
    plt.grid(True)
    plt.savefig(figures_path / 'episode_lengths.png')
    plt.close()
    
    # Combined plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(episode_rewards, alpha=0.6)
    plt.plot(np.arange(len(episode_rewards)), avg_rewards, 'r-', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'{agent_type} Training Progress')
    plt.legend(['Episode Reward', 'Average Reward (last 100)'])
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(episode_lengths, alpha=0.6)
    plt.xlabel('Episode')
    plt.ylabel('Length')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(figures_path / 'training_summary.png')
    plt.close()
    
    logger.info(f"Training plots saved to {figures_path}")

def compare_agents(env, agents_config, num_episodes=1000, save_dir="models/comparison"):
    """
    Train and compare multiple agents on the same environment.
    
    Args:
        env (LunarLanderEnvironment): The environment to train on.
        agents_config (list): List of agent configurations.
        num_episodes (int): Number of episodes to train for.
        save_dir (str): Directory to save comparison results.
        
    Returns:
        dict: Dictionary containing comparison results.
    """
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = Path(save_dir) / f"comparison_{timestamp}"
    save_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting comparison of {len(agents_config)} agents")
    
    # Train each agent
    results = {}
    all_rewards = {}
    all_avg_rewards = {}
    
    for agent_config in agents_config:
        agent_type = agent_config.pop("type")
        logger.info(f"Training {agent_type} agent")
        
        # Train agent
        agent_results = train_agent(
            agent_type=agent_type,
            env=env,
            num_episodes=num_episodes,
            **agent_config
        )
        
        # Store results
        results[agent_type] = agent_results
        all_rewards[agent_type] = agent_results["episode_rewards"]
        all_avg_rewards[agent_type] = agent_results["avg_rewards"]
    
    # Plot comparison
    compare_agents_plot(all_rewards, all_avg_rewards, save_path)
    
    logger.info(f"Comparison completed and saved to {save_path}")
    
    return results

def compare_agents_plot(all_rewards, all_avg_rewards, save_path):
    """
    Plot comparison of multiple agents.
    
    Args:
        all_rewards (dict): Dictionary of episode rewards for each agent.
        all_avg_rewards (dict): Dictionary of average rewards for each agent.
        save_path (Path): Directory to save comparison figures.
    """
    # Create figure directory
    figures_path = save_path / "figures"
    figures_path.mkdir(exist_ok=True)
    
    # Plot average rewards
    plt.figure(figsize=(12, 8))
    
    for agent_type, avg_rewards in all_avg_rewards.items():
        plt.plot(avg_rewards, linewidth=2, label=agent_type)
    
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (last 100)')
    plt.title('Comparison of Agent Performance')
    plt.legend()
    plt.grid(True)
    plt.savefig(figures_path / 'comparison_avg_rewards.png')
    plt.close()
    
    # Plot episode rewards (with smoothing)
    plt.figure(figsize=(12, 8))
    
    for agent_type, rewards in all_rewards.items():
        # Apply smoothing with a window of 10
        smoothed_rewards = np.convolve(rewards, np.ones(10)/10, mode='valid')
        plt.plot(smoothed_rewards, linewidth=1, alpha=0.7, label=f"{agent_type} (smoothed)")
    
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Episode Reward')
    plt.title('Comparison of Agent Performance (Smoothed)')
    plt.legend()
    plt.grid(True)
    plt.savefig(figures_path / 'comparison_smoothed_rewards.png')
    plt.close()
    
    # Box plot of final 100 episode rewards
    plt.figure(figsize=(10, 6))
    
    data = [rewards[-100:] for agent_type, rewards in all_rewards.items()]
    labels = list(all_rewards.keys())
    
    plt.boxplot(data, labels=labels)
    plt.ylabel('Reward')
    plt.title('Final 100 Episode Rewards Distribution')
    plt.grid(True, axis='y')
    plt.savefig(figures_path / 'comparison_final_distribution.png')
    plt.close()
    
    logger.info(f"Comparison plots saved to {figures_path}")

def main():
    """
    Main function for training.
    """
    parser = argparse.ArgumentParser(description="Train RL agent for Lunar Lander")
    parser.add_argument("--agent", type=str, default="dqn", choices=["dqn", "double_dqn"],
                        help="Type of agent to train")
    parser.add_argument("--continuous", action="store_true",
                        help="Use continuous action space (LunarLanderContinuous-v2)")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Number of training episodes")
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="Maximum steps per episode")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--epsilon_start", type=float, default=1.0,
                        help="Initial exploration rate")
    parser.add_argument("--epsilon_end", type=float, default=0.01,
                        help="Final exploration rate")
    parser.add_argument("--epsilon_decay", type=float, default=0.995,
                        help="Exploration rate decay")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--hidden_dim", type=int, nargs="+", default=[128, 128],
                        help="Hidden dimensions of the network")
    parser.add_argument("--memory_size", type=int, default=100000,
                        help="Size of the replay buffer")
    parser.add_argument("--target_update", type=int, default=10,
                        help="Target network update frequency")
    parser.add_argument("--eval_interval", type=int, default=50,
                        help="Interval between evaluations")
    parser.add_argument("--num_eval_episodes", type=int, default=10,
                        help="Number of episodes for evaluation")
    parser.add_argument("--save_interval", type=int, default=100,
                        help="Interval between model saves")
    parser.add_argument("--render", action="store_true",
                        help="Render training episodes")
    parser.add_argument("--render_eval", action="store_true",
                        help="Render evaluation episodes")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Name of the experiment")
    parser.add_argument("--save_dir", type=str, default="models",
                        help="Directory to save models")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto, cpu, cuda)")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Set random seed
    set_seed(args.seed)
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    
    # Create environment
    env = LunarLanderEnvironment(
        continuous=args.continuous,
        render_mode="human" if args.render else None,
        seed=args.seed
    )
    
    # Create agent
    agent = create_agent(
        agent_type=args.agent,
        env=env,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        memory_size=args.memory_size,
        batch_size=args.batch_size,
        target_update=args.target_update,
        device=device
    )
    
    # Train agent
    train_agent(
        agent_type=args.agent,
        env=env,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        num_eval_episodes=args.num_eval_episodes,
        save_interval=args.save_interval,
        render=args.render,
        render_eval=args.render_eval,
        save_dir=args.save_dir,
        exp_name=args.exp_name
    )
    
    # Close environment
    env.close()

if __name__ == "__main__":
    main() 