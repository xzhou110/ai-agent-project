#!/usr/bin/env python3
import os
import argparse
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json
import time

from environment import LunarLanderEnvironment
from agent import DQNAgent, DoubleDQNAgent

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Evaluation")

def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Set random seed to {seed}")

def load_agent(model_path: str, agent_type: str, env: LunarLanderEnvironment, device: torch.device) -> object:
    """
    Load a trained agent from a checkpoint.
    
    Args:
        model_path: Path to the model checkpoint
        agent_type: Type of agent ('dqn' or 'double_dqn')
        env: Environment instance
        device: Device to load the model on
        
    Returns:
        Loaded agent
    """
    state_dim = env.state_dim
    action_dim = env.action_dim
    
    # Create agent based on type
    if agent_type.lower() == 'dqn':
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device
        )
    elif agent_type.lower() == 'double_dqn':
        agent = DoubleDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    # Load model
    agent.load(model_path)
    logger.info(f"Loaded {agent_type} agent from {model_path}")
    
    return agent

def evaluate_agent(
    agent: object,
    env: LunarLanderEnvironment,
    num_episodes: int = 100,
    max_steps: int = 1000,
    render: bool = False,
    record_video: bool = False,
    video_path: str = "results/videos",
    verbose: bool = True
) -> dict:
    """
    Evaluate an agent's performance.
    
    Args:
        agent: The agent to evaluate
        env: The environment to evaluate in
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        render: Whether to render evaluation episodes
        record_video: Whether to record video
        video_path: Path to save videos
        verbose: Whether to print detailed information
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Set agent to evaluation mode
    agent.eval()
    
    # Metrics to track
    rewards = []
    episode_lengths = []
    success_count = 0
    
    # If recording, setup video dir
    if record_video:
        os.makedirs(video_path, exist_ok=True)
    
    # Set environment to render or record if needed
    if render and not env.render_mode:
        env.render_mode = "human"
    
    # Run evaluation episodes
    if verbose:
        logger.info(f"Evaluating agent over {num_episodes} episodes...")
        episode_range = tqdm(range(num_episodes))
    else:
        episode_range = range(num_episodes)
    
    for episode in episode_range:
        state, _ = env.reset()
        episode_reward = 0
        
        # Record video for selected episodes
        if record_video and episode < 5:  # Record first 5 episodes
            video_file = os.path.join(video_path, f"episode_{episode}.mp4")
            env.start_recording(video_file)
        
        for step in range(max_steps):
            # Select action without exploration
            action = agent.select_action(state, explore=False)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
            # Render if needed
            if render:
                env.render()
            
            # Break if done
            if done:
                # Check if it was a successful landing
                if terminated and reward >= 100:  # Assuming reward threshold for success
                    success_count += 1
                break
        
        # Stop recording if we were recording
        if record_video and episode < 5:
            env.stop_recording()
        
        # Store episode metrics
        rewards.append(episode_reward)
        episode_lengths.append(step + 1)
    
    # Calculate statistics
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    min_reward = np.min(rewards)
    max_reward = np.max(rewards)
    median_reward = np.median(rewards)
    success_rate = success_count / num_episodes
    mean_episode_length = np.mean(episode_lengths)
    
    # Create results dictionary
    results = {
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "min_reward": float(min_reward),
        "max_reward": float(max_reward),
        "median_reward": float(median_reward),
        "success_rate": float(success_rate),
        "mean_episode_length": float(mean_episode_length),
        "episodes": num_episodes
    }
    
    # Print results
    if verbose:
        logger.info(f"Evaluation Results:")
        logger.info(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        logger.info(f"Min/Max Reward: {min_reward:.2f}/{max_reward:.2f}")
        logger.info(f"Median Reward: {median_reward:.2f}")
        logger.info(f"Success Rate: {success_rate:.2%}")
        logger.info(f"Mean Episode Length: {mean_episode_length:.2f}")
    
    return results

def plot_reward_distribution(rewards: list, save_path: str = None) -> None:
    """
    Plot the distribution of rewards.
    
    Args:
        rewards: List of episode rewards
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.hist(rewards, bins=20, alpha=0.7)
    plt.axvline(np.mean(rewards), color='r', linestyle='--', label=f'Mean: {np.mean(rewards):.2f}')
    plt.axvline(np.median(rewards), color='g', linestyle='--', label=f'Median: {np.median(rewards):.2f}')
    plt.xlabel('Episode Reward')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved reward distribution plot to {save_path}")
    
    plt.close()

def compare_agents(
    agents: list,
    agent_names: list,
    env: LunarLanderEnvironment,
    num_episodes: int = 50,
    max_steps: int = 1000,
    save_dir: str = "results/comparisons"
) -> dict:
    """
    Compare multiple agents on the same environment.
    
    Args:
        agents: List of agents to compare
        agent_names: Names of the agents
        env: Environment to evaluate in
        num_episodes: Number of evaluation episodes per agent
        max_steps: Maximum steps per episode
        save_dir: Directory to save comparison results
        
    Returns:
        Dictionary with comparison results
    """
    if len(agents) != len(agent_names):
        raise ValueError("Number of agents must match number of agent names")
    
    # Create directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Results for each agent
    all_results = {}
    all_rewards = {}
    
    # Evaluate each agent
    for agent, name in zip(agents, agent_names):
        logger.info(f"Evaluating {name}...")
        
        # Evaluate the agent
        results = evaluate_agent(
            agent=agent,
            env=env,
            num_episodes=num_episodes,
            max_steps=max_steps,
            render=False,
            verbose=False
        )
        
        # Store results
        all_results[name] = results
        all_rewards[name] = results["mean_reward"]
        
        logger.info(f"{name} - Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    
    # Create comparison plot
    plt.figure(figsize=(12, 8))
    
    # Bar plot for mean rewards
    means = [all_results[name]["mean_reward"] for name in agent_names]
    stds = [all_results[name]["std_reward"] for name in agent_names]
    
    plt.bar(agent_names, means, yerr=stds, alpha=0.7, capsize=10)
    plt.axhline(y=200, color='r', linestyle='--', label='Target Score (200)')
    plt.ylabel('Mean Reward')
    plt.title('Agent Performance Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save comparison plot
    plt.savefig(os.path.join(save_dir, "agent_comparison.png"))
    plt.close()
    
    # Save results to JSON
    with open(os.path.join(save_dir, "comparison_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"Comparison completed and saved to {save_dir}")
    
    return all_results

def main():
    """
    Main function for evaluating agents.
    """
    parser = argparse.ArgumentParser(description="Evaluate RL agents on Lunar Lander")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model checkpoint")
    parser.add_argument("--agent", type=str, default="dqn", choices=["dqn", "double_dqn"],
                        help="Type of agent to evaluate")
    parser.add_argument("--continuous", action="store_true",
                        help="Use continuous action space (LunarLanderContinuous-v2)")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of evaluation episodes")
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="Maximum steps per episode")
    parser.add_argument("--render", action="store_true",
                        help="Render the environment during evaluation")
    parser.add_argument("--record", action="store_true",
                        help="Record evaluation episodes")
    parser.add_argument("--video_path", type=str, default="results/videos",
                        help="Directory to save videos")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare this agent with others in the specified directory")
    parser.add_argument("--compare_dir", type=str, default="models",
                        help="Directory containing models to compare")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Set random seed
    set_seed(args.seed)
    
    # Create environment
    render_mode = "human" if args.render else None
    env = LunarLanderEnvironment(
        continuous=args.continuous,
        render_mode=render_mode,
        seed=args.seed
    )
    
    # Load agent
    agent = load_agent(args.model_path, args.agent, env, device)
    
    if args.compare:
        # Collect all models to compare
        agents = [agent]
        agent_names = [f"{args.agent}_current"]
        
        # Look for other models in compare_dir
        if os.path.exists(args.compare_dir):
            for model_file in os.listdir(args.compare_dir):
                if model_file.endswith(".pt") and os.path.join(args.compare_dir, model_file) != args.model_path:
                    try:
                        # Try to determine agent type from filename
                        if "double_dqn" in model_file.lower():
                            agent_type = "double_dqn"
                        elif "dqn" in model_file.lower():
                            agent_type = "dqn"
                        else:
                            # Skip if can't determine
                            continue
                        
                        # Load the agent
                        compare_agent = load_agent(
                            os.path.join(args.compare_dir, model_file),
                            agent_type, 
                            env,
                            device
                        )
                        
                        agents.append(compare_agent)
                        agent_names.append(f"{agent_type}_{model_file.replace('.pt', '')}")
                    except Exception as e:
                        logger.warning(f"Failed to load model {model_file}: {e}")
        
        if len(agents) > 1:
            # Compare agents
            comparison_results = compare_agents(
                agents=agents,
                agent_names=agent_names,
                env=env,
                num_episodes=args.episodes,
                max_steps=args.max_steps,
                save_dir="results/comparisons"
            )
        else:
            logger.warning("No other models found for comparison. Evaluating single agent.")
            # Fall back to single agent evaluation
            args.compare = False
    
    if not args.compare:
        # Standard evaluation
        results = evaluate_agent(
            agent=agent,
            env=env,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            render=args.render,
            record_video=args.record,
            video_path=args.video_path
        )
        
        # Save results
        os.makedirs("results", exist_ok=True)
        
        # Save as JSON
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"results/evaluation_{args.agent}_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {results_file}")
    
    # Close environment
    env.close()

if __name__ == "__main__":
    main()