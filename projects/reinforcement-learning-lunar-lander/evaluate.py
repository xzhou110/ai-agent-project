import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import json
import logging
from tqdm import tqdm
import time
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from environment import LunarLanderEnvironment
from agent import DQNAgent, DoubleDQNAgent
from utils import moving_average, save_training_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def evaluate_agent(env: LunarLanderEnvironment, 
                  agent, 
                  num_episodes: int = 10, 
                  render: bool = False,
                  record: bool = False,
                  record_dir: str = "results/evaluation",
                  record_freq: int = 2) -> Dict[str, Any]:
    """
    Evaluate an agent's performance over several episodes.
    
    Args:
        env: The environment to evaluate on
        agent: The agent to evaluate
        num_episodes: Number of evaluation episodes
        render: Whether to render the environment
        record: Whether to record episodes as GIFs/MP4s
        record_dir: Directory to save recordings
        record_freq: Frequency of episodes to record
        
    Returns:
        Dictionary containing evaluation statistics
    """
    logger.info(f"Evaluating agent over {num_episodes} episodes")
    
    # Create recording directory if needed
    if record:
        os.makedirs(record_dir, exist_ok=True)
    
    # Track metrics
    rewards = []
    episode_lengths = []
    successful_landings = 0
    crashes = 0
    
    # Run evaluation episodes
    for episode in tqdm(range(1, num_episodes + 1)):
        state = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        # Whether to record this episode
        should_record = record and (episode % record_freq == 0)
        frames = [] if should_record else None
        
        while not done:
            # Select action using the agent's policy
            action = agent.predict(state)
            
            # Take a step in the environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Collect frames if recording
            if should_record and env.render_mode == 'rgb_array':
                frames.append(env.render())
            elif render:
                env.render()
                time.sleep(0.01)  # Small delay to make rendering visible
            
            # Update state and counters
            state = next_state
            episode_reward += reward
            steps += 1
            
            # Check if landing was successful
            if terminated:
                # In LunarLander, reward threshold for solving is 200
                # A reasonably good landing often gives rewards > 100
                if episode_reward > 100:
                    successful_landings += 1
                else:
                    crashes += 1
        
        # Record episode statistics
        rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        # Save recording
        if should_record and frames:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            gif_path = os.path.join(record_dir, f"episode_{episode}_{timestamp}.gif")
            env.save_frames_as_gif(frames, gif_path)
            logger.info(f"Saved recording to {gif_path}")
            
            # Optionally save as MP4 as well
            mp4_path = os.path.join(record_dir, f"episode_{episode}_{timestamp}.mp4")
            env.save_frames_as_mp4(frames, mp4_path)
    
    # Compute statistics
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    avg_length = np.mean(episode_lengths)
    
    success_rate = (successful_landings / num_episodes) * 100
    crash_rate = (crashes / num_episodes) * 100
    
    # Log results
    logger.info(f"Evaluation results over {num_episodes} episodes:")
    logger.info(f"  Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
    logger.info(f"  Average episode length: {avg_length:.2f}")
    logger.info(f"  Success rate: {success_rate:.2f}%")
    logger.info(f"  Crash rate: {crash_rate:.2f}%")
    
    # Create evaluation statistics dictionary
    eval_stats = {
        "rewards": rewards,
        "episode_lengths": episode_lengths,
        "avg_reward": avg_reward,
        "std_reward": std_reward,
        "min_reward": np.min(rewards),
        "max_reward": np.max(rewards),
        "avg_length": avg_length,
        "success_rate": success_rate,
        "crash_rate": crash_rate,
        "num_episodes": num_episodes
    }
    
    # Plot reward distribution
    plt.figure(figsize=(10, 6))
    plt.hist(rewards, bins=15, alpha=0.7)
    plt.axvline(avg_reward, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {avg_reward:.2f}')
    plt.xlabel("Episode Reward")
    plt.ylabel("Frequency")
    plt.title("Reward Distribution During Evaluation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if record:
        plt.savefig(os.path.join(record_dir, "reward_distribution.png"))
        plt.close()
    
    return eval_stats

def analyze_states(env: LunarLanderEnvironment,
                  agent,
                  num_episodes: int = 5,
                  save_dir: str = "results/state_analysis"):
    """
    Analyze the state values and agent's actions during evaluation.
    
    Args:
        env: The environment to evaluate on
        agent: The agent to evaluate
        num_episodes: Number of episodes to analyze
        save_dir: Directory to save analysis results
    """
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Analyzing agent behavior across {num_episodes} episodes")
    
    # Track state values
    states = []
    actions = []
    rewards = []
    
    # Run episodes
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_states = []
        episode_actions = []
        episode_rewards = []
        done = False
        
        while not done:
            # Select action
            action = agent.predict(state)
            
            # Record state and action
            episode_states.append(state)
            episode_actions.append(action)
            
            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            
            episode_rewards.append(reward)
        
        # Store episode data
        states.append(np.array(episode_states))
        actions.append(np.array(episode_actions))
        rewards.append(np.array(episode_rewards))
        
        logger.info(f"Episode {episode}: {len(episode_states)} steps, " +
                    f"final reward: {sum(episode_rewards):.2f}")
    
    # Create state distribution plots
    plt.figure(figsize=(15, 10))
    
    # State components to analyze
    state_components = ["x_position", "y_position", "x_velocity", "y_velocity",
                        "angle", "angular_velocity", "left_leg_contact", "right_leg_contact"]
    
    for i, component in enumerate(state_components):
        plt.subplot(2, 4, i+1)
        
        # Collect data across all episodes
        component_data = []
        for episode_states in states:
            component_data.extend(episode_states[:, i])
        
        plt.hist(component_data, bins=20, alpha=0.7)
        plt.title(component)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "state_distribution.png"))
    
    # Analyze action distribution
    plt.figure(figsize=(10, 6))
    all_actions = np.concatenate(actions)
    action_counts = np.bincount(all_actions, minlength=env.num_actions)
    action_names = ["Do nothing", "Fire left engine", "Fire main engine", "Fire right engine"]
    
    plt.bar(range(env.num_actions), action_counts)
    plt.xticks(range(env.num_actions), action_names, rotation=45)
    plt.title("Action Distribution")
    plt.xlabel("Action")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "action_distribution.png"))
    
    logger.info("State and action analysis completed")
    
    return {
        "states": states,
        "actions": actions,
        "rewards": rewards
    }

def load_agent(model_path: str) -> Any:
    """
    Load a trained agent from a file.
    
    Args:
        model_path: Path to the saved model file
    
    Returns:
        The loaded agent
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Determine model type from filename
    model_type = "DQN"  # Default
    if "double" in model_path.lower():
        model_type = "DoubleDQN"
    
    logger.info(f"Loading {model_type} agent from {model_path}")
    
    # Create dummy environment to get state and action dimensions
    dummy_env = LunarLanderEnvironment()
    state_dim = dummy_env.num_observations
    action_dim = dummy_env.num_actions
    dummy_env.close()
    
    # Create agent
    if model_type == "DoubleDQN":
        agent = DoubleDQNAgent(state_dim=state_dim, action_dim=action_dim)
    else:
        agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    
    # Load weights
    agent.load(model_path)
    return agent

def compare_agents(model_paths: List[str], 
                  num_episodes: int = 10,
                  render: bool = False,
                  save_dir: str = "results/comparison"):
    """
    Compare multiple trained agents on the same environment.
    
    Args:
        model_paths: List of paths to saved model files
        num_episodes: Number of episodes for evaluation
        render: Whether to render the environment
        save_dir: Directory to save comparison results
    
    Returns:
        Dictionary of evaluation statistics for each agent
    """
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Comparing {len(model_paths)} agents")
    
    results = {}
    model_names = []
    
    # Evaluate each model
    for i, model_path in enumerate(model_paths):
        model_name = os.path.basename(model_path).replace(".pth", "")
        model_names.append(model_name)
        logger.info(f"Evaluating model {i+1}/{len(model_paths)}: {model_name}")
        
        # Load agent
        agent = load_agent(model_path)
        
        # Create environment
        render_mode = "human" if render else None
        env = LunarLanderEnvironment(render_mode=render_mode)
        
        # Evaluate agent
        stats = evaluate_agent(env, agent, num_episodes, render)
        results[model_name] = stats
        
        # Close environment
        env.close()
    
    # Create comparison bar chart
    plt.figure(figsize=(12, 6))
    x = np.arange(len(model_names))
    width = 0.35
    
    rewards = [results[name]["avg_reward"] for name in model_names]
    std_devs = [results[name]["std_reward"] for name in model_names]
    
    plt.bar(x, rewards, width, yerr=std_devs, align='center', alpha=0.7, ecolor='black', capsize=10)
    plt.ylabel('Average Reward')
    plt.title('Agent Performance Comparison')
    plt.xticks(x, model_names, rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, "agent_comparison.png"))
    
    # Save comparison results
    with open(os.path.join(save_dir, "comparison_results.json"), 'w') as f:
        # Convert numpy values to Python natives for JSON serialization
        serializable_results = {}
        for model, stats in results.items():
            serializable_stats = {}
            for key, value in stats.items():
                if isinstance(value, np.ndarray):
                    serializable_stats[key] = value.tolist()
                elif isinstance(value, (np.float32, np.float64)):
                    serializable_stats[key] = float(value)
                elif isinstance(value, (np.int32, np.int64)):
                    serializable_stats[key] = int(value)
                else:
                    serializable_stats[key] = value
            serializable_results[model] = serializable_stats
        
        json.dump(serializable_results, f, indent=4)
    
    logger.info(f"Comparison completed. Results saved to {save_dir}")
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained RL agents on Lunar Lander")
    
    parser.add_argument("--model", type=str, required=True,
                      help="Path to model file or directory containing model files")
    parser.add_argument("--episodes", type=int, default=10,
                      help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true",
                      help="Render environment during evaluation")
    parser.add_argument("--record", action="store_true",
                      help="Record episodes as GIFs")
    parser.add_argument("--analyze", action="store_true",
                      help="Perform detailed state and action analysis")
    parser.add_argument("--compare", action="store_true",
                      help="Compare multiple models (--model should be a directory)")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results/evaluation", exist_ok=True)
    
    # Handle comparison mode
    if args.compare:
        if not os.path.isdir(args.model):
            logger.error("For comparison mode, --model should be a directory")
            return
        
        # Find all model files in the directory
        model_files = [os.path.join(args.model, f) for f in os.listdir(args.model) 
                      if f.endswith(".pth")]
        
        if not model_files:
            logger.error(f"No model files found in {args.model}")
            return
        
        compare_agents(model_files, args.episodes, args.render)
        return
    
    # Single model evaluation
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        return
    
    # Create environment
    render_mode = "human" if args.render else ("rgb_array" if args.record else None)
    env = LunarLanderEnvironment(render_mode=render_mode, seed=args.seed)
    
    # Load agent
    agent = load_agent(args.model)
    
    # Evaluate agent
    record_dir = "results/evaluation"
    eval_stats = evaluate_agent(
        env=env,
        agent=agent,
        num_episodes=args.episodes,
        render=args.render,
        record=args.record,
        record_dir=record_dir
    )
    
    # Save evaluation results
    results_file = os.path.join(record_dir, "evaluation_results.json")
    save_training_data(eval_stats, results_file)
    
    # Perform state analysis if requested
    if args.analyze:
        analyze_states(env, agent, num_episodes=min(5, args.episodes))
    
    # Clean up
    env.close()

if __name__ == "__main__":
    main() 