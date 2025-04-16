#!/usr/bin/env python3
import os
import argparse
import logging
import torch
import json
from datetime import datetime

from environment import LunarLanderEnvironment
from agent import DQNAgent, DoubleDQNAgent
from train import train_agent
from evaluate import evaluate_agent, plot_reward_distribution

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/main.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Main")

def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    import numpy as np
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Set random seed to {seed}")

def main():
    """Main entry point for training and evaluating RL agents on LunarLander."""
    parser = argparse.ArgumentParser(description="Train and evaluate RL agents on LunarLander")
    
    # General settings
    parser.add_argument("--continuous", action="store_true",
                        help="Use continuous action space (LunarLanderContinuous-v2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto, cpu, cuda)")
    
    # Mode selection
    subparsers = parser.add_subparsers(dest="mode", help="Operating mode", required=True)
    
    # Train mode
    train_parser = subparsers.add_parser("train", help="Train an agent")
    train_parser.add_argument("--agent", type=str, default="dqn", choices=["dqn", "double_dqn"],
                       help="Type of agent to train")
    train_parser.add_argument("--episodes", type=int, default=1000,
                       help="Number of training episodes")
    train_parser.add_argument("--max_steps", type=int, default=1000,
                       help="Maximum steps per episode")
    train_parser.add_argument("--lr", type=float, default=0.001,
                       help="Learning rate")
    train_parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor")
    train_parser.add_argument("--epsilon_start", type=float, default=1.0,
                       help="Starting epsilon for exploration")
    train_parser.add_argument("--epsilon_end", type=float, default=0.01,
                       help="Final epsilon for exploration")
    train_parser.add_argument("--epsilon_decay", type=float, default=0.995,
                       help="Decay rate for epsilon")
    train_parser.add_argument("--hidden_dims", type=int, nargs="+", default=[128, 128],
                       help="Dimensions of hidden layers")
    train_parser.add_argument("--memory_size", type=int, default=100000,
                       help="Size of replay memory")
    train_parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for training")
    train_parser.add_argument("--target_update", type=int, default=10,
                       help="Target network update frequency (episodes)")
    train_parser.add_argument("--eval_interval", type=int, default=50,
                       help="Evaluation interval (episodes)")
    train_parser.add_argument("--eval_episodes", type=int, default=10,
                       help="Number of episodes for evaluation during training")
    train_parser.add_argument("--save_interval", type=int, default=100,
                       help="Model saving interval (episodes)")
    train_parser.add_argument("--render_train", action="store_true",
                       help="Render training episodes (slows down training)")
    train_parser.add_argument("--render_eval", action="store_true",
                       help="Render evaluation episodes during training")
    train_parser.add_argument("--output_dir", type=str, default=None,
                       help="Custom output directory for model and results")
    
    # Evaluate mode
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained agent")
    eval_parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the trained model checkpoint")
    eval_parser.add_argument("--agent", type=str, default="dqn", choices=["dqn", "double_dqn"],
                       help="Type of agent to evaluate")
    eval_parser.add_argument("--episodes", type=int, default=100,
                       help="Number of evaluation episodes")
    eval_parser.add_argument("--render", action="store_true",
                       help="Render the environment during evaluation")
    eval_parser.add_argument("--record", action="store_true",
                       help="Record evaluation episodes as video")
    eval_parser.add_argument("--video_path", type=str, default="results/videos",
                       help="Directory to save videos")
    
    # Compare mode
    compare_parser = subparsers.add_parser("compare", help="Compare multiple trained agents")
    compare_parser.add_argument("--model_paths", type=str, nargs="+", required=True,
                          help="Paths to the trained model checkpoints")
    compare_parser.add_argument("--agent_types", type=str, nargs="+", required=True,
                          help="Types of agents corresponding to model paths")
    compare_parser.add_argument("--episodes", type=int, default=50,
                          help="Number of evaluation episodes per agent")
    compare_parser.add_argument("--output_dir", type=str, default="results/comparisons",
                          help="Directory to save comparison results")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Set random seed
    set_seed(args.seed)
    
    # Prepare output directory for train mode
    if args.mode == "train" and not args.output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"results/{args.agent}_{timestamp}"
    
    os.makedirs(args.output_dir, exist_ok=True) if hasattr(args, 'output_dir') and args.output_dir else None
    
    # Create environment with appropriate settings
    render_mode = None
    if args.mode == "train" and args.render_train:
        render_mode = "human"
    elif args.mode == "evaluate" and args.render:
        render_mode = "human"
    
    env = LunarLanderEnvironment(
        continuous=args.continuous,
        render_mode=render_mode,
        seed=args.seed
    )
    
    # Execute selected mode
    if args.mode == "train":
        # Create agent
        if args.agent == "dqn":
            agent = DQNAgent(
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                hidden_dims=args.hidden_dims,
                learning_rate=args.lr,
                gamma=args.gamma,
                epsilon_start=args.epsilon_start,
                epsilon_end=args.epsilon_end,
                epsilon_decay=args.epsilon_decay,
                memory_size=args.memory_size,
                batch_size=args.batch_size,
                target_update_frequency=args.target_update,
                device=device
            )
            logger.info(f"Created DQN Agent with hidden dims {args.hidden_dims}")
        else:  # double_dqn
            agent = DoubleDQNAgent(
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                hidden_dims=args.hidden_dims,
                learning_rate=args.lr,
                gamma=args.gamma,
                epsilon_start=args.epsilon_start,
                epsilon_end=args.epsilon_end,
                epsilon_decay=args.epsilon_decay,
                memory_size=args.memory_size,
                batch_size=args.batch_size,
                target_update_frequency=args.target_update,
                device=device
            )
            logger.info(f"Created Double DQN Agent with hidden dims {args.hidden_dims}")
        
        # Train agent
        logger.info(f"Starting training for {args.episodes} episodes")
        metrics = train_agent(
            agent_type=args.agent,
            env=env,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            target_update_freq=args.target_update,
            batch_size=args.batch_size,
            gamma=args.gamma,
            learning_rate=args.lr,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay,
            device=device,
            render=args.render_train,
            save_dir=args.output_dir,
            eval_interval=args.eval_interval
        )
        
        # Save final metrics
        with open(f"{args.output_dir}/final_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Training completed. Model and metrics saved to {args.output_dir}")
        
    elif args.mode == "evaluate":
        # Load agent
        if args.agent == "dqn":
            agent = DQNAgent(
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                device=device
            )
        else:  # double_dqn
            agent = DoubleDQNAgent(
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                device=device
            )
        
        # Load model
        agent.load(args.model_path)
        logger.info(f"Loaded {args.agent} agent from {args.model_path}")
        
        # Evaluate agent
        results = evaluate_agent(
            agent=agent,
            env=env,
            num_episodes=args.episodes,
            max_steps=1000,
            render=args.render,
            record_video=args.record,
            video_path=args.video_path
        )
        
        # Save results
        os.makedirs("results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"results/evaluation_{args.agent}_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # Plot reward distribution
        plot_path = f"results/reward_dist_{args.agent}_{timestamp}.png"
        plot_reward_distribution(results["rewards"] if "rewards" in results else [], plot_path)
        
        logger.info(f"Evaluation completed. Results saved to {results_file}")
        
    elif args.mode == "compare":
        # Check if number of models matches number of agent types
        if len(args.model_paths) != len(args.agent_types):
            raise ValueError("Number of model paths must match number of agent types")
        
        # Prepare output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Load agents
        agents = []
        agent_names = []
        
        for i, (model_path, agent_type) in enumerate(zip(args.model_paths, args.agent_types)):
            # Create agent
            if agent_type.lower() == "dqn":
                agent = DQNAgent(
                    state_dim=env.state_dim,
                    action_dim=env.action_dim,
                    device=device
                )
            elif agent_type.lower() == "double_dqn":
                agent = DoubleDQNAgent(
                    state_dim=env.state_dim,
                    action_dim=env.action_dim,
                    device=device
                )
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")
            
            # Load model
            agent.load(model_path)
            logger.info(f"Loaded {agent_type} agent from {model_path}")
            
            agents.append(agent)
            agent_names.append(f"{agent_type}_{i+1}")
        
        logger.info(f"Comparing {len(agents)} agents: {agent_names}")
        
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
                num_episodes=args.episodes,
                max_steps=1000,
                render=False,
                verbose=False
            )
            
            # Store results
            all_results[name] = results
            all_rewards[name] = results["mean_reward"]
            
            logger.info(f"{name} - Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        
        # Create comparison plot
        import matplotlib.pyplot as plt
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
        plt.savefig(os.path.join(args.output_dir, "agent_comparison.png"))
        plt.close()
        
        # Save results to JSON
        with open(os.path.join(args.output_dir, "comparison_results.json"), "w") as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"Comparison completed and saved to {args.output_dir}")
    
    # Close environment
    env.close()
    logger.info("Done.")

if __name__ == "__main__":
    main() 