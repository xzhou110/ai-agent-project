#!/usr/bin/env python3
import os
import random
import logging
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Configure logging
logger = logging.getLogger("Utils")

def set_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set CUDA seed if available
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        logger.info(f"Random seed set to {seed}")

def create_output_dir(base_dir="results", agent_type=None, timestamp=None):
    """
    Create output directory for experiment results.
    
    Args:
        base_dir (str): Base directory for results.
        agent_type (str): Type of agent.
        timestamp (str): Timestamp for directory name (if None, current time is used).
        
    Returns:
        Path: Path to created directory.
    """
    # Create timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Create directory name
    dirname = timestamp
    if agent_type is not None:
        dirname = f"{agent_type}_{timestamp}"
    
    # Create directory
    output_dir = Path(base_dir) / dirname
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created output directory: {output_dir}")
    
    return output_dir

def save_config(config, output_dir, filename="config.json"):
    """
    Save configuration to JSON file.
    
    Args:
        config (dict): Configuration dictionary.
        output_dir (str or Path): Output directory.
        filename (str): Output filename.
    """
    output_path = Path(output_dir) / filename
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    logger.info(f"Configuration saved to {output_path}")

def load_config(config_path):
    """
    Load configuration from JSON file.
    
    Args:
        config_path (str or Path): Path to configuration file.
        
    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    logger.info(f"Configuration loaded from {config_path}")
    
    return config

def plot_rewards(rewards, window=100, title="Training Rewards", output_path=None):
    """
    Plot episode rewards with moving average.
    
    Args:
        rewards (list): List of episode rewards.
        window (int): Window size for moving average.
        title (str): Plot title.
        output_path (str or Path): Path to save the plot (if None, plot is displayed).
    """
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot rewards
    plt.plot(rewards, alpha=0.6)
    
    # Plot moving average
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(np.arange(len(moving_avg)) + window-1, moving_avg, 'r-')
        plt.legend(['Episode rewards', f'{window}-episode moving average'])
    else:
        plt.legend(['Episode rewards'])
    
    # Set labels
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Save or display plot
    if output_path is not None:
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Plot saved to {output_path}")
    else:
        plt.show()

def plot_evaluation_comparison(eval_results, labels, metric='mean_reward', title=None, output_path=None):
    """
    Plot comparison of evaluation results.
    
    Args:
        eval_results (list): List of evaluation result dictionaries.
        labels (list): List of labels for each set of results.
        metric (str): Metric to compare ('mean_reward', 'success_rate', etc.).
        title (str): Plot title.
        output_path (str or Path): Path to save the plot (if None, plot is displayed).
    """
    # Extract values
    values = [results[metric] for results in eval_results]
    
    # Create default title if none provided
    if title is None:
        title = f"Comparison of {metric.replace('_', ' ').title()}"
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Create bar plot
    plt.bar(labels, values, alpha=0.7)
    
    # Add values on top of bars
    for i, v in enumerate(values):
        plt.text(i, v, f"{v:.2f}", ha='center', va='bottom')
    
    # Set labels
    plt.xlabel('Agent')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(title)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Save or display plot
    if output_path is not None:
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Plot saved to {output_path}")
    else:
        plt.show()

def plot_learning_curves(metrics, output_dir, prefix="learning_curves"):
    """
    Plot learning curves from training metrics.
    
    Args:
        metrics (dict): Dictionary of training metrics.
        output_dir (str or Path): Output directory.
        prefix (str): Prefix for filenames.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot episode rewards
    if 'episode_rewards' in metrics:
        plot_rewards(
            metrics['episode_rewards'],
            title="Episode Rewards",
            output_path=output_dir / f"{prefix}_rewards.png"
        )
    
    # Plot evaluation rewards
    if 'eval_rewards' in metrics and len(metrics['eval_rewards']) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['eval_rewards'], 'g-', marker='o')
        plt.xlabel('Evaluation')
        plt.ylabel('Average Reward')
        plt.title('Evaluation Rewards')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / f"{prefix}_eval_rewards.png")
        plt.close()
    
    # Plot losses
    if 'losses' in metrics and len(metrics['losses']) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['losses'])
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / f"{prefix}_loss.png")
        plt.close()
    
    # Plot epsilon values
    if 'epsilon_values' in metrics and len(metrics['epsilon_values']) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['epsilon_values'])
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.title('Exploration Rate (Epsilon)')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / f"{prefix}_epsilon.png")
        plt.close()
    
    logger.info(f"Learning curves plotted and saved to {output_dir}")

def plot_reward_distribution(rewards, title="Reward Distribution", output_path=None):
    """
    Plot distribution of rewards.
    
    Args:
        rewards (list): List of rewards.
        title (str): Plot title.
        output_path (str or Path): Path to save the plot (if None, plot is displayed).
    """
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Create distribution plot
    sns.histplot(rewards, kde=True)
    
    # Add statistics lines
    mean_reward = np.mean(rewards)
    median_reward = np.median(rewards)
    
    plt.axvline(mean_reward, color='r', linestyle='--', label=f'Mean: {mean_reward:.2f}')
    plt.axvline(median_reward, color='g', linestyle='--', label=f'Median: {median_reward:.2f}')
    
    # Set labels
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save or display plot
    if output_path is not None:
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Plot saved to {output_path}")
    else:
        plt.show()

def get_device(device=None):
    """
    Get the appropriate device for PyTorch.
    
    Args:
        device (str): Device specification ('cuda', 'cpu', or None for auto-detection).
        
    Returns:
        torch.device: The device to use.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    device = torch.device(device)
    logger.info(f"Using device: {device}")
    
    # Print GPU info if available
    if device.type == 'cuda':
        gpu_count = torch.cuda.device_count()
        current_gpu = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_gpu)
        logger.info(f"GPU: {gpu_name} ({current_gpu+1}/{gpu_count})")
    
    return device

def calculate_success_metrics(rewards, threshold=200.0):
    """
    Calculate success metrics based on rewards.
    
    Args:
        rewards (list): List of episode rewards.
        threshold (float): Success threshold (reward >= threshold is considered success).
        
    Returns:
        dict: Dictionary with success metrics.
    """
    successful_episodes = [r for r in rewards if r >= threshold]
    success_count = len(successful_episodes)
    success_rate = (success_count / len(rewards)) * 100 if rewards else 0
    
    metrics = {
        "success_count": success_count,
        "total_episodes": len(rewards),
        "success_rate": success_rate,
        "mean_success_reward": np.mean(successful_episodes) if successful_episodes else None,
        "threshold": threshold
    }
    
    return metrics

def log_hyperparameters(hyperparams, output_dir, filename="hyperparameters.json"):
    """
    Log hyperparameters to a file.
    
    Args:
        hyperparams (dict): Dictionary of hyperparameters.
        output_dir (str or Path): Output directory.
        filename (str): Output filename.
    """
    output_path = Path(output_dir) / filename
    
    # Convert hyperparameters to JSON-serializable format
    clean_params = {}
    for k, v in hyperparams.items():
        if isinstance(v, (int, float, str, bool, list, dict, type(None))):
            clean_params[k] = v
        else:
            clean_params[k] = str(v)
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(clean_params, f, indent=4)
    
    logger.info(f"Hyperparameters saved to {output_path}")
    
def create_comparison_table(results_list, labels):
    """
    Create a comparison table of results from different agents/experiments.
    
    Args:
        results_list (list): List of dictionaries containing evaluation results.
        labels (list): List of labels for each set of results.
        
    Returns:
        str: Formatted table as a string.
    """
    if len(results_list) != len(labels):
        raise ValueError("Number of results must match number of labels")
    
    # Define metrics to include
    metrics = [
        ("Mean Reward", "mean_reward", ".2f"),
        ("Median Reward", "median_reward", ".2f"),
        ("Min Reward", "min_reward", ".2f"),
        ("Max Reward", "max_reward", ".2f"),
        ("Success Rate", "success_rate", ".2f"),
        ("Average Episode Length", "avg_episode_length", ".2f"),
    ]
    
    # Build header
    header = ["Metric"] + labels
    separator = ["-" * len(h) for h in header]
    
    # Build rows
    rows = []
    for metric_name, metric_key, format_str in metrics:
        # Some metrics might not be present in all results
        row = [metric_name]
        for results in results_list:
            if metric_key in results:
                value = results[metric_key]
                formatted_value = f"{value:{format_str}}" if value is not None else "N/A"
                row.append(formatted_value)
            else:
                row.append("N/A")
        rows.append(row)
    
    # Calculate column widths
    all_rows = [header, separator] + rows
    col_widths = [max(len(row[i]) for row in all_rows) for i in range(len(header))]
    
    # Format table
    table = []
    for i, row in enumerate(all_rows):
        formatted_row = " | ".join(f"{cell:{col_widths[j]}s}" for j, cell in enumerate(row))
        table.append(formatted_row)
        
        # Add separator after header
        if i == 0:
            separator_row = "-+-".join("-" * width for width in col_widths)
            table.append(separator_row)
    
    return "\n".join(table)

if __name__ == "__main__":
    print("This module provides utility functions for the Lunar Lander project.")
    print("It is not intended to be run directly.") 