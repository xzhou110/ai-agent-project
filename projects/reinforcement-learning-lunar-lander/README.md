# Reinforcement Learning for Lunar Lander

This project implements and compares different reinforcement learning algorithms to solve the Lunar Lander environment from OpenAI's Gym. We focus on implementing both the standard Deep Q-Network (DQN) algorithm and its extension, Double DQN.

![Lunar Lander Environment](https://gymnasium.farama.org/_images/lunar_lander.gif)

## Project Status

This is an early learning project and is not actively maintained. Training with `python main.py train` works and has been run on the real Lunar Lander environment for both agents. The rest is unfinished and currently fails: the `evaluate` and `compare` modes, running `train.py` directly, `visualize.py`, and the `--continuous` option (these agents choose from a fixed list of actions, which the continuous environment does not accept).

## Project Overview

The Lunar Lander task is a classic reinforcement learning problem where an agent must learn to safely land a lunar module on a landing pad. The environment provides a reward function that incentivizes:
- Landing on the landing pad (+100 to +140 points)
- Moving the lander to the landing pad and stopping
- Penalizing crashing, firing the main engine, and flying away

The agent must learn to control the lander by applying discrete or continuous actions to navigate to the landing pad and land softly.

## Features

- Implementation of DQN and Double DQN algorithms
- Support for both discrete and continuous action spaces
- Comprehensive training and evaluation pipelines
- Visualization tools for analyzing agent performance
- Comparison framework for benchmarking different algorithms
- Recording capabilities for generating videos of trained agents

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>/projects/reinforcement-learning-lunar-lander
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
reinforcement-learning-lunar-lander/
├── agent.py              # Implementation of DQN and Double DQN agents
├── environment.py        # Wrapper for the Lunar Lander environment
├── main.py               # Main entry point for training and evaluation
├── train.py              # Training functions and logic
├── evaluate.py           # Evaluation functions and metrics
├── utils.py              # Utility functions for the project
├── visualize.py          # Visualization tools and functions
├── requirements.txt      # Required dependencies
├── README.md             # Project documentation
├── data/                 # Directory for storing datasets
├── models/               # Directory for storing trained models
├── results/              # Directory for storing results
│   ├── plots/            # Saved plots and visualizations
│   └── videos/           # Saved videos of agent performance
└── logs/                 # Training and evaluation logs
```

## Usage

### Training

To train a DQN agent on the Lunar Lander environment:

```bash
python main.py train --agent dqn --episodes 1000
```

For Double DQN:

```bash
python main.py train --agent double_dqn --episodes 1000
```

Additional options, written after `train`:
- `--lr`: Learning rate for the optimizer (default: 0.001)
- `--gamma`: Discount factor (default: 0.99)
- `--epsilon_start`: Initial exploration rate (default: 1.0)
- `--epsilon_end`: Final exploration rate (default: 0.01)
- `--epsilon_decay`: Decay rate for epsilon (default: 0.995)
- `--hidden_dims`: Sizes of the hidden layers (default: 128 128)
- `--memory_size`: Size of the replay buffer (default: 100000)
- `--batch_size`: Batch size for training (default: 64)
- `--target_update`: Learning steps between target network updates (default: 10)
- `--max_steps`: Maximum steps per episode (default: 1000)
- `--eval_interval`: Episodes between evaluations (default: 50)
- `--render_train`: Render the environment during training
- `--output_dir`: Folder for the models, plots and metrics (default: `results/<agent>_<timestamp>`)

Two options apply to every mode and are written before `train`:
- `--seed`: Random seed for reproducibility (default: 42)
- `--device`: Device to use, `auto`, `cpu` or `cuda` (default: `auto`)

```bash
python main.py --seed 7 --device cpu train --agent dqn --episodes 1000
```

### Evaluation

To evaluate a trained agent:

```bash
python main.py --mode evaluate --agent dqn --model models/dqn_final_model.pt --episodes 100
```

Additional evaluation options:
- `--save-video`: Save videos of the agent's performance
- `--render`: Render the environment during evaluation

### Comparing Agents

To compare different algorithms:

```bash
python main.py --mode compare --agents dqn double_dqn --episodes 1000
```

### Visualizing Results

To visualize training results:

```bash
python visualize.py --rewards results/dqn_20230101-120000/episode_rewards.npy --output results/visualizations
```

To compare multiple agents:

```bash
python visualize.py --compare --agent-files results/dqn_*/episode_rewards.npy results/double_dqn_*/episode_rewards.npy --agent-names "DQN" "Double DQN" --output results/comparisons
```

## Algorithms

### Deep Q-Network (DQN)

DQN combines Q-learning with deep neural networks to learn value functions in complex environments. Key components:

1. **Neural Network Function Approximation**: Replaces the traditional Q-table with a neural network.
2. **Experience Replay**: Stores experiences in a replay buffer and randomly samples batches for learning.
3. **Target Network**: Uses a separate target network for generating TD targets to improve stability.

### Double DQN

Double DQN addresses the overestimation bias in standard DQN by using:

1. **Decoupled Selection and Evaluation**: Uses the online network to select actions and the target network to evaluate them.
2. **Reduced Overestimation**: This leads to more accurate Q-value estimates and more stable learning.

## Results

The project compares DQN and Double DQN on the Lunar Lander environment. Metrics include:

- Learning curves showing average rewards over episodes
- Success rates (percentage of successful landings)
- Episode lengths
- Reward distributions
- Visual demonstrations of trained agents

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- OpenAI Gym/Gymnasium for the Lunar Lander environment
- The Deep Q-Learning paper by Mnih et al. (2015)
- The Double DQN paper by van Hasselt et al. (2016) 