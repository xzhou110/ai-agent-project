# AI-Agent-Projects

This repository is a hands-on exploration of building end-to-end projects with the assistance of an AI agent. It includes a refactored machine learning pipeline and a reinforcement learning project, each demonstrating practical approaches to problem-solving with modern AI techniques.

## Overview

By harnessing AI agents to streamline workflows, this project aims to show how such tools can assist with code refactoring, model optimization, and experimentation. From data preprocessing to training RL agents, each section highlights different use cases where an AI agent’s input can expedite development and enhance maintainability.

## Repository Structure

- **projects/**
  - **ml-pipeline/**  
    A refactored machine learning pipeline demonstrating how AI-driven code improvements can benefit data handling, model building, and evaluation.
  - **reinforcement-learning-lunar-lander/**  
    An introductory reinforcement learning project that showcases training an agent to complete landings successfully.
- **tests/**  
  Contains tests to verify code functionality and maintain reliability.
- **utils/**  
  Shared helper scripts and utility functions.
- **.cursorrules**  
  Configuration for certain development environments.
- **.gitignore**  
  Specifies files and directories to be excluded from version control.
- **requirements.txt**  
  Lists Python dependencies required for both sub-projects.
- **LICENSE**  
  Specifies the terms and conditions for using and distributing this project.

## Getting Started

1. **Clone the Repository**
   ```bash
   git clone https://github.com/xzhou110/AI-Agent-Projects.git
   cd AI-Agent-Projects
   ```
2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Explore the Projects**
   - **ml-pipeline/**: Investigate how the pipeline is structured, paying attention to data transforms, model training scripts, and evaluation metrics.
   - **reinforcement-learning-lunar-lander/**: Try running the agent in a simulation environment to see how it learns over time.

## Machine Learning Pipeline

Inside `ml-pipeline/`, you’ll find scripts or notebooks for:
- Data preparation (cleaning and feature engineering).
- Model training with various algorithms.
- Evaluation and potential deployment strategies.

Refactoring guidance by an AI agent helped create a more modular architecture, making it easier to add new steps and tweak configurations.

## Reinforcement Learning: Lunar Lander

In `reinforcement-learning-lunar-lander/`, you’ll see:
- Environment setup with OpenAI Gym.
- An approach for training an agent (e.g., DQN, PPO).
- Logs and performance metrics to track learning progress.

Small adjustments in hyperparameters can significantly impact the agent’s performance, making this a great playground for quick iteration and testing.

## Contributing

Pull requests are welcome. If you have ideas to improve the pipeline or expand RL experiments, feel free to create a new branch and open a PR. Suggestions for additional features or refactoring are also appreciated.

## License

All code in this repository is released under the [LICENSE](LICENSE) provided. Feel free to modify and use it within the terms of the license.

Enjoy exploring the projects, and have fun experimenting with AI agents in your own workflows!
