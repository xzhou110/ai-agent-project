#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
from typing import List, Tuple, Dict, Optional, Union
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define experience replay memory
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """Experience replay buffer to store and sample agent experiences."""
    
    def __init__(self, capacity: int = 100000):
        """
        Initialize replay buffer with fixed capacity.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def add(self, state, action, reward, next_state, done):
        """
        Add an experience to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Tuple:
        """
        Sample a random batch of experiences from the buffer.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        experiences = random.sample(self.buffer, batch_size)
        
        # Convert to separate arrays for batch processing
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float()
            
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough experiences for sampling."""
        return len(self) >= batch_size


class QNetwork(nn.Module):
    """Neural network model to approximate Q-values."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: List[int] = [128, 64]):
        """
        Initialize Q-Network.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_sizes: List of hidden layer sizes
        """
        super(QNetwork, self).__init__()
        
        # Build the network layers
        layers = []
        input_dim = state_dim
        
        # Add hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size
        
        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))
        
        # Combine all layers into sequential model
        self.model = nn.Sequential(*layers)
    
    def forward(self, state):
        """
        Forward pass through the network to predict Q-values.
        
        Args:
            state: Current state
            
        Returns:
            Q-values for each action
        """
        return self.model(state)


class DQNAgent:
    """Deep Q-Network agent."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: List[int] = [128, 64],
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        device: str = None
    ):
        """
        Initialize DQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_sizes: List of hidden layer sizes
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Starting value for exploration rate
            epsilon_end: Minimum value for exploration rate
            epsilon_decay: Decay rate for exploration
            buffer_size: Size of replay buffer
            batch_size: Batch size for training
            target_update_freq: Frequency to update target network
            device: Device to run model on (cpu or cuda)
        """
        # Set device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Agent parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.learning_rate = learning_rate
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Training parameters
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_counter = 0
        
        # Create Q-networks
        self.q_network = QNetwork(state_dim, action_dim, hidden_sizes).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_sizes).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Set target network to evaluation mode
        
        # Create optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Create replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training metrics
        self.losses = []
    
    def select_action(self, state, epsilon=None, evaluate=False, return_q_values=False):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            epsilon: Override exploration rate (for evaluation)
            evaluate: If True, use greedy policy (no exploration)
            return_q_values: If True, return Q-values along with action
            
        Returns:
            Selected action and optionally Q-values
        """
        # Use provided epsilon or class epsilon, or 0 if evaluate is True
        eps = 0.0 if evaluate else (epsilon if epsilon is not None else self.epsilon)
        
        # Epsilon-greedy action selection
        if np.random.random() < eps:
            # Random action
            action = np.random.randint(self.action_dim)
            if return_q_values:
                # If we need Q-values but took a random action, compute them anyway
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    q_values = self.q_network(state_tensor).cpu().numpy()[0]
                return action, q_values
            return action
        else:
            # Greedy action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor).cpu().numpy()[0]
            action = q_values.argmax().item()
            
            if return_q_values:
                return action, q_values
            return action
    
    def store_experience(self, state, action, reward, next_state, done):
        """
        Store experience in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Convert action to array for the buffer
        action_array = np.array([[action]])
        
        # Add experience to replay buffer
        self.replay_buffer.add(state, action_array, reward, next_state, done)
    
    def can_learn(self):
        """Check if enough experiences are available for learning."""
        return self.replay_buffer.is_ready(self.batch_size)
    
    def step(self, state, action, reward, next_state, done):
        """
        Take a step in the environment and learn from it.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Add experience to replay buffer
        self.store_experience(state, action, reward, next_state, done)
        
        # Learn from experiences if buffer is ready
        if self.can_learn():
            return self.learn()
        
        return None

    def learn(self):
        """Update Q-network parameters using a batch of experiences."""
        # Sample a batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Transfer to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Get current Q values for taken actions
        q_values = self.q_network(states).gather(1, actions)
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Compute loss
        loss = F.mse_loss(q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1)
        self.optimizer.step()
        
        # Record loss
        self.losses.append(loss.item())
        
        # Update target network if it's time
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def update_epsilon(self):
        """Decay epsilon value for exploration."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, path: str):
        """
        Save agent state to disk.
        
        Args:
            path: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model and agent parameters
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'losses': self.losses,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'gamma': self.gamma,
            'learning_rate': self.learning_rate,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay,
            'batch_size': self.batch_size,
            'target_update_freq': self.target_update_freq,
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """
        Load agent state from disk.
        
        Args:
            path: Path to load the model from
        """
        # Check if file exists
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No model found at {path}")
        
        # Load checkpoint
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model parameters
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load agent parameters
        self.epsilon = checkpoint['epsilon']
        self.losses = checkpoint['losses']
        
        # Set target network to evaluation mode
        self.target_network.eval()
        
        logger.info(f"Model loaded from {path}")

    def update_target_network(self):
        """Update target network by copying parameters from online network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
        logger.info("Target network updated")


class DoubleDQNAgent(DQNAgent):
    """Double DQN Agent, extending the base DQN agent with double Q-learning."""
    
    def __init__(self, state_dim: int, action_dim: int, **kwargs):
        """
        Initialize Double DQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            **kwargs: Additional arguments to pass to DQNAgent
        """
        super(DoubleDQNAgent, self).__init__(state_dim, action_dim, **kwargs)
        logger.info("Initialized Double DQN Agent")
    
    def learn(self):
        """Update Q-network parameters using Double DQN algorithm."""
        # Sample a batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Transfer to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Get current Q values for taken actions
        q_values = self.q_network(states).gather(1, actions)
        
        # Compute target Q values using Double DQN
        with torch.no_grad():
            # Select actions using the online network
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            
            # Evaluate those actions using the target network
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            
            # Compute target Q values
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Compute loss
        loss = F.mse_loss(q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1)
        self.optimizer.step()
        
        # Record loss
        self.losses.append(loss.item())
        
        # Update target network if it's time
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict()) 