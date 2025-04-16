#!/usr/bin/env python3
import os
import gymnasium as gym
import numpy as np
import pygame
import time
import matplotlib.pyplot as plt
from matplotlib import animation
from typing import Tuple, Dict, Any, Optional, List
import logging
from PIL import Image
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/environment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Environment")

class LunarLanderEnvironment:
    """
    A wrapper for the Lunar Lander environment from OpenAI Gym.
    Provides a consistent interface for the environment and handles rendering,
    recording videos, and environment resets.
    """
    
    def __init__(
        self,
        continuous=False,
        render_mode=None,
        seed=None,
        gravity=None,
        enable_wind=False,
        wind_power=None,
        turbulence_power=None
    ):
        """
        Initialize the Lunar Lander environment.
        
        Args:
            continuous (bool): Whether to use continuous (True) or discrete (False) action space.
            render_mode (str): Rendering mode (None, 'human', 'rgb_array').
            seed (int): Random seed for reproducibility.
            gravity (float): Gravity strength (default: -10.0).
            enable_wind (bool): Whether to enable wind effects.
            wind_power (float): Strength of the wind (default: 15.0).
            turbulence_power (float): Strength of turbulence (default: 1.5).
        """
        self.continuous = continuous
        self.render_mode = render_mode
        self.seed = seed
        
        # Create the Gym environment
        env_id = "LunarLander-v2"
        if continuous:
            env_id = "LunarLanderContinuous-v2"
        
        logger.info(f"Creating {env_id} environment with render_mode: {render_mode}")
        
        # Initialize environment with custom parameters if provided
        env_kwargs = {
            "render_mode": render_mode,
        }
        
        if seed is not None:
            env_kwargs["seed"] = seed
        
        if gravity is not None or enable_wind or wind_power is not None or turbulence_power is not None:
            # Only add these kwargs if they are explicitly set
            if gravity is not None:
                env_kwargs["gravity"] = gravity
            if enable_wind:
                env_kwargs["enable_wind"] = enable_wind
            if wind_power is not None:
                env_kwargs["wind_power"] = wind_power
            if turbulence_power is not None:
                env_kwargs["turbulence_power"] = turbulence_power
        
        self.env = gym.make(env_id, **env_kwargs)
        
        # Initialize attributes
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        # Get dimensions for agents
        self.state_dim = self.observation_space.shape[0]
        
        if continuous:
            self.action_dim = self.action_space.shape[0]
        else:
            self.action_dim = self.action_space.n
        
        logger.info(f"State dimension: {self.state_dim}, Action dimension: {self.action_dim}")
        
        # Tracking variables
        self.current_episode = 0
        self.current_step = 0
        self.total_reward = 0
        self.is_recording = False
        self.recording_env = None
        
    def reset(self):
        """
        Reset the environment to start a new episode.
        
        Returns:
            observation (numpy.ndarray): Initial observation from the environment.
        """
        self.current_step = 0
        self.total_reward = 0
        
        observation, info = self.env.reset(seed=self.seed)
        return observation
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: Action to take in the environment.
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        self.current_step += 1
        self.total_reward += reward
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """
        Render the current environment state.
        Only works if render_mode was set during initialization.
        
        Returns:
            Rendered frame (depends on render_mode).
        """
        if self.render_mode is not None:
            return self.env.render()
        return None
    
    def close(self):
        """Close the environment."""
        if self.recording_env is not None:
            self.recording_env.close()
            self.recording_env = None
            self.is_recording = False
        
        self.env.close()
        
    def enable_video_recording(self, video_folder="videos", episode_trigger=None):
        """
        Enable video recording of episodes.
        
        Args:
            video_folder (str): Directory to save videos.
            episode_trigger (callable): Function that takes the episode index as input
                and returns a boolean indicating whether to record the episode.
        """
        if self.is_recording:
            logger.warning("Video recording is already enabled.")
            return
        
        # Create video directory if it doesn't exist
        os.makedirs(video_folder, exist_ok=True)
        
        # If no episode trigger is provided, record all episodes
        if episode_trigger is None:
            episode_trigger = lambda episode_id: True
        
        # Create a new environment wrapped with RecordVideo
        self.recording_env = RecordVideo(
            self.env,
            video_folder=video_folder,
            episode_trigger=episode_trigger,
            name_prefix="lunar_lander"
        )
        
        # Replace the current environment with the recording one
        self.env = self.recording_env
        self.is_recording = True
        
        logger.info(f"Video recording enabled. Videos will be saved to {video_folder}")
    
    def disable_video_recording(self):
        """Disable video recording."""
        if not self.is_recording:
            logger.warning("Video recording is not enabled.")
            return
        
        # Close the recording environment
        self.recording_env.close()
        
        # Recreate the original environment
        env_id = "LunarLander-v2"
        if self.continuous:
            env_id = "LunarLanderContinuous-v2"
        
        self.env = gym.make(
            env_id,
            render_mode=self.render_mode,
            seed=self.seed
        )
        
        self.recording_env = None
        self.is_recording = False
        
        logger.info("Video recording disabled.")
    
    def get_success_metrics(self, rewards, threshold=200.0):
        """
        Calculate success rate based on rewards.
        In Lunar Lander, a reward of 200+ is considered a successful landing.
        
        Args:
            rewards (list): List of episode rewards.
            threshold (float): Success threshold.
            
        Returns:
            dict: Dictionary with success metrics.
        """
        success_count = sum(1 for r in rewards if r >= threshold)
        success_rate = (success_count / len(rewards)) * 100 if rewards else 0
        
        return {
            "success_count": success_count,
            "success_rate": success_rate,
            "total_episodes": len(rewards)
        }
    
    def __str__(self):
        """String representation of the environment."""
        return f"LunarLanderEnvironment(continuous={self.continuous}, render_mode={self.render_mode})"
    
    def get_env_info(self) -> Dict[str, Any]:
        """
        Get information about the environment.
        
        Returns:
            A dictionary containing environment information.
        """
        return {
            "env_name": self.env.spec.id,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "continuous": self.continuous,
            "observation_space": self.observation_space,
            "action_space": self.action_space
        }
    
    def seed(self, seed: int) -> None:
        """
        Set the random seed for the environment.
        
        Args:
            seed: The random seed.
        """
        self.env.seed(seed)
        np.random.seed(seed)
    
    def get_state_description(self, state: np.ndarray) -> Dict[str, float]:
        """
        Convert the state vector to a more interpretable dictionary.
        
        Args:
            state: The state vector
            
        Returns:
            A dictionary mapping state components to their values
        """
        return {
            "x_position": state[0],
            "y_position": state[1],
            "x_velocity": state[2],
            "y_velocity": state[3],
            "angle": state[4],
            "angular_velocity": state[5],
            "left_leg_contact": state[6],
            "right_leg_contact": state[7]
        }
    
    def human_readable_action(self, action: int) -> str:
        """
        Convert an action index to a human-readable string.
        
        Args:
            action: The action index
            
        Returns:
            A string description of the action
        """
        actions = {
            0: "Do nothing",
            1: "Fire left engine",
            2: "Fire main engine",
            3: "Fire right engine"
        }
        return actions.get(action, "Unknown action")
    
    def evaluate_performance(self, rewards: List[float]) -> Dict[str, float]:
        """
        Evaluate the performance based on episode rewards.
        
        Args:
            rewards: List of episode rewards
            
        Returns:
            A dictionary of performance metrics
        """
        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
            "median_reward": np.median(rewards)
        }
    
    def save_frames_as_gif(self, frames: List[np.ndarray], filename: str) -> None:
        """
        Save a list of frames as a GIF.
        
        Args:
            frames: List of RGB arrays
            filename: Output filename
        """
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # Convert frames to PIL images
            pil_images = [Image.fromarray(frame) for frame in frames]
            
            # Save as GIF
            pil_images[0].save(
                filename,
                save_all=True,
                append_images=pil_images[1:],
                duration=50,  # milliseconds between frames
                loop=0  # 0 means loop forever
            )
            logger.info(f"Saved {len(frames)} frames as GIF to {filename}")
        except Exception as e:
            logger.error(f"Error saving GIF: {e}")
    
    @staticmethod
    def save_frames_as_mp4(frames: List[np.ndarray], filename: str, fps: int = 30) -> None:
        """
        Save frames as an MP4 video using matplotlib animation.
        
        Args:
            frames: List of RGB arrays
            filename: Output filename
            fps: Frames per second
        """
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # Create figure and axes
            plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
            plt.axis('off')
            
            # Create animation
            patch = plt.imshow(frames[0])
            plt.tight_layout()
            
            def animate(i):
                patch.set_data(frames[i])
                return [patch]
            
            anim = animation.FuncAnimation(
                plt.gcf(), animate, frames=len(frames),
                interval=1000/fps, blit=True
            )
            
            # Save as MP4
            writer = animation.FFMpegWriter(fps=fps)
            anim.save(filename, writer=writer)
            
            plt.close()
            logger.info(f"Saved {len(frames)} frames as MP4 to {filename}")
        except Exception as e:
            logger.error(f"Error saving MP4: {e}")
    
    def record_episode(self, model, max_steps: int = 1000) -> Tuple[float, List[np.ndarray]]:
        """
        Record an episode using the given model.
        
        Args:
            model: The RL agent/model
            max_steps: Maximum number of steps
            
        Returns:
            total_reward: The total reward obtained
            frames: List of frames for visualization
        """
        frames = []
        observation = self.reset()
        total_reward = 0
        done = False
        truncated = False
        
        # Make Lunar Lander environment with rgb_array render mode for recording
        temp_env = gym.make("LunarLander-v2", render_mode='rgb_array')
        observation, _ = temp_env.reset()
        
        for _ in range(max_steps):
            frames.append(temp_env.render())
            action = model.predict(observation)
            observation, reward, done, truncated, _ = temp_env.step(action)
            total_reward += reward
            
            if done or truncated:
                frames.append(temp_env.render())
                break
                
        temp_env.close()
        return total_reward, frames 