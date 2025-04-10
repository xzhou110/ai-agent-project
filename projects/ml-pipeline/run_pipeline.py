#!/usr/bin/env python
"""
ML Pipeline Runner

This script serves as the entry point for running the ML pipeline.
It handles command-line arguments, configuration loading,
pipeline creation, and execution with proper error handling.

Usage:
    python run_pipeline.py --config path/to/config.yaml [--debug]
    python run_pipeline.py --task classification --data path/to/data.csv [options]
"""

import os
import sys
import logging
import argparse
import traceback
import time
from datetime import datetime
import yaml
import pandas as pd

# Add project root to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from core.base import create_pipeline, MLPipelineError, ParameterError
except ImportError:
    # Fallback to direct imports if package structure is not available
    print("Core modules not found. Make sure you have the correct directory structure.")
    sys.exit(1)

def setup_logging(log_level="INFO", log_file=None):
    """
    Set up logging configuration.
    
    Parameters:
    -----------
    log_level : str, optional, default: "INFO"
        Logging level
    log_file : str, optional
        Path to log file
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers
    )

def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
    --------
    args : argparse.Namespace
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description="Run ML pipeline")
    
    # Configuration options
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument("--config", type=str, help="Path to YAML configuration file")
    config_group.add_argument("--generate-config", action="store_true", 
                            help="Generate a default configuration file and exit")
    config_group.add_argument("--config-output", type=str, default="config.yaml",
                            help="Path to save generated configuration file")
    
    # Quick-start options
    quickstart_group = parser.add_argument_group("Quick-start")
    quickstart_group.add_argument("--task", type=str, choices=["classification", "clustering"],
                                help="ML task type")
    quickstart_group.add_argument("--data", type=str, help="Path to input data file")
    quickstart_group.add_argument("--target", type=str, help="Target column name (for classification)")
    
    # Output options
    output_group = parser.add_argument_group("Output")
    output_group.add_argument("--output-dir", type=str, help="Directory to save results")
    output_group.add_argument("--log-level", type=str, default="INFO",
                            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                            help="Logging level")
    
    # Debug options
    debug_group = parser.add_argument_group("Debug")
    debug_group.add_argument("--debug", action="store_true", help="Enable debug mode")
    debug_group.add_argument("--profile", action="store_true", 
                            help="Enable performance profiling")
    
    return parser.parse_args()

def generate_default_config(task_type="classification", output_path="config.yaml"):
    """
    Generate a default configuration file.
    
    Parameters:
    -----------
    task_type : str, optional, default: "classification"
        ML task type (classification or clustering)
    output_path : str, optional, default: "config.yaml"
        Path to save configuration file
        
    Returns:
    --------
    config_path : str
        Path to generated configuration file
    """
    # Define default configuration based on task type
    if task_type == "classification":
        default_config = {
            "data": {
                "path": "data/customer_personality_analysis.csv",
                "target_column": "Response",
                "test_size": 0.2,
                "random_state": 42,
                "memory_efficient": True
            },
            "task": {
                "type": "classification"
            },
            "preprocessing": {
                "missing_threshold": 0.9,
                "max_unique_values_cat": 50,
                "correlation_threshold": 0.9,
                "handle_outliers": True,
                "scale_data": True
            },
            "feature_selection": {
                "enabled": True,
                "n_features": None
            },
            "classification": {
                "algorithms": ["logistic_regression", "random_forest", "xgboost", "svm"],
                "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"],
                "cv_folds": 5,
                "use_smote": True,
                "tune_hyperparams": True
            },
            "reporting": {
                "save_models": True,
                "generate_visualizations": True,
                "generate_summary": True,
                "output_dir": None,
                "log_level": "INFO"
            },
            "advanced": {
                "n_jobs": -1,
                "batch_size": 1000,
                "seed": 42
            }
        }
    else:  # clustering
        default_config = {
            "data": {
                "path": "data/customer_personality_analysis.csv",
                "target_column": None,
                "test_size": 0.0,
                "random_state": 42,
                "memory_efficient": True
            },
            "task": {
                "type": "clustering"
            },
            "preprocessing": {
                "missing_threshold": 0.9,
                "max_unique_values_cat": 50,
                "correlation_threshold": 0.9,
                "handle_outliers": True,
                "scale_data": True
            },
            "feature_selection": {
                "enabled": True,
                "n_features": 15
            },
            "clustering": {
                "algorithms": ["kmeans", "hierarchical", "dbscan"],
                "n_clusters_range": [2, 8],
                "metrics": ["silhouette", "calinski_harabasz", "davies_bouldin"]
            },
            "reporting": {
                "save_models": True,
                "generate_visualizations": True,
                "generate_summary": True,
                "output_dir": None,
                "log_level": "INFO"
            },
            "advanced": {
                "n_jobs": -1,
                "batch_size": 1000,
                "seed": 42
            }
        }
    
    # Save configuration to file
    with open(output_path, "w") as f:
        yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Default {task_type} configuration saved to {output_path}")
    return output_path

def create_config_from_args(args):
    """
    Create configuration dictionary from command-line arguments.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
        
    Returns:
    --------
    config : dict
        Configuration dictionary
    """
    config = {
        "data": {
            "path": args.data,
            "target_column": args.target,
            "test_size": 0.2 if args.task == "classification" else 0.0,
            "random_state": 42,
            "memory_efficient": True
        },
        "task": {
            "type": args.task
        },
        "preprocessing": {
            "missing_threshold": 0.9,
            "max_unique_values_cat": 50,
            "correlation_threshold": 0.9,
            "handle_outliers": True,
            "scale_data": True
        },
        "feature_selection": {
            "enabled": True,
            "n_features": None if args.task == "classification" else 15
        },
        "reporting": {
            "save_models": True,
            "generate_visualizations": True,
            "generate_summary": True,
            "output_dir": args.output_dir,
            "log_level": args.log_level
        },
        "advanced": {
            "n_jobs": -1,
            "batch_size": 1000,
            "seed": 42,
            "debug": args.debug
        }
    }
    
    # Add task-specific settings
    if args.task == "classification":
        config["classification"] = {
            "algorithms": ["logistic_regression", "random_forest", "xgboost", "svm"],
            "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"],
            "cv_folds": 5,
            "use_smote": True,
            "tune_hyperparams": True
        }
    else:  # clustering
        config["clustering"] = {
            "algorithms": ["kmeans", "hierarchical", "dbscan"],
            "n_clusters_range": [2, 8],
            "metrics": ["silhouette", "calinski_harabasz", "davies_bouldin"]
        }
    
    return config

def profile_execution(func, *args, **kwargs):
    """
    Profile execution of a function.
    
    Parameters:
    -----------
    func : callable
        Function to profile
    *args, **kwargs
        Arguments to pass to function
        
    Returns:
    --------
    result
        Result of function execution
    """
    try:
        from cProfile import Profile
        from pstats import Stats
        profiler = Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        stats = Stats(profiler)
        stats.strip_dirs()
        stats.sort_stats('cumulative')
        stats.print_stats(20)
        return result
    except ImportError:
        print("Profiling requires the cProfile module. Running without profiling.")
        return func(*args, **kwargs)

def main():
    """Main function to run the pipeline."""
    start_time = time.time()
    
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Generate default configuration if requested
        if args.generate_config:
            task_type = args.task if args.task else "classification"
            generate_default_config(task_type, args.config_output)
            return 0
        
        # Get configuration
        if args.config:
            # Load configuration from file
            if not os.path.exists(args.config):
                logger.error(f"Configuration file not found: {args.config}")
                return 1
            
            try:
                with open(args.config, "r") as f:
                    config = yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Error loading configuration: {str(e)}")
                return 1
        elif args.task and args.data:
            # Create configuration from command-line arguments
            config = create_config_from_args(args)
        else:
            logger.error("Either --config or --task and --data must be provided")
            return 1
        
        # Add debug flag to config
        config["debug"] = args.debug
        
        # Create and run pipeline
        logger.info("Creating pipeline")
        pipeline = create_pipeline(config=config)
        
        if args.profile:
            logger.info("Running pipeline with profiling")
            results = profile_execution(pipeline.run)
        else:
            logger.info("Running pipeline")
            results = pipeline.run()
        
        # Log completion
        elapsed_time = time.time() - start_time
        logger.info(f"Pipeline completed successfully in {elapsed_time:.2f} seconds")
        
        # Print output directory
        logger.info(f"Results saved to {pipeline.output_dir}")
        
        return 0
    
    except ParameterError as e:
        logger.error(f"Parameter error: {str(e)}")
        if args.debug:
            logger.error(traceback.format_exc())
        return 1
    
    except MLPipelineError as e:
        logger.error(f"Pipeline error: {str(e)}")
        if args.debug:
            logger.error(traceback.format_exc())
        return 1
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        if args.debug:
            logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 