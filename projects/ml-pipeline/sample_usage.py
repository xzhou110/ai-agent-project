#!/usr/bin/env python
"""
Sample usage of the ML pipeline.

This script demonstrates how to use the ML pipeline for both
classification and clustering tasks programmatically.

Usage:
    python sample_usage.py
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import core modules
try:
    from core.base import create_pipeline
except ImportError:
    logger.warning("Core modules not found. Using flexible_pipeline instead.")
    from flexible_pipeline import run_classification_pipeline, run_clustering_pipeline

def load_data(file_path):
    """
    Load data from a CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to CSV file
        
    Returns:
    --------
    df : pandas.DataFrame
        Loaded data
    """
    logger.info(f"Loading data from {file_path}")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        sys.exit(1)

def create_results_dir():
    """
    Create a timestamped results directory.
    
    Returns:
    --------
    results_dir : str
        Path to results directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_sample_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    logger.info(f"Created results directory: {results_dir}")
    return results_dir

def sample_classification():
    """Run a sample classification pipeline."""
    logger.info("Running sample classification pipeline")
    
    # Load data
    data_path = "data/customer_personality_analysis.csv"
    df = load_data(data_path)
    
    # Create results directory
    results_dir = create_results_dir()
    
    # Check if core modules are available
    if 'create_pipeline' in globals():
        # Use new pipeline architecture
        logger.info("Using new pipeline architecture")
        
        # Create configuration for classification
        config = {
            "data": {
                "path": data_path,
                "target_column": "Response",
                "test_size": 0.2,
                "random_state": 42
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
                "n_features": 10
            },
            "classification": {
                "algorithms": ["logistic_regression", "random_forest", "xgboost"],
                "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"],
                "cv_folds": 5,
                "use_smote": True,
                "tune_hyperparams": True
            },
            "reporting": {
                "save_models": True,
                "generate_visualizations": True,
                "generate_summary": True,
                "output_dir": results_dir,
                "log_level": "INFO"
            },
            "advanced": {
                "n_jobs": -1,
                "batch_size": 1000,
                "seed": 42
            }
        }
        
        # Create and run pipeline
        pipeline = create_pipeline(config=config)
        results = pipeline.run()
        
        # Access results
        best_model = results['models']['best_model']
        evaluation = results['evaluation']
        
        logger.info(f"Best model: {results['models']['best_model_name']}")
        logger.info(f"Evaluation metrics: {evaluation['metrics']}")
    else:
        # Use flexible pipeline
        logger.info("Using flexible pipeline")
        
        # Run classification pipeline
        run_classification_pipeline(
            df=df,
            target_column="Response",
            results_dir=results_dir,
            use_smote=True,
            tune_hyperparams=True,
            feature_selection=True,
            n_features=10
        )
    
    logger.info(f"Classification results saved to {results_dir}")

def sample_clustering():
    """Run a sample clustering pipeline."""
    logger.info("Running sample clustering pipeline")
    
    # Load data
    data_path = "data/customer_personality_analysis.csv"
    df = load_data(data_path)
    
    # Create results directory
    results_dir = create_results_dir()
    
    # Check if core modules are available
    if 'create_pipeline' in globals():
        # Use new pipeline architecture
        logger.info("Using new pipeline architecture")
        
        # Create configuration for clustering
        config = {
            "data": {
                "path": data_path,
                "target_column": None,
                "test_size": 0.0,
                "random_state": 42
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
                "n_clusters_range": [3, 7],
                "metrics": ["silhouette", "calinski_harabasz", "davies_bouldin"]
            },
            "reporting": {
                "save_models": True,
                "generate_visualizations": True,
                "generate_summary": True,
                "output_dir": results_dir,
                "log_level": "INFO"
            },
            "advanced": {
                "n_jobs": -1,
                "batch_size": 1000,
                "seed": 42
            }
        }
        
        # Create and run pipeline
        pipeline = create_pipeline(config=config)
        results = pipeline.run()
        
        # Access results
        best_model = results['models']['best_model']
        labels = results['models']['labels']
        best_method = results['models']['best_method']
        
        logger.info(f"Best clustering method: {best_method}")
        logger.info(f"Number of clusters: {len(np.unique(labels))}")
    else:
        # Use flexible pipeline
        logger.info("Using flexible pipeline")
        
        # Run clustering pipeline
        run_clustering_pipeline(
            df=df,
            results_dir=results_dir,
            n_clusters_range=(3, 7),
            feature_selection=True,
            n_features=15
        )
    
    logger.info(f"Clustering results saved to {results_dir}")

def main():
    """Main function to run the sample pipelines."""
    logger.info("Starting sample usage script")
    
    # Run classification sample
    sample_classification()
    
    # Run clustering sample
    sample_clustering()
    
    logger.info("Sample usage script completed")

if __name__ == "__main__":
    main() 