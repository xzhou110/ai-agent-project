import pandas as pd
import numpy as np
import os
import logging
import argparse
import time
from datetime import datetime
from tools.data_preprocessing import preprocess_data
from tools.feature_selection import select_features

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_classification_pipeline(df, target_column, results_dir, 
                              use_smote=True, tune_hyperparams=True, 
                              feature_selection=True, n_features=None):
    """
    Run the classification pipeline on the given dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataset
    target_column : str
        The name of the target column for classification
    results_dir : str
        Directory to save results
    use_smote : bool, optional, default: True
        Whether to use SMOTE for handling class imbalance
    tune_hyperparams : bool, optional, default: True
        Whether to tune hyperparameters of the models
    feature_selection : bool, optional, default: True
        Whether to perform feature selection
    n_features : int, optional, default: None
        Number of features to select if feature_selection is True
    """
    logging.info("Starting classification pipeline...")
    
    # Preprocess the data
    logging.info("Preprocessing data...")
    preprocessed_df = preprocess_data(
        df=df,
        label=target_column,
        missing_threshold=0.9,
        max_unique_values_cat=50,
        correlation_threshold=0.9
    )
    
    # Save preprocessed data
    preprocessed_df.to_csv(os.path.join(results_dir, 'preprocessed_data.csv'), index=False)
    
    # Extract features and target
    X = preprocessed_df.drop(columns=[target_column])
    y = preprocessed_df[target_column]
    
    # Perform feature selection if requested
    if feature_selection:
        logging.info("Selecting important features...")
        X_selected = select_features(
            X=X,
            y=y,
            problem_type='classification',
            n_features_to_select=n_features,
            downsample=True,
            log=True,
            plot=True,
            results_dir=results_dir
        )
    else:
        X_selected = X
    
    # Import classification models here to avoid circular imports
    from tools.classification_models import run_classification_models
    
    # Train and evaluate classification models
    logging.info("Training and evaluating classification models...")
    best_model_name, best_model, best_score = run_classification_models(
        X=X_selected,
        y=y,
        use_smote=use_smote,
        tune_hyperparams=tune_hyperparams,
        plot_roc=True,
        results_dir=results_dir
    )
    
    logging.info(f"Best model: {best_model_name} with score: {best_score}")
    
    # Generate summary report
    with open(os.path.join(results_dir, 'classification_summary.txt'), 'w') as f:
        f.write(f"Classification Pipeline Summary\n")
        f.write(f"==============================\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Target variable: {target_column}\n")
        f.write(f"Original data shape: {df.shape}\n")
        f.write(f"Preprocessed data shape: {preprocessed_df.shape}\n")
        f.write(f"Number of selected features: {X_selected.shape[1]}\n")
        f.write(f"Selected features: {', '.join(X_selected.columns)}\n\n")
        f.write(f"Best model: {best_model_name}\n")
        f.write(f"Best model score: {best_score}\n")
    
    logging.info(f"Classification pipeline complete! Results saved to {results_dir}/")

def run_clustering_pipeline(df, results_dir, n_clusters_range=(2, 8),
                          feature_selection=True, n_features=15):
    """
    Run the clustering pipeline on the given dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataset
    results_dir : str
        Directory to save results
    n_clusters_range : tuple, optional, default: (2, 8)
        Range of number of clusters to try (min, max)
    feature_selection : bool, optional, default: True
        Whether to perform feature selection
    n_features : int, optional, default: 15
        Number of features to select if feature_selection is True
    """
    logging.info("Starting clustering pipeline...")
    
    # Preprocess the data (no target for clustering)
    logging.info("Preprocessing data...")
    preprocessed_df = preprocess_data(
        df=df,
        label=None,  # No label for clustering
        missing_threshold=0.9,
        max_unique_values_cat=50,
        correlation_threshold=0.9
    )
    
    # Save preprocessed data
    preprocessed_df.to_csv(os.path.join(results_dir, 'preprocessed_data.csv'), index=False)
    
    # Perform feature selection if requested
    if feature_selection:
        logging.info("Selecting important features for clustering...")
        X_selected = select_features(
            X=preprocessed_df,
            y=None,  # No target for unsupervised learning
            problem_type='clustering',
            n_features_to_select=n_features,
            downsample=False,
            log=True,
            plot=True,
            results_dir=results_dir
        )
    else:
        X_selected = preprocessed_df
    
    # Import clustering models here to avoid circular imports
    from tools.clustering_models import customer_segmentation_analysis
    
    # Perform customer segmentation
    logging.info("Running clustering models for customer segmentation...")
    best_model, labels, best_method, best_n_clusters = customer_segmentation_analysis(
        df=X_selected,  # Use selected features
        n_clusters_range=n_clusters_range,
        use_original_features=True,  # Already selected features
        plot=True,
        results_dir=results_dir
    )
    
    # Save results and provide summary
    logging.info(f"Customer segmentation completed.")
    logging.info(f"Best clustering method: {best_method}")
    logging.info(f"Optimal number of clusters: {best_n_clusters}")
    
    # Add cluster labels to original data and save
    df['Cluster'] = labels
    df.to_csv(os.path.join(results_dir, 'data_with_clusters.csv'), index=False)
    
    # Generate summary report
    with open(os.path.join(results_dir, 'clustering_summary.txt'), 'w') as f:
        f.write(f"Clustering Pipeline Summary\n")
        f.write(f"=========================\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Original data shape: {df.shape}\n")
        f.write(f"Preprocessed data shape: {preprocessed_df.shape}\n")
        f.write(f"Number of selected features: {X_selected.shape[1]}\n")
        f.write(f"Selected features: {', '.join(X_selected.columns)}\n\n")
        f.write(f"Best clustering method: {best_method}\n")
        f.write(f"Optimal number of clusters: {best_n_clusters}\n")
        f.write(f"Cluster distribution:\n")
        
        # Count clusters
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        for cluster, count in cluster_counts.items():
            f.write(f"  Cluster {cluster}: {count} samples ({count/len(df)*100:.1f}%)\n")
    
    logging.info(f"Clustering pipeline complete! Results saved to {results_dir}/")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run ML pipeline for classification or clustering')
    
    parser.add_argument('--data', type=str, required=True, 
                      help='Path to the input CSV data file')
    
    parser.add_argument('--task', type=str, required=True, choices=['classification', 'clustering'],
                      help='ML task type: classification or clustering')
    
    parser.add_argument('--target', type=str, default=None,
                      help='Target column name (required for classification)')
    
    parser.add_argument('--results-dir', type=str, default=None,
                      help='Directory to save results (if not specified, a timestamped directory will be created)')
    
    parser.add_argument('--min-clusters', type=int, default=2,
                      help='Minimum number of clusters to try (for clustering)')
    
    parser.add_argument('--max-clusters', type=int, default=8,
                      help='Maximum number of clusters to try (for clustering)')
    
    parser.add_argument('--feature-selection', action='store_true', default=True,
                      help='Whether to perform feature selection')
    
    parser.add_argument('--no-feature-selection', action='store_false', dest='feature_selection',
                      help='Skip feature selection')
    
    parser.add_argument('--n-features', type=int, default=None,
                      help='Number of features to select (if feature selection is enabled)')
    
    parser.add_argument('--use-smote', action='store_true', default=True,
                      help='Use SMOTE for handling class imbalance in classification')
    
    parser.add_argument('--no-smote', action='store_false', dest='use_smote',
                      help='Do not use SMOTE for classification')
    
    parser.add_argument('--tune-hyperparams', action='store_true', default=True,
                      help='Tune hyperparameters for classification models')
    
    parser.add_argument('--no-tune-hyperparams', action='store_false', dest='tune_hyperparams',
                      help='Skip hyperparameter tuning for classification models')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.task == 'classification' and args.target is None:
        parser.error("--target is required for classification tasks")
    
    # Create results directory
    if args.results_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"results_{args.task}_{timestamp}"
    else:
        results_dir = args.results_dir
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Load the data
    logging.info(f"Loading data from {args.data}...")
    try:
        df = pd.read_csv(args.data, sep='\t')
        logging.info(f"Dataset shape: {df.shape}")
        logging.info(f"Columns: {df.columns.tolist()}")
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return
    
    # Run the appropriate pipeline
    if args.task == 'classification':
        # Verify target column exists
        if args.target not in df.columns:
            logging.error(f"Target column '{args.target}' not found in dataset. Available columns: {df.columns.tolist()}")
            return
            
        # Run classification pipeline
        run_classification_pipeline(
            df=df,
            target_column=args.target,
            results_dir=results_dir,
            use_smote=args.use_smote,
            tune_hyperparams=args.tune_hyperparams,
            feature_selection=args.feature_selection,
            n_features=args.n_features
        )
    else:  # clustering
        # Run clustering pipeline
        run_clustering_pipeline(
            df=df,
            results_dir=results_dir,
            n_clusters_range=(args.min_clusters, args.max_clusters),
            feature_selection=args.feature_selection,
            n_features=args.n_features if args.n_features else 15
        )

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    logging.info(f"Total execution time: {elapsed_time:.2f} seconds") 