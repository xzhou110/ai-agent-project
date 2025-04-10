import pandas as pd
import os
import logging
from tools.data_preprocessing import preprocess_data
from tools.feature_selection import select_features
from tools.clustering_models import customer_segmentation_analysis

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Create results directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load the data
    logging.info("Loading data...")
    data_path = "data/customer_personality_analysis.csv"
    df = pd.read_csv(data_path)
    
    # Display basic information about the dataset
    logging.info(f"Dataset shape: {df.shape}")
    logging.info(f"Columns: {df.columns.tolist()}")
    
    # Preprocess the data
    logging.info("Preprocessing data...")
    # No label column since it's a clustering problem
    preprocessed_df = preprocess_data(
        df=df,
        label=None,  # No label for clustering
        missing_threshold=0.9,
        max_unique_values_cat=50,
        correlation_threshold=0.9
    )
    
    # Select important features
    logging.info("Selecting important features...")
    # For clustering, we use recursive feature elimination without a specific target
    X_selected = select_features(
        X=preprocessed_df,
        y=None,  # No target for unsupervised learning
        problem_type='clustering',
        n_features_to_select=15,  # Select top 15 features
        downsample=False,
        log=True,
        plot=True,
        results_dir=results_dir
    )
    
    # Perform customer segmentation
    logging.info("Running clustering models for customer segmentation...")
    best_model, labels, best_method, best_n_clusters = customer_segmentation_analysis(
        df=preprocessed_df,  # Use preprocessed data
        n_clusters_range=(2, 8),  # Try 2 to 8 clusters
        use_original_features=False,  # Use selected features
        plot=True,
        results_dir=results_dir
    )
    
    # Save results and provide summary
    logging.info(f"Customer segmentation completed.")
    logging.info(f"Best clustering method: {best_method}")
    logging.info(f"Optimal number of clusters: {best_n_clusters}")
    
    # Add cluster labels to original data and save
    df['Cluster'] = labels
    df.to_csv(os.path.join(results_dir, "customer_segments_with_clusters.csv"), index=False)
    
    logging.info(f"Results saved to {results_dir} directory")

if __name__ == "__main__":
    main() 