import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import os
import logging
from sklearn.preprocessing import StandardScaler
import time
from tools.visualization import visualize_clusters, create_customer_segment_dashboard

logging.basicConfig(level=logging.INFO)

def run_clustering_models(X, n_clusters_range=(2, 10), methods=['kmeans', 'hierarchical', 'dbscan'], 
                         plot=True, results_dir=None):
    """
    Run multiple clustering models on the given dataset and return the best model.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        The input features for clustering.
    n_clusters_range : tuple, optional, default: (2, 10)
        Range of number of clusters to try (min, max inclusive).
    methods : list, optional, default: ['kmeans', 'hierarchical', 'dbscan']
        Clustering methods to use.
    plot : bool, optional, default: True
        Whether to plot clustering results.
    results_dir : str, optional, default: None
        Directory to save results. If None, results will not be saved.
        
    Returns:
    --------
    best_model : object
        The best clustering model.
    labels : array
        Cluster labels for the dataset.
    best_method : str
        Name of the best clustering method.
    best_n_clusters : int
        Number of clusters in the best model.
    """
    start_time = time.time()
    logging.info("Starting clustering analysis...")
    
    # Scale the data if not already scaled
    if X.max().max() > 10 or X.min().min() < -10:
        logging.info("Scaling features for clustering...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.values
    
    # Initialize variables to track the best model
    best_silhouette = -1
    best_model = None
    best_labels = None
    best_method = None
    best_n_clusters = None
    
    # For dimensionality reduction and visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Prepare the results
    results = []
    
    # Run different clustering methods
    for method in methods:
        logging.info(f"Running {method} clustering...")
        
        if method == 'kmeans':
            # Find the optimal number of clusters for K-means
            silhouette_scores = []
            models = []
            
            for n_clusters in range(n_clusters_range[0], n_clusters_range[1] + 1):
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                
                # Calculate silhouette score (only valid for n_clusters > 1)
                sil_score = silhouette_score(X_scaled, labels) if n_clusters > 1 else -1
                silhouette_scores.append(sil_score)
                models.append((kmeans, labels, n_clusters))
                
                results.append({
                    'method': 'kmeans',
                    'n_clusters': n_clusters,
                    'silhouette': sil_score
                })
                
                logging.info(f"K-means with {n_clusters} clusters: Silhouette score = {sil_score:.4f}")
            
            # Get the best K-means model
            best_kmeans_idx = np.argmax(silhouette_scores)
            kmeans_model, kmeans_labels, kmeans_n_clusters = models[best_kmeans_idx]
            kmeans_silhouette = silhouette_scores[best_kmeans_idx]
            
            if kmeans_silhouette > best_silhouette:
                best_silhouette = kmeans_silhouette
                best_model = kmeans_model
                best_labels = kmeans_labels
                best_method = 'kmeans'
                best_n_clusters = kmeans_n_clusters
            
            # Plot K-means results
            if plot:
                plt.figure(figsize=(12, 5))
                
                # Plot silhouette scores
                plt.subplot(1, 2, 1)
                plt.plot(range(n_clusters_range[0], n_clusters_range[1] + 1), silhouette_scores, marker='o')
                plt.xlabel('Number of clusters')
                plt.ylabel('Silhouette Score')
                plt.title('K-means: Silhouette Score vs. Number of Clusters')
                plt.grid(True)
                
                # Plot the best clustering
                plt.subplot(1, 2, 2)
                scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.6)
                plt.xlabel('PCA 1')
                plt.ylabel('PCA 2')
                plt.title(f'K-means Clustering (n_clusters={kmeans_n_clusters})')
                plt.colorbar(scatter, label='Cluster')
                plt.tight_layout()
                
                if results_dir:
                    plt.savefig(os.path.join(results_dir, 'kmeans_clustering.png'))
                plt.close()
        
        elif method == 'hierarchical':
            # Find the optimal number of clusters for hierarchical clustering
            silhouette_scores = []
            models = []
            
            for n_clusters in range(n_clusters_range[0], n_clusters_range[1] + 1):
                hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
                labels = hierarchical.fit_predict(X_scaled)
                
                # Calculate silhouette score (only valid for n_clusters > 1)
                sil_score = silhouette_score(X_scaled, labels) if n_clusters > 1 else -1
                silhouette_scores.append(sil_score)
                models.append((hierarchical, labels, n_clusters))
                
                results.append({
                    'method': 'hierarchical',
                    'n_clusters': n_clusters,
                    'silhouette': sil_score
                })
                
                logging.info(f"Hierarchical with {n_clusters} clusters: Silhouette score = {sil_score:.4f}")
            
            # Get the best hierarchical model
            best_hierarchical_idx = np.argmax(silhouette_scores)
            hierarchical_model, hierarchical_labels, hierarchical_n_clusters = models[best_hierarchical_idx]
            hierarchical_silhouette = silhouette_scores[best_hierarchical_idx]
            
            if hierarchical_silhouette > best_silhouette:
                best_silhouette = hierarchical_silhouette
                best_model = hierarchical_model
                best_labels = hierarchical_labels
                best_method = 'hierarchical'
                best_n_clusters = hierarchical_n_clusters
            
            # Plot hierarchical results
            if plot:
                plt.figure(figsize=(12, 5))
                
                # Plot silhouette scores
                plt.subplot(1, 2, 1)
                plt.plot(range(n_clusters_range[0], n_clusters_range[1] + 1), silhouette_scores, marker='o')
                plt.xlabel('Number of clusters')
                plt.ylabel('Silhouette Score')
                plt.title('Hierarchical: Silhouette Score vs. Number of Clusters')
                plt.grid(True)
                
                # Plot the best clustering
                plt.subplot(1, 2, 2)
                scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=hierarchical_labels, cmap='viridis', alpha=0.6)
                plt.xlabel('PCA 1')
                plt.ylabel('PCA 2')
                plt.title(f'Hierarchical Clustering (n_clusters={hierarchical_n_clusters})')
                plt.colorbar(scatter, label='Cluster')
                plt.tight_layout()
                
                if results_dir:
                    plt.savefig(os.path.join(results_dir, 'hierarchical_clustering.png'))
                plt.close()
        
        elif method == 'dbscan':
            # Try different epsilon values for DBSCAN
            eps_values = np.linspace(0.1, 2.0, 10)
            silhouette_scores = []
            models = []
            
            for eps in eps_values:
                dbscan = DBSCAN(eps=eps, min_samples=5)
                labels = dbscan.fit_predict(X_scaled)
                
                # Count number of clusters (excluding noise points labeled as -1)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                
                # Only calculate silhouette score if there are multiple clusters and no noise points
                if n_clusters > 1 and -1 not in labels:
                    sil_score = silhouette_score(X_scaled, labels)
                else:
                    sil_score = -1
                
                silhouette_scores.append(sil_score)
                models.append((dbscan, labels, n_clusters, eps))
                
                results.append({
                    'method': 'dbscan',
                    'eps': eps,
                    'n_clusters': n_clusters,
                    'silhouette': sil_score
                })
                
                logging.info(f"DBSCAN with eps={eps:.2f}: {n_clusters} clusters, Silhouette score = {sil_score:.4f}")
            
            # Get the best DBSCAN model (if there is a valid one)
            valid_scores = [s for s in silhouette_scores if s > -1]
            if valid_scores:
                best_dbscan_idx = silhouette_scores.index(max(valid_scores))
                dbscan_model, dbscan_labels, dbscan_n_clusters, dbscan_eps = models[best_dbscan_idx]
                dbscan_silhouette = silhouette_scores[best_dbscan_idx]
                
                if dbscan_silhouette > best_silhouette:
                    best_silhouette = dbscan_silhouette
                    best_model = dbscan_model
                    best_labels = dbscan_labels
                    best_method = 'dbscan'
                    best_n_clusters = dbscan_n_clusters
                
                # Plot DBSCAN results
                if plot:
                    plt.figure(figsize=(12, 5))
                    
                    # Plot silhouette scores
                    plt.subplot(1, 2, 1)
                    valid_indices = [i for i, s in enumerate(silhouette_scores) if s > -1]
                    valid_eps = [eps_values[i] for i in valid_indices]
                    valid_scores = [silhouette_scores[i] for i in valid_indices]
                    
                    if valid_eps:
                        plt.plot(valid_eps, valid_scores, marker='o')
                        plt.xlabel('Epsilon')
                        plt.ylabel('Silhouette Score')
                        plt.title('DBSCAN: Silhouette Score vs. Epsilon')
                        plt.grid(True)
                    else:
                        plt.text(0.5, 0.5, 'No valid DBSCAN configurations found', 
                                ha='center', va='center', transform=plt.gca().transAxes)
                    
                    # Plot the best clustering
                    plt.subplot(1, 2, 2)
                    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap='viridis', alpha=0.6)
                    plt.xlabel('PCA 1')
                    plt.ylabel('PCA 2')
                    plt.title(f'DBSCAN Clustering (eps={dbscan_eps:.2f}, clusters={dbscan_n_clusters})')
                    plt.colorbar(scatter, label='Cluster')
                    plt.tight_layout()
                    
                    if results_dir:
                        plt.savefig(os.path.join(results_dir, 'dbscan_clustering.png'))
                    plt.close()
    
    # Save results to CSV
    if results_dir:
        pd.DataFrame(results).to_csv(os.path.join(results_dir, 'clustering_results.csv'), index=False)
    
    # Create cluster profile analysis
    if best_labels is not None and isinstance(X, pd.DataFrame):
        X_with_clusters = X.copy()
        X_with_clusters['Cluster'] = best_labels
        
        # Generate cluster profiles
        cluster_profiles = X_with_clusters.groupby('Cluster').mean()
        
        if plot:
            # Plot heatmap of cluster profiles
            plt.figure(figsize=(15, 10))
            sns.heatmap(cluster_profiles, cmap='coolwarm', annot=True, fmt='.2f', cbar=True)
            plt.title(f'Cluster Profiles ({best_method.capitalize()}, n_clusters={best_n_clusters})')
            plt.tight_layout()
            
            if results_dir:
                plt.savefig(os.path.join(results_dir, 'cluster_profiles.png'))
            plt.close()
            
            # Create radar chart for each cluster
            create_radar_chart(cluster_profiles, results_dir)
        
        # Save cluster profiles
        if results_dir:
            cluster_profiles.to_csv(os.path.join(results_dir, 'cluster_profiles.csv'))
    
    elapsed_time = time.time() - start_time
    logging.info(f"Clustering analysis completed in {elapsed_time:.2f} seconds")
    logging.info(f"Best clustering method: {best_method} with {best_n_clusters} clusters")
    logging.info(f"Best silhouette score: {best_silhouette:.4f}")
    
    return best_model, best_labels, best_method, best_n_clusters

def create_radar_chart(cluster_profiles, results_dir=None):
    """
    Create radar charts for visualizing cluster profiles.
    
    Parameters:
    -----------
    cluster_profiles : pandas.DataFrame
        DataFrame containing the mean values for each feature per cluster.
    results_dir : str, optional, default: None
        Directory to save results. If None, results will not be saved.
    """
    # Normalize the data for radar chart
    normalized_profiles = (cluster_profiles - cluster_profiles.min()) / (cluster_profiles.max() - cluster_profiles.min())
    
    # Select top features for readability (up to 10)
    n_features = min(10, normalized_profiles.shape[1])
    
    if n_features < normalized_profiles.shape[1]:
        # Choose features with highest variance across clusters
        feature_variance = normalized_profiles.var()
        top_features = feature_variance.nlargest(n_features).index.tolist()
        normalized_profiles = normalized_profiles[top_features]
    
    # Prepare the radar chart
    n_clusters = normalized_profiles.shape[0]
    features = normalized_profiles.columns.tolist()
    
    # Set figure and subplot parameters based on number of clusters
    n_cols = min(3, n_clusters)
    n_rows = (n_clusters + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))
    
    for i, cluster_idx in enumerate(normalized_profiles.index):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, polar=True)
        
        # Calculate angles for each feature
        angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
        
        # Close the polygon
        values = normalized_profiles.loc[cluster_idx].tolist()
        values += [values[0]]
        angles += [angles[0]]
        features_plot = features + [features[0]]
        
        # Plot the polygon
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        
        # Set feature labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(features, size=8)
        
        # Set y-axis limits
        ax.set_ylim(0, 1)
        
        # Add title
        ax.set_title(f'Cluster {cluster_idx} Profile', size=14)
    
    plt.tight_layout()
    
    if results_dir:
        plt.savefig(os.path.join(results_dir, 'cluster_radar_charts.png'))
    plt.close()

def create_cluster_visualizations(X, labels, original_data=None, results_dir=None):
    """
    Create visualizations for clustering analysis.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        The input features used for clustering.
    labels : array
        Cluster labels for the dataset.
    original_data : pandas.DataFrame, optional, default: None
        Original dataset with additional features not used in clustering.
    results_dir : str, optional, default: None
        Directory to save results. If None, results will not be saved.
    """
    # Add cluster labels to the data
    X_with_clusters = X.copy()
    X_with_clusters['Cluster'] = labels
    
    # Apply PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X.values)
    
    # Create PCA scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('Cluster Visualization with PCA')
    plt.colorbar(scatter, label='Cluster')
    plt.tight_layout()
    
    if results_dir:
        plt.savefig(os.path.join(results_dir, 'cluster_pca_visualization.png'))
    plt.close()
    
    # If original data is provided, create additional visualizations
    if original_data is not None and isinstance(original_data, pd.DataFrame):
        # Add cluster labels to original data
        data_with_clusters = original_data.copy()
        data_with_clusters['Cluster'] = labels
        
        # Count customers by cluster
        cluster_counts = data_with_clusters['Cluster'].value_counts().sort_index()
        
        # Plot cluster sizes
        plt.figure(figsize=(10, 6))
        sns.barplot(x=cluster_counts.index, y=cluster_counts.values)
        plt.xlabel('Cluster')
        plt.ylabel('Number of Customers')
        plt.title('Number of Customers per Cluster')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if results_dir:
            plt.savefig(os.path.join(results_dir, 'cluster_sizes.png'))
        plt.close()
        
        # Select numerical columns for analysis
        numerical_cols = data_with_clusters.select_dtypes(include=['number']).columns.tolist()
        numerical_cols = [col for col in numerical_cols if col != 'Cluster']
        
        # Create boxplots for important features by cluster
        for feature in numerical_cols[:10]:  # Limit to top 10 features for readability
            plt.figure(figsize=(12, 6))
            sns.boxplot(x='Cluster', y=feature, data=data_with_clusters)
            plt.title(f'{feature} Distribution by Cluster')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            if results_dir:
                plt.savefig(os.path.join(results_dir, f'cluster_{feature}_boxplot.png'))
            plt.close()

def customer_segmentation_analysis(df, n_clusters_range=(2, 8), use_original_features=False, plot=True, results_dir=None):
    """
    Perform customer segmentation analysis using multiple clustering algorithms.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The preprocessed customer data to cluster.
    n_clusters_range : tuple, optional, default: (2, 8)
        The range of numbers of clusters to try (min, max).
    use_original_features : bool, optional, default: False
        Whether to use all features in the DataFrame. If False, will automatically select
        features relevant for customer segmentation.
    plot : bool, optional, default: True
        Whether to create and save plots.
    results_dir : str, optional, default: None
        Directory to save results. If None, results will not be saved.
        
    Returns:
    --------
    best_model : sklearn model
        The best clustering model found.
    labels : numpy.ndarray
        Cluster labels from the best model.
    best_method : str
        The name of the best clustering method.
    best_n_clusters : int
        The optimal number of clusters.
    """
    logging.info("Starting customer segmentation analysis...")
    
    # Create cluster results directory
    if results_dir:
        cluster_dir = os.path.join(results_dir, 'clustering')
        os.makedirs(cluster_dir, exist_ok=True)
    else:
        cluster_dir = None
    
    # Select features for clustering if not using all
    if not use_original_features:
        # Define feature groups relevant for customer segmentation
        demographic_cols = ['Year_Birth', 'Education', 'Marital_Status', 'Income', 
                           'Kidhome', 'Teenhome', 'Dt_Customer', 'Recency']
        
        purchase_cols = [col for col in df.columns if col.startswith('Mnt')]
        
        channel_cols = ['NumWebPurchases', 'NumCatalogPurchases', 
                       'NumStorePurchases', 'NumWebVisitsMonth']
        
        promotion_cols = ['NumDealsPurchases'] + [col for col in df.columns 
                                                if col.startswith('Accepted') or col == 'Response']
        
        # Select only available columns that exist in the dataset
        all_feature_cols = []
        for col_group in [demographic_cols, purchase_cols, channel_cols, promotion_cols]:
            all_feature_cols.extend([col for col in col_group if col in df.columns])
        
        X = df[all_feature_cols].copy()
        
        # Handle categorical columns (one-hot encoding)
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        # Handle missing values
        X = X.fillna(X.median(numeric_only=True))
    else:
        # Use all available features
        X = df.copy()
    
    logging.info(f"Using {X.shape[1]} features for clustering")
    
    # Run different clustering algorithms
    # 1. KMeans
    kmeans_model, kmeans_labels, kmeans_n_clusters, kmeans_scores = run_kmeans(
        X, n_clusters_range=n_clusters_range, plot=plot, results_dir=cluster_dir
    )
    
    # 2. Hierarchical (Ward)
    hierarchical_model, hierarchical_labels, hierarchical_n_clusters, hierarchical_scores = run_hierarchical(
        X, n_clusters_range=n_clusters_range, linkage='ward', plot=plot, results_dir=cluster_dir
    )
    
    # 3. DBSCAN (try to find a good range of eps values based on data)
    # Compute a reasonable eps range based on data distribution
    if X.shape[0] > 1000:
        # For large datasets, use a sample for computing distances
        sample_idx = np.random.choice(X.shape[0], min(1000, X.shape[0]), replace=False)
        X_sample = X.iloc[sample_idx] if isinstance(X, pd.DataFrame) else X[sample_idx]
    else:
        X_sample = X
    
    # Scale the sample
    X_sample_scaled = StandardScaler().fit_transform(X_sample)
    
    # Compute reasonable eps values
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=min(50, X_sample.shape[0])).fit(X_sample_scaled)
    distances, _ = nbrs.kneighbors(X_sample_scaled)
    
    # Use percentiles of distances as eps values
    eps_min = np.percentile(distances[:, 1:].flatten(), 5)  # 5th percentile
    eps_max = np.percentile(distances[:, 1:].flatten(), 95)  # 95th percentile
    
    dbscan_model, dbscan_labels, dbscan_params, dbscan_scores = run_dbscan(
        X, eps_range=(eps_min, eps_max, 4), min_samples_range=(5, 20, 3), 
        plot=plot, results_dir=cluster_dir
    )
    
    # Compare clustering results based on silhouette scores
    kmeans_silhouette = max(kmeans_scores.values()) if kmeans_scores else 0
    hierarchical_silhouette = max(hierarchical_scores.values()) if hierarchical_scores else 0
    
    # For DBSCAN, get the silhouette of the best model if available
    dbscan_silhouette = 0
    if dbscan_scores:
        dbscan_silhouette = max(dbscan_scores.values())
    
    # Determine the best method
    methods_silhouettes = {
        'KMeans': kmeans_silhouette,
        'Hierarchical': hierarchical_silhouette,
        'DBSCAN': dbscan_silhouette
    }
    
    best_method = max(methods_silhouettes, key=methods_silhouettes.get)
    logging.info(f"Best clustering method: {best_method} (silhouette score: {methods_silhouettes[best_method]:.4f})")
    
    # Return the best model and its labels
    if best_method == 'KMeans':
        best_model = kmeans_model
        labels = kmeans_labels
        best_n_clusters = kmeans_n_clusters
    elif best_method == 'Hierarchical':
        best_model = hierarchical_model
        labels = hierarchical_labels
        best_n_clusters = hierarchical_n_clusters
    else:  # DBSCAN
        best_model = dbscan_model
        labels = dbscan_labels
        best_n_clusters = dbscan_params['n_clusters']
    
    # Create visualizations for the clusters
    if plot and results_dir:
        # Create a copy of the original data with cluster labels
        df_with_clusters = df.copy()
        df_with_clusters['Cluster'] = labels
        
        # Visualize clusters
        visualize_clusters(df_with_clusters, cluster_col='Cluster', output_dir=results_dir)
        
        # Create customer segment dashboard
        create_customer_segment_dashboard(df_with_clusters, cluster_col='Cluster', output_dir=results_dir)
    
    logging.info("Customer segmentation analysis completed.")
    
    return best_model, labels, best_method, best_n_clusters

def run_kmeans(X, n_clusters_range=(2, 10), random_state=42, plot=False, results_dir=None):
    """
    Run KMeans clustering with different numbers of clusters and evaluate using silhouette score.
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        The data to cluster.
    n_clusters_range : tuple, optional, default: (2, 10)
        The range of numbers of clusters to try (min, max).
    random_state : int, optional, default: 42
        Random seed for reproducibility.
    plot : bool, optional, default: False
        Whether to create and save plots.
    results_dir : str, optional, default: None
        Directory to save results. If None, results will not be saved.
        
    Returns:
    --------
    best_model : sklearn.cluster.KMeans
        The best KMeans model.
    labels : numpy.ndarray
        Cluster labels from the best model.
    best_n_clusters : int
        The optimal number of clusters.
    silhouette_scores : dict
        Dictionary of silhouette scores for each number of clusters.
    """
    logging.info("Running KMeans clustering...")
    start_time = time.time()
    
    # Ensure X is a numpy array
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = X
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_array)
    
    # Try different numbers of clusters
    silhouette_scores = {}
    inertia_values = {}
    
    # Generate range of clusters to try
    min_clusters, max_clusters = n_clusters_range
    clusters_to_try = range(max(2, min_clusters), min(max_clusters + 1, X_array.shape[0] // 10 + 1))
    
    for n_clusters in clusters_to_try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        # Calculate silhouette score if there are at least 2 clusters and samples in each cluster
        if len(np.unique(labels)) > 1:
            silhouette_avg = silhouette_score(X_scaled, labels)
            silhouette_scores[n_clusters] = silhouette_avg
            
        inertia_values[n_clusters] = kmeans.inertia_
    
    # Find the best number of clusters based on silhouette score
    if silhouette_scores:
        best_n_clusters = max(silhouette_scores, key=silhouette_scores.get)
        logging.info(f"Best number of clusters for KMeans: {best_n_clusters}")
        
        # Train the final model with the best number of clusters
        best_model = KMeans(n_clusters=best_n_clusters, random_state=random_state, n_init=10)
        labels = best_model.fit_predict(X_scaled)
    else:
        # Fallback to using elbow method if silhouette scores couldn't be calculated
        inertia_deltas = {k: inertia_values[k-1] - v if k-1 in inertia_values else 0 
                         for k, v in inertia_values.items() if k > min(clusters_to_try)}
        if inertia_deltas:
            best_n_clusters = max(inertia_deltas, key=inertia_deltas.get)
        else:
            best_n_clusters = min(clusters_to_try)
        
        logging.info(f"Using elbow method, best number of clusters: {best_n_clusters}")
        
        # Train the final model with the best number of clusters
        best_model = KMeans(n_clusters=best_n_clusters, random_state=random_state, n_init=10)
        labels = best_model.fit_predict(X_scaled)
    
    # Create plots if requested
    if plot and results_dir:
        # Plot silhouette scores
        if silhouette_scores:
            plt.figure(figsize=(12, 6))
            plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), marker='o')
            plt.axvline(x=best_n_clusters, color='r', linestyle='--', 
                      label=f'Best n_clusters: {best_n_clusters}')
            plt.xlabel('Number of Clusters')
            plt.ylabel('Silhouette Score')
            plt.title('Silhouette Score for Different Numbers of Clusters (KMeans)')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(results_dir, 'kmeans_silhouette_scores.png'))
            plt.close()
        
        # Plot elbow curve (inertia)
        plt.figure(figsize=(12, 6))
        plt.plot(list(inertia_values.keys()), list(inertia_values.values()), marker='o')
        plt.axvline(x=best_n_clusters, color='r', linestyle='--', 
                  label=f'Best n_clusters: {best_n_clusters}')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal Number of Clusters (KMeans)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, 'kmeans_elbow_curve.png'))
        plt.close()
        
        # Plot the clusters using PCA for dimensionality reduction
        if X_array.shape[1] > 2:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)
            plt.colorbar(scatter, label='Cluster')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.title(f'KMeans Clustering (n_clusters={best_n_clusters})')
            plt.savefig(os.path.join(results_dir, 'kmeans_clusters_pca.png'))
            plt.close()
    
    elapsed_time = time.time() - start_time
    logging.info(f"KMeans clustering completed in {elapsed_time:.2f} seconds")
    
    return best_model, labels, best_n_clusters, silhouette_scores

def run_hierarchical(X, n_clusters_range=(2, 10), linkage='ward', plot=False, results_dir=None):
    """
    Run Hierarchical (Agglomerative) clustering with different numbers of clusters.
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        The data to cluster.
    n_clusters_range : tuple, optional, default: (2, 10)
        The range of numbers of clusters to try (min, max).
    linkage : str, optional, default: 'ward'
        The linkage criterion to use (ward, complete, average, single).
    plot : bool, optional, default: False
        Whether to create and save plots.
    results_dir : str, optional, default: None
        Directory to save results. If None, results will not be saved.
        
    Returns:
    --------
    best_model : sklearn.cluster.AgglomerativeClustering
        The best Hierarchical clustering model.
    labels : numpy.ndarray
        Cluster labels from the best model.
    best_n_clusters : int
        The optimal number of clusters.
    silhouette_scores : dict
        Dictionary of silhouette scores for each number of clusters.
    """
    logging.info(f"Running Hierarchical clustering with {linkage} linkage...")
    start_time = time.time()
    
    # Ensure X is a numpy array
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = X
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_array)
    
    # Try different numbers of clusters
    silhouette_scores = {}
    
    # Generate range of clusters to try
    min_clusters, max_clusters = n_clusters_range
    clusters_to_try = range(max(2, min_clusters), min(max_clusters + 1, X_array.shape[0] // 10 + 1))
    
    for n_clusters in clusters_to_try:
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels = hierarchical.fit_predict(X_scaled)
        
        # Calculate silhouette score if there are at least 2 clusters and samples in each cluster
        if len(np.unique(labels)) > 1:
            silhouette_avg = silhouette_score(X_scaled, labels)
            silhouette_scores[n_clusters] = silhouette_avg
    
    # Find the best number of clusters based on silhouette score
    if silhouette_scores:
        best_n_clusters = max(silhouette_scores, key=silhouette_scores.get)
        logging.info(f"Best number of clusters for Hierarchical: {best_n_clusters}")
    else:
        best_n_clusters = min(clusters_to_try)
        logging.info(f"Using minimum number of clusters for Hierarchical: {best_n_clusters}")
    
    # Train the final model with the best number of clusters
    best_model = AgglomerativeClustering(n_clusters=best_n_clusters, linkage=linkage)
    labels = best_model.fit_predict(X_scaled)
    
    # Create plots if requested
    if plot and results_dir:
        # Plot silhouette scores
        if silhouette_scores:
            plt.figure(figsize=(12, 6))
            plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), marker='o')
            plt.axvline(x=best_n_clusters, color='r', linestyle='--', 
                      label=f'Best n_clusters: {best_n_clusters}')
            plt.xlabel('Number of Clusters')
            plt.ylabel('Silhouette Score')
            plt.title(f'Silhouette Score for Different Numbers of Clusters (Hierarchical - {linkage})')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(results_dir, f'hierarchical_{linkage}_silhouette_scores.png'))
            plt.close()
        
        # Plot the clusters using PCA for dimensionality reduction
        if X_array.shape[1] > 2:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)
            plt.colorbar(scatter, label='Cluster')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.title(f'Hierarchical Clustering - {linkage} (n_clusters={best_n_clusters})')
            plt.savefig(os.path.join(results_dir, f'hierarchical_{linkage}_clusters_pca.png'))
            plt.close()
    
    elapsed_time = time.time() - start_time
    logging.info(f"Hierarchical clustering completed in {elapsed_time:.2f} seconds")
    
    return best_model, labels, best_n_clusters, silhouette_scores

def run_dbscan(X, eps_range=(0.1, 1.0, 5), min_samples_range=(5, 20, 4), plot=False, results_dir=None):
    """
    Run DBSCAN clustering with different parameters.
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        The data to cluster.
    eps_range : tuple, optional, default: (0.1, 1.0, 5)
        The range of eps values to try (min, max, num_steps).
    min_samples_range : tuple, optional, default: (5, 20, 4)
        The range of min_samples values to try (min, max, num_steps).
    plot : bool, optional, default: False
        Whether to create and save plots.
    results_dir : str, optional, default: None
        Directory to save results. If None, results will not be saved.
        
    Returns:
    --------
    best_model : sklearn.cluster.DBSCAN
        The best DBSCAN model.
    labels : numpy.ndarray
        Cluster labels from the best model.
    best_params : dict
        The optimal parameters (eps, min_samples).
    silhouette_scores : dict
        Dictionary of silhouette scores for each parameter combination.
    """
    logging.info("Running DBSCAN clustering...")
    start_time = time.time()
    
    # Ensure X is a numpy array
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = X
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_array)
    
    # Generate parameter combinations to try
    eps_min, eps_max, eps_steps = eps_range
    min_samples_min, min_samples_max, min_samples_steps = min_samples_range
    
    eps_values = np.linspace(eps_min, eps_max, eps_steps)
    min_samples_values = np.linspace(min_samples_min, min_samples_max, min_samples_steps).astype(int)
    
    # Try different parameter combinations
    results = []
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X_scaled)
            
            # Count number of clusters (excluding noise points labeled as -1)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            # Calculate silhouette score if there are at least 2 clusters (excluding noise)
            # and samples in each cluster
            silhouette_avg = None
            if n_clusters > 1:
                # Remove noise points for silhouette calculation
                mask = labels != -1
                if np.sum(mask) > n_clusters:  # Make sure we have enough points after removing noise
                    try:
                        silhouette_avg = silhouette_score(X_scaled[mask], labels[mask])
                    except:
                        silhouette_avg = None
            
            results.append({
                'eps': eps,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'silhouette': silhouette_avg
            })
    
    # Filter results to include only valid clusterings (at least 2 clusters, not too many noise points)
    valid_results = [r for r in results if r['n_clusters'] >= 2 and 
                    r['n_noise'] < 0.5 * X_array.shape[0] and  # Less than 50% noise
                    r['silhouette'] is not None]
    
    # Find the best parameters based on silhouette score
    if valid_results:
        best_result = max(valid_results, key=lambda x: x['silhouette'])
        best_eps = best_result['eps']
        best_min_samples = best_result['min_samples']
        best_n_clusters = best_result['n_clusters']
        logging.info(f"Best DBSCAN parameters: eps={best_eps:.2f}, min_samples={best_min_samples}")
        logging.info(f"Number of clusters: {best_n_clusters}")
        
        # Train the final model with the best parameters
        best_model = DBSCAN(eps=best_eps, min_samples=best_min_samples)
        labels = best_model.fit_predict(X_scaled)
        
        # Create silhouette scores dictionary
        silhouette_scores = {(r['eps'], r['min_samples']): r['silhouette'] for r in valid_results}
    else:
        logging.warning("DBSCAN did not find any valid clustering. Using default parameters.")
        best_eps = 0.5
        best_min_samples = 5
        best_model = DBSCAN(eps=best_eps, min_samples=best_min_samples)
        labels = best_model.fit_predict(X_scaled)
        best_n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        silhouette_scores = {}
    
    # Create plots if requested
    if plot and results_dir and valid_results:
        # Plot silhouette scores vs parameters
        eps_values = sorted(list(set([r['eps'] for r in valid_results])))
        min_samples_values = sorted(list(set([r['min_samples'] for r in valid_results])))
        
        # Create a heatmap of silhouette scores
        if len(eps_values) > 1 and len(min_samples_values) > 1:
            silhouette_matrix = np.zeros((len(eps_values), len(min_samples_values)))
            for i, eps in enumerate(eps_values):
                for j, min_samples in enumerate(min_samples_values):
                    for r in valid_results:
                        if r['eps'] == eps and r['min_samples'] == min_samples:
                            silhouette_matrix[i, j] = r['silhouette'] if r['silhouette'] is not None else np.nan
            
            plt.figure(figsize=(12, 8))
            ax = sns.heatmap(silhouette_matrix, annot=True, fmt='.3f', cmap='viridis',
                          xticklabels=[str(x) for x in min_samples_values],
                          yticklabels=[f'{x:.2f}' for x in eps_values])
            plt.ylabel('eps')
            plt.xlabel('min_samples')
            plt.title('DBSCAN Silhouette Scores')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'dbscan_silhouette_heatmap.png'))
            plt.close()
        
        # Plot the clusters using PCA for dimensionality reduction
        if X_array.shape[1] > 2:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            plt.figure(figsize=(10, 8))
            
            # Plot non-noise points with cluster colors
            unique_labels = set(labels)
            colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels) - (1 if -1 in unique_labels else 0)))
            color_idx = 0
            
            for label in unique_labels:
                if label == -1:
                    # Plot noise points in black
                    mask = labels == label
                    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c='black', marker='x', alpha=0.6, label='Noise')
                else:
                    # Plot cluster points
                    mask = labels == label
                    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[colors[color_idx]], alpha=0.6, label=f'Cluster {label}')
                    color_idx += 1
            
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.title(f'DBSCAN Clustering (eps={best_eps:.2f}, min_samples={best_min_samples})')
            plt.legend()
            plt.savefig(os.path.join(results_dir, 'dbscan_clusters_pca.png'))
            plt.close()
    
    elapsed_time = time.time() - start_time
    logging.info(f"DBSCAN clustering completed in {elapsed_time:.2f} seconds")
    
    return best_model, labels, {'eps': best_eps, 'min_samples': best_min_samples, 'n_clusters': best_n_clusters}, silhouette_scores 