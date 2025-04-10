import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os

logging.basicConfig(level=logging.INFO)

def select_features(X, y=None, problem_type='classification', n_features_to_select=None, 
                    downsample=False, log=True, plot=True, results_dir=None):
    """
    Selects the most relevant features for a machine learning model using appropriate feature selection methods
    based on the problem type.

    Parameters:
    -----------
    X : pandas.DataFrame
        The input features (independent variables) to perform feature selection on.
    y : pandas.Series or numpy.array, optional, default: None
        The target variable (dependent variable) corresponding to the input features. 
        Required for classification and regression problems, optional for clustering.
    problem_type : str, optional, default: 'classification'
        The type of machine learning problem. Supported values are 'classification', 'regression', and 'clustering'.
    n_features_to_select : int, optional, default: None
        The number of top features to select. If None, an optimal number will be determined automatically.
    downsample : bool, optional, default: False
        Whether to downsample the data to speed up computation (for large datasets).
    log : bool, optional, default: True
        Whether to log progress and results.
    plot : bool, optional, default: True
        Whether to create and save plots.
    results_dir : str, optional, default: None
        Directory to save results. If None, results will not be saved.

    Returns:
    --------
    X_selected : pandas.DataFrame
        DataFrame containing only the selected features.
    """
    start_time = time.time()
    
    if log:
        logging.info(f"Starting feature selection for {problem_type} problem...")
    
    # Create a copy of the input data
    X_copy = X.copy()
    
    # For clustering, we can't use RFECV directly as it requires a target
    if problem_type == 'clustering':
        if log:
            logging.info("Using PCA and variance-based feature selection for clustering...")
        
        # For clustering, select features based on:
        # 1. Feature variance (higher variance = more information)
        # 2. PCA contribution for top components
        
        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_copy)
        
        # Calculate variance for each feature
        feature_variances = np.var(X_scaled, axis=0)
        variance_ranking = pd.Series(feature_variances, index=X_copy.columns).sort_values(ascending=False)
        
        # Apply PCA to identify important features
        pca = PCA(n_components=min(10, X_scaled.shape[1]))
        pca_result = pca.fit_transform(X_scaled)
        
        # Calculate feature importance based on PCA components
        feature_importance = np.abs(pca.components_[:3]).sum(axis=0)  # Use top 3 components
        pca_ranking = pd.Series(feature_importance, index=X_copy.columns).sort_values(ascending=False)
        
        # Combined ranking (normalized sum of variance and PCA rankings)
        variance_rank_normalized = (variance_ranking - variance_ranking.min()) / (variance_ranking.max() - variance_ranking.min())
        pca_rank_normalized = (pca_ranking - pca_ranking.min()) / (pca_ranking.max() - pca_ranking.min())
        combined_ranking = (variance_rank_normalized + pca_rank_normalized) / 2
        
        # Determine number of features to select
        if n_features_to_select is None:
            # Choose optimal number based on explained variance
            explained_variance = pca.explained_variance_ratio_.cumsum()
            n_features_to_select = np.argmax(explained_variance > 0.8) + 1
            n_features_to_select = max(min(n_features_to_select, X_copy.shape[1] // 2), 5)  # At least 5, at most half
        
        # Select top features
        selected_features = combined_ranking.nlargest(n_features_to_select).index.tolist()
        
        # Evaluate clustering quality with selected features using silhouette score
        X_selected_scaled = scaler.fit_transform(X_copy[selected_features])
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X_selected_scaled)
        silhouette = silhouette_score(X_selected_scaled, clusters)
        
        if log:
            logging.info(f"Selected {len(selected_features)} features for clustering")
            logging.info(f"Silhouette score with selected features: {silhouette:.4f}")
        
        # Create and save plots
        if plot and results_dir:
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            combined_ranking.nlargest(min(20, len(combined_ranking))).sort_values().plot(kind='barh')
            plt.title('Feature Importance for Clustering')
            plt.xlabel('Importance Score')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'clustering_feature_importance.png'))
            plt.close()
            
            # Plot PCA of selected features
            plt.figure(figsize=(10, 8))
            pca_selected = PCA(n_components=2).fit_transform(X_selected_scaled)
            plt.scatter(pca_selected[:, 0], pca_selected[:, 1], c=clusters, cmap='viridis', alpha=0.6)
            plt.title('PCA of Selected Features')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.colorbar(label='Cluster')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'clustering_pca_selected_features.png'))
            plt.close()
    else:
        # For classification and regression, use RFECV
        if y is None and problem_type in ['classification', 'regression']:
            raise ValueError(f"Target variable (y) is required for {problem_type} problems")
        
        # Downsample if requested for large datasets
        if downsample and X_copy.shape[0] > 10000:
            sample_size = min(10000, X_copy.shape[0] // 2)
            if log:
                logging.info(f"Downsampling from {X_copy.shape[0]} to {sample_size} samples")
            
            # Stratified sampling for classification
            if problem_type == 'classification':
                X_sampled = pd.DataFrame()
                y_sampled = pd.Series()
                
                for label in np.unique(y):
                    idx = (y == label)
                    label_size = min(int(sample_size * sum(idx) / len(y)), sum(idx))
                    if label_size > 0:
                        X_label = X_copy[idx].sample(label_size, random_state=42)
                        X_sampled = pd.concat([X_sampled, X_label])
                        y_sampled = pd.concat([y_sampled, pd.Series([label] * len(X_label), index=X_label.index)])
            else:
                # Random sampling for regression
                sampled_indices = np.random.choice(X_copy.index, sample_size, replace=False)
                X_sampled = X_copy.loc[sampled_indices]
                y_sampled = y.loc[sampled_indices]
        else:
            X_sampled = X_copy
            y_sampled = y
        
        # Create estimator based on problem type
        if problem_type == 'classification':
            estimator = DecisionTreeClassifier(class_weight='balanced', random_state=42)
        elif problem_type == 'regression':
            estimator = LinearRegression()
        
        # Perform RFECV
        if log:
            logging.info(f"Running RFECV for {problem_type} with {X_sampled.shape[1]} features...")
        
        selector = RFECV(
            estimator=estimator,
            step=1,
            cv=5,
            scoring='accuracy' if problem_type == 'classification' else 'r2',
            n_jobs=-1
        )
        
        selector.fit(X_sampled, y_sampled)
        
        # Get RFECV results
        optimal_num_features = selector.n_features_
        optimal_features = X_sampled.columns[selector.support_].tolist()
        
        # Adjust number of features if requested
        if n_features_to_select is not None:
            if n_features_to_select < len(optimal_features):
                if log:
                    logging.info(f"Limiting to {n_features_to_select} features as requested (down from {len(optimal_features)})")
                # Get feature importances
                if hasattr(selector.estimator_, 'feature_importances_'):
                    importances = selector.estimator_.feature_importances_
                else:
                    importances = np.abs(selector.estimator_.coef_[0])
                
                # Get indices of selected features
                selected_indices = np.where(selector.support_)[0]
                
                # Sort selected features by importance
                selected_sorted = sorted(zip(selected_indices, importances[selected_indices]), 
                                        key=lambda x: x[1], reverse=True)
                
                # Take the top n_features_to_select
                top_feature_indices = [idx for idx, _ in selected_sorted[:n_features_to_select]]
                selected_features = X_sampled.columns[top_feature_indices].tolist()
            else:
                selected_features = optimal_features
                if log:
                    logging.info(f"Using all {len(optimal_features)} optimal features")
        else:
            selected_features = optimal_features
        
        # Create and save plots
        if plot and results_dir:
            # Plot RFECV results
            plt.figure(figsize=(12, 6))
            plt.plot(range(1, len(selector.cv_results_['mean_test_score']) + 1), 
                    selector.cv_results_['mean_test_score'], marker='o')
            plt.axvline(x=optimal_num_features, color='r', linestyle='--', 
                        label=f'Optimal number of features: {optimal_num_features}')
            plt.xlabel('Number of features')
            plt.ylabel('Cross-validation score')
            plt.title(f'RFECV Results for {problem_type.capitalize()}')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f'{problem_type}_rfecv_results.png'))
            plt.close()
            
            # Plot feature importance if available
            if hasattr(selector.estimator_, 'feature_importances_') or hasattr(selector.estimator_, 'coef_'):
                plt.figure(figsize=(12, 8))
                
                if hasattr(selector.estimator_, 'feature_importances_'):
                    importances = selector.estimator_.feature_importances_
                else:
                    importances = np.abs(selector.estimator_.coef_[0])
                
                # Get only selected features
                selected_indices = np.where(selector.support_)[0]
                selected_importances = importances[selected_indices]
                selected_names = X_sampled.columns[selected_indices]
                
                # Sort features by importance
                sorted_idx = np.argsort(selected_importances)
                
                # Plot
                plt.barh(range(len(sorted_idx)), selected_importances[sorted_idx])
                plt.yticks(range(len(sorted_idx)), selected_names[sorted_idx])
                plt.xlabel('Feature Importance')
                plt.title(f'Feature Importance for {problem_type.capitalize()}')
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, f'{problem_type}_feature_importance.png'))
                plt.close()
    
    # Create final selected features DataFrame
    X_selected = X_copy[selected_features]
    
    # Save selected features to CSV
    if results_dir:
        with open(os.path.join(results_dir, 'selected_features.txt'), 'w') as f:
            f.write(f"Problem type: {problem_type}\n")
            f.write(f"Number of selected features: {len(selected_features)}\n")
            f.write(f"Selected features:\n")
            for feature in selected_features:
                f.write(f"- {feature}\n")
    
    elapsed_time = time.time() - start_time
    if log:
        logging.info(f"Feature selection completed in {elapsed_time:.2f} seconds")
        logging.info(f"Selected {len(selected_features)} features: {selected_features}")
    
    return X_selected
