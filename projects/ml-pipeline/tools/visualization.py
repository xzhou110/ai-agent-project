import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import logging
from typing import List, Dict, Tuple, Optional, Union, Any
import itertools
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve,
    auc, silhouette_samples, silhouette_score,
    calinski_harabasz_score, davies_bouldin_score
)

logging.basicConfig(level=logging.INFO)

def set_style():
    """Set consistent plotting style for all visualizations."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 18

def create_output_dir(base_dir: str, subdir: str = None) -> str:
    """
    Create output directory for saving visualizations.
    
    Parameters:
    -----------
    base_dir : str
        Base directory for saving results
    subdir : str, optional
        Subdirectory within base_dir
        
    Returns:
    --------
    output_dir : str
        Path to created directory
    """
    output_dir = base_dir
    if subdir:
        output_dir = os.path.join(base_dir, subdir)
    
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def save_figure(fig, filename, output_dir, dpi=300, formats=['png', 'pdf']):
    """
    Save a matplotlib figure to disk in multiple formats.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to save
    filename : str
        Base filename without extension
    output_dir : str
        Directory to save figure
    dpi : int, optional, default: 300
        Resolution of saved figure
    formats : list of str, optional, default: ['png', 'pdf']
        File formats to save
    """
    for fmt in formats:
        filepath = os.path.join(output_dir, f"{filename}.{fmt}")
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        logging.debug(f"Figure saved to {filepath}")

def exploratory_data_analysis(df, categorical_cols=None, numerical_cols=None, results_dir=None):
    """
    Perform exploratory data analysis and create visualizations.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to analyze.
    categorical_cols : list, optional, default: None
        List of categorical columns to analyze. If None, will be inferred.
    numerical_cols : list, optional, default: None
        List of numerical columns to analyze. If None, will be inferred.
    results_dir : str, optional, default: None
        Directory to save results. If None, results will not be saved.
    """
    logging.info("Starting exploratory data analysis...")
    
    # Create results directory if specified
    if results_dir:
        os.makedirs(os.path.join(results_dir, 'eda'), exist_ok=True)
        eda_dir = os.path.join(results_dir, 'eda')
    
    # Auto-detect column types if not provided
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    if numerical_cols is None:
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Basic statistics
    summary_stats = df[numerical_cols].describe().T
    summary_stats['missing'] = df[numerical_cols].isnull().sum()
    summary_stats['missing_pct'] = df[numerical_cols].isnull().mean() * 100
    
    if results_dir:
        summary_stats.to_csv(os.path.join(eda_dir, 'numerical_summary.csv'))
    
    # Missing values visualization
    plt.figure(figsize=(12, 6))
    missing = df.isnull().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    
    if len(missing) > 0:
        missing_pct = missing / len(df) * 100
        missing_df = pd.DataFrame({'Count': missing, 'Percent': missing_pct})
        
        plt.barh(range(len(missing)), missing_pct, color='steelblue')
        plt.yticks(range(len(missing)), missing.index)
        plt.xlabel('Percentage of Missing Values')
        plt.title('Missing Values Analysis')
        plt.tight_layout()
        
        if results_dir:
            plt.savefig(os.path.join(eda_dir, 'missing_values.png'))
            missing_df.to_csv(os.path.join(eda_dir, 'missing_values.csv'))
        plt.close()
    
    # Distribution of numerical features
    for i, col in enumerate(numerical_cols[:15]):  # Limit to first 15 numerical columns
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f'Distribution of {col}')
        
        plt.subplot(1, 2, 2)
        sns.boxplot(y=df[col].dropna())
        plt.title(f'Boxplot of {col}')
        
        plt.tight_layout()
        if results_dir:
            plt.savefig(os.path.join(eda_dir, f'distribution_{col}.png'))
        plt.close()
    
    # Correlation heatmap
    if len(numerical_cols) > 1:
        plt.figure(figsize=(12, 10))
        corr = df[numerical_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=False, 
                    vmin=-1, vmax=1, square=True, linewidths=0.5)
        plt.title('Correlation Heatmap of Numerical Features')
        plt.tight_layout()
        
        if results_dir:
            plt.savefig(os.path.join(eda_dir, 'correlation_heatmap.png'))
            corr.to_csv(os.path.join(eda_dir, 'correlation_matrix.csv'))
        plt.close()
    
    # Top correlations (positive and negative)
    if len(numerical_cols) > 1:
        corr_pairs = []
        for i in range(len(numerical_cols)):
            for j in range(i+1, len(numerical_cols)):
                col1 = numerical_cols[i]
                col2 = numerical_cols[j]
                correlation = df[col1].corr(df[col2])
                corr_pairs.append((col1, col2, correlation))
        
        corr_df = pd.DataFrame(corr_pairs, columns=['Feature1', 'Feature2', 'Correlation'])
        top_pos_corr = corr_df.sort_values('Correlation', ascending=False).head(10)
        top_neg_corr = corr_df.sort_values('Correlation').head(10)
        
        if results_dir:
            top_pos_corr.to_csv(os.path.join(eda_dir, 'top_positive_correlations.csv'), index=False)
            top_neg_corr.to_csv(os.path.join(eda_dir, 'top_negative_correlations.csv'), index=False)
    
    # Distribution of categorical features
    for col in categorical_cols[:10]:  # Limit to first 10 categorical columns
        plt.figure(figsize=(10, 6))
        value_counts = df[col].value_counts().reset_index()
        value_counts.columns = [col, 'Count']
        value_counts['Percentage'] = value_counts['Count'] / len(df) * 100
        
        if len(value_counts) <= 10:  # Show all values if less than 10
            sns.barplot(x=col, y='Count', data=value_counts)
            plt.xticks(rotation=45, ha='right')
        else:  # Show top 10 values if more than 10
            top_10 = value_counts.head(10)
            sns.barplot(x=col, y='Count', data=top_10)
            plt.xticks(rotation=45, ha='right')
            plt.title(f'Top 10 values of {col}')
        
        plt.tight_layout()
        if results_dir:
            plt.savefig(os.path.join(eda_dir, f'categorical_{col}.png'))
            value_counts.to_csv(os.path.join(eda_dir, f'value_counts_{col}.csv'), index=False)
        plt.close()
    
    # PCA visualization
    if len(numerical_cols) > 2:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[numerical_cols].dropna())
        
        if X_scaled.shape[0] > 0:  # Make sure we have data after dropping NAs
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            plt.figure(figsize=(10, 8))
            plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.title('PCA of Numerical Features')
            plt.tight_layout()
            
            if results_dir:
                plt.savefig(os.path.join(eda_dir, 'pca_visualization.png'))
            plt.close()
    
    logging.info("Exploratory data analysis completed.")

def visualize_clusters(
    df: pd.DataFrame,
    cluster_col: str = 'Cluster',
    n_components: int = 2,
    plot_type: str = 'pca',
    output_dir: str = None,
    random_state: int = 42
) -> Union[plt.Figure, Dict[str, plt.Figure]]:
    """
    Visualize clusters using dimensionality reduction.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with cluster assignments
    cluster_col : str, optional, default: 'Cluster'
        Name of column containing cluster assignments
    n_components : int, optional, default: 2
        Number of components for dimensionality reduction
    plot_type : str, optional, default: 'pca'
        Type of plot, one of 'pca', 'pairs', 'all'
    output_dir : str, optional
        Directory to save figures
    random_state : int, optional, default: 42
        Random state for reproducibility
        
    Returns:
    --------
    fig or dict
        Created figure(s)
    """
    set_style()
    
    # Extract features and cluster labels
    X = df.drop(columns=[cluster_col])
    labels = df[cluster_col].values
    
    # Handle categorical features
    X = pd.get_dummies(X, drop_first=True)
    
    # Create output directory if needed
    if output_dir:
        output_dir = create_output_dir(output_dir, 'clusters')
    
    if plot_type in ['pca', 'all']:
        # Perform PCA
        pca = PCA(n_components=n_components, random_state=random_state)
        X_pca = pca.fit_transform(X)
        
        # Create PCA plot
        fig_pca, ax = plt.subplots(figsize=(12, 10))
        
        # Plot clusters
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.7, s=50)
        
        # Add legend
        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend1)
        
        # Add labels and title
        ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax.set_title('PCA Cluster Visualization')
        
        # Save if output_dir provided
        if output_dir:
            save_figure(fig_pca, 'pca_clusters', output_dir)
        
        if plot_type == 'pca':
            return fig_pca
    
    if plot_type in ['pairs', 'all']:
        # Select most important features for pairs plot
        if X.shape[1] > 10:
            # If we have many features, select top features by variance
            variances = X.var().sort_values(ascending=False)
            top_features = variances.index[:8]  # Top 8 features for pairs plot
            X_pairs = X[top_features]
        else:
            X_pairs = X
        
        # Create pairs plot
        X_pairs_df = X_pairs.copy()
        X_pairs_df['Cluster'] = labels
        
        fig_pairs = plt.figure(figsize=(16, 12))
        g = sns.pairplot(X_pairs_df, hue='Cluster', palette='viridis', 
                        plot_kws={'alpha': 0.7, 's': 30}, diag_kind='kde')
        g.fig.suptitle('Feature Relationships by Cluster', y=1.02, fontsize=16)
        
        # Save if output_dir provided
        if output_dir:
            save_figure(g.fig, 'pairs_clusters', output_dir)
        
        if plot_type == 'pairs':
            return g.fig
    
    # Return all figures if plot_type is 'all'
    if plot_type == 'all':
        return {'pca': fig_pca, 'pairs': g.fig}

def plot_silhouette_analysis(
    X: np.ndarray,
    labels: np.ndarray,
    metric: str = 'euclidean',
    title: str = "Silhouette Analysis",
    output_dir: str = None,
    filename: str = "silhouette_analysis"
) -> plt.Figure:
    """
    Plot silhouette analysis for clustering.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Data matrix
    labels : numpy.ndarray
        Cluster labels
    metric : str, optional, default: 'euclidean'
        Distance metric for silhouette calculation
    title : str, optional
        Plot title
    output_dir : str, optional
        Directory to save figure
    filename : str, optional
        Filename for saved figure
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Created figure
    """
    set_style()
    
    # Calculate silhouette scores
    silhouette_avg = silhouette_score(X, labels, metric=metric)
    sample_silhouette_values = silhouette_samples(X, labels, metric=metric)
    
    # Create silhouette plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_lower = 10
    n_clusters = len(np.unique(labels))
    
    # Plot silhouette scores for each cluster
    for i in range(n_clusters):
        # Get silhouette scores for current cluster
        ith_cluster_values = sample_silhouette_values[labels == i]
        ith_cluster_values.sort()
        
        size_cluster_i = ith_cluster_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = plt.cm.viridis(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_values,
                         facecolor=color, edgecolor=color, alpha=0.7)
        
        # Label the silhouette plots with cluster numbers
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10
    
    # Add average silhouette score line
    ax.axvline(x=silhouette_avg, color="red", linestyle="--", 
              label=f"Average Silhouette: {silhouette_avg:.3f}")
    
    ax.set_title(title)
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster")
    ax.set_yticks([])  # Clear y-axis labels
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.legend(loc="best")
    
    plt.tight_layout()
    
    # Save if output_dir provided
    if output_dir:
        save_figure(fig, filename, output_dir)
    
    return fig

def create_customer_segment_dashboard(
    df: pd.DataFrame,
    cluster_col: str = 'Cluster',
    output_dir: str = None
) -> Dict[str, plt.Figure]:
    """
    Create a comprehensive dashboard for customer segmentation.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with cluster assignments
    cluster_col : str, optional, default: 'Cluster'
        Name of column containing cluster assignments
    output_dir : str, optional
        Directory to save figures
        
    Returns:
    --------
    figs : dict
        Dictionary of created figures
    """
    set_style()
    
    # Check if output_dir exists
    if output_dir:
        output_dir = create_output_dir(output_dir, 'dashboard')
    
    # Store all created figures
    figs = {}
    
    # 1. Cluster distribution
    fig_dist, ax = plt.subplots(figsize=(10, 6))
    cluster_counts = df[cluster_col].value_counts().sort_index()
    
    # Calculate percentages
    percentages = [f"{count/len(df)*100:.1f}%" for count in cluster_counts]
    
    # Create bar plot
    bars = ax.bar(cluster_counts.index.astype(str), cluster_counts.values, 
                 color=plt.cm.viridis(np.linspace(0, 1, len(cluster_counts))))
    
    # Add values on top of bars
    for bar, count, percentage in zip(bars, cluster_counts, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
               f"{count}\n{percentage}", ha='center', va='bottom')
    
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of Customers')
    ax.set_title('Customer Distribution Across Clusters')
    
    # Save if output_dir provided
    if output_dir:
        save_figure(fig_dist, 'cluster_distribution', output_dir)
    
    figs['distribution'] = fig_dist
    
    # 2. Feature distributions by cluster
    # Identify numeric features for analysis
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != cluster_col]
    
    # Select top features (to avoid too many plots)
    if len(numeric_cols) > 10:
        # Calculate variance for each feature across clusters
        variances = []
        for col in numeric_cols:
            # Group by cluster and calculate means
            cluster_means = df.groupby(cluster_col)[col].mean()
            # Calculate variance of means across clusters
            variances.append((col, cluster_means.var()))
        
        # Sort by variance (which features differ most across clusters)
        variances.sort(key=lambda x: x[1], reverse=True)
        top_features = [x[0] for x in variances[:8]]  # Top 8 features
    else:
        top_features = numeric_cols
    
    # Create boxplots for top features
    for feature in top_features:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(x=cluster_col, y=feature, data=df, palette='viridis', ax=ax)
        ax.set_title(f'Distribution of {feature} Across Clusters')
        ax.set_xlabel('Cluster')
        ax.set_ylabel(feature)
        
        # Add median values
        medians = df.groupby(cluster_col)[feature].median()
        for xtick, median in zip(range(len(medians)), medians):
            ax.text(xtick, median, f'{median:.2f}', horizontalalignment='center',
                   size='small', color='black', weight='semibold')
        
        # Save if output_dir provided
        if output_dir:
            save_figure(fig, f'boxplot_{feature}', output_dir)
        
        figs[f'boxplot_{feature}'] = fig
    
    # 3. Radar chart of cluster profiles
    # Calculate cluster profiles (mean of each feature by cluster)
    cluster_profiles = df.groupby(cluster_col)[top_features].mean()
    
    # Normalize the data for radar chart
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    cluster_profiles_scaled = pd.DataFrame(
        scaler.fit_transform(cluster_profiles),
        index=cluster_profiles.index,
        columns=cluster_profiles.columns
    )
    
    # Create radar chart
    fig = plt.figure(figsize=(12, 10))
    
    # Number of variables
    categories = top_features
    N = len(categories)
    
    # Create angle for each feature
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create subplot with polar projection
    ax = fig.add_subplot(111, polar=True)
    
    # Add feature labels
    plt.xticks(angles[:-1], categories, size=12)
    
    # Draw y-axis labels
    ax.set_rlabel_position(0)
    plt.yticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"], color="grey", size=10)
    plt.ylim(0, 1)
    
    # Plot each cluster
    for i, cluster in enumerate(cluster_profiles_scaled.index):
        values = cluster_profiles_scaled.loc[cluster].values.tolist()
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=f"Cluster {cluster}")
        ax.fill(angles, values, alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Cluster Profiles', size=20, y=1.1)
    
    # Save if output_dir provided
    if output_dir:
        save_figure(fig, 'radar_chart', output_dir)
    
    figs['radar_chart'] = fig
    
    return figs

# Composite visualization functions that combine multiple plots
def generate_classification_report(
    X: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: Dict[str, np.ndarray],
    feature_importances: Optional[np.ndarray] = None,
    model_metrics: Optional[Dict[str, float]] = None,
    class_names: Optional[List[str]] = None,
    output_dir: str = None
) -> Dict[str, Any]:
    """
    Generate comprehensive classification report with visualizations.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
    y_true : numpy.ndarray
        True labels
    y_pred : numpy.ndarray
        Predicted labels
    y_probs : dict
        Dictionary of model names to predicted probabilities
    feature_importances : numpy.ndarray, optional
        Feature importance values (if available)
    model_metrics : dict, optional
        Dictionary of metric names to values
    class_names : list of str, optional
        Names of classes
    output_dir : str, optional
        Directory to save figures
        
    Returns:
    --------
    report : dict
        Dictionary containing report components
    """
    if output_dir:
        output_dir = create_output_dir(output_dir, 'classification_report')
    
    report = {}
    
    # 1. Confusion Matrix
    if class_names is None:
        class_names = [str(i) for i in np.unique(y_true)]
    
    cm = confusion_matrix(y_true, y_pred)
    fig_cm = plot_confusion_matrix(
        cm, 
        classes=class_names,
        normalize=True,
        title="Normalized Confusion Matrix",
        output_dir=output_dir,
        filename="confusion_matrix_norm"
    )
    report['confusion_matrix'] = fig_cm
    
    # 2. ROC Curves (for binary classification)
    if len(class_names) == 2:
        fig_roc = plot_roc_curves(
            y_true,
            y_probs,
            title="ROC Curves",
            output_dir=output_dir,
            filename="roc_curves"
        )
        report['roc_curves'] = fig_roc
        
        # 3. Precision-Recall Curves
        fig_pr = plot_precision_recall_curves(
            y_true,
            y_probs,
            title="Precision-Recall Curves",
            output_dir=output_dir,
            filename="precision_recall_curves"
        )
        report['precision_recall_curves'] = fig_pr
    
    # 4. Feature Importance (if available)
    if feature_importances is not None:
        fig_imp = plot_feature_importance(
            feature_names=X.columns.tolist(),
            importances=feature_importances,
            title="Feature Importance",
            output_dir=output_dir,
            filename="feature_importance"
        )
        report['feature_importance'] = fig_imp
    
    # 5. Metrics Summary Table (if available)
    if model_metrics is not None:
        # Create a figure for the metrics table
        fig_metrics = plt.figure(figsize=(10, 6))
        ax = fig_metrics.add_subplot(111)
        
        # Hide axes
        ax.axis('off')
        ax.axis('tight')
        
        # Create table
        metrics_table = ax.table(
            cellText=[[f"{value:.4f}" for value in model_metrics.values()]],
            colLabels=list(model_metrics.keys()),
            loc='center'
        )
        
        # Modify table
        metrics_table.auto_set_font_size(False)
        metrics_table.set_fontsize(12)
        metrics_table.scale(1.2, 1.5)
        
        fig_metrics.suptitle('Model Performance Metrics', fontsize=16)
        fig_metrics.tight_layout()
        
        # Save if output_dir provided
        if output_dir:
            save_figure(fig_metrics, 'metrics_summary', output_dir)
        
        report['metrics_summary'] = fig_metrics
    
    return report

def generate_clustering_report(
    X: pd.DataFrame,
    labels: np.ndarray,
    cluster_metrics: Optional[Dict[str, float]] = None,
    output_dir: str = None
) -> Dict[str, Any]:
    """
    Generate comprehensive clustering report with visualizations.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
    labels : numpy.ndarray
        Cluster labels
    cluster_metrics : dict, optional
        Dictionary of metric names to values
    output_dir : str, optional
        Directory to save figures
        
    Returns:
    --------
    report : dict
        Dictionary containing report components
    """
    if output_dir:
        output_dir = create_output_dir(output_dir, 'clustering_report')
    
    report = {}
    
    # 1. PCA Visualization
    df_with_clusters = X.copy()
    df_with_clusters['Cluster'] = labels
    
    figs_clusters = visualize_clusters(
        df=df_with_clusters,
        cluster_col='Cluster',
        plot_type='all',
        output_dir=output_dir
    )
    report.update(figs_clusters)
    
    # 2. Silhouette Analysis
    fig_silhouette = plot_silhouette_analysis(
        X=X.values,
        labels=labels,
        title="Silhouette Analysis",
        output_dir=output_dir,
        filename="silhouette_analysis"
    )
    report['silhouette_analysis'] = fig_silhouette
    
    # 3. Cluster Profiles Dashboard
    figs_dashboard = create_customer_segment_dashboard(
        df=df_with_clusters,
        cluster_col='Cluster',
        output_dir=output_dir
    )
    report.update(figs_dashboard)
    
    # 4. Metrics Summary Table (if available)
    if cluster_metrics is not None:
        # Create a figure for the metrics table
        fig_metrics = plt.figure(figsize=(10, 6))
        ax = fig_metrics.add_subplot(111)
        
        # Hide axes
        ax.axis('off')
        ax.axis('tight')
        
        # Create table
        metrics_table = ax.table(
            cellText=[[f"{value:.4f}" for value in cluster_metrics.values()]],
            colLabels=list(cluster_metrics.keys()),
            loc='center'
        )
        
        # Modify table
        metrics_table.auto_set_font_size(False)
        metrics_table.set_fontsize(12)
        metrics_table.scale(1.2, 1.5)
        
        fig_metrics.suptitle('Clustering Performance Metrics', fontsize=16)
        fig_metrics.tight_layout()
        
        # Save if output_dir provided
        if output_dir:
            save_figure(fig_metrics, 'metrics_summary', output_dir)
        
        report['metrics_summary'] = fig_metrics
    
    return report 