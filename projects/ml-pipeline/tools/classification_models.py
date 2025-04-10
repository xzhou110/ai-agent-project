import os
import time
import datetime
import pickle
import concurrent.futures
import pandas as pd
import numpy as np
from matplotlib import (cm, colors as mcolors)
# from matplotlib import colors as mcolors
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc)
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt


def run_classification_models(X_train, y_train, X_test, y_test, n_samples=None, search_type=None, scoring_metric='roc auc'):
    """
    Train and evaluate multiple classification models on the given dataset.

    This function trains and evaluate five different classification models: Logistic Regression, K-Nearest Neighbors,
    Support Vector Machine, Random Forest, and XGBoost. It also performs hyperparameter tuning if specified and plots
    ROC curves for all models. It evaluates model performance based on user defined scoring metric. It saves the best 
    performing model into results folder.

    Parameters
    ----------
    X_train : pd.DataFrame
        The training feature matrix (X) as a pandas DataFrame.
    y_train : pd.Series
        The target variable for training (y) as a pandas Series.
    X_test : pd.DataFrame
        The testing feature matrix (X) as a pandas DataFrame.
    y_test : pd.Series
        The target variable for testing (y) as a pandas Series.
    n_samples : int, optional, default: None
        Number of samples to downsample the data to. If None, no downsampling is performed.
    search_type : str, optional, default: None
        The type of hyperparameter search to perform. Choices are 'grid', 'random', or None. Default is None.
    scoring_metric : str, optional, default: 'roc auc'
        The scoring metric to optimize during hyperparameter tuning and to select the best model. 
        Some options are 'accuracy', 'precision', 'recall', 'f1 score', and 'roc auc'.

    Returns
    -------
    best_model_instance : object
        The best performing model instance (fitted) according to the highest score of the scoring_metric.
    """
    
    # Handling class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    # Downsample data if n_samples is provided
    if n_samples is not None:
        X_train, y_train = downsample_data(X_train, y_train, target_records=n_samples)
    
    # Hyperparameter grids
    lr_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'max_iter': [500, 1000, 3000]}
    knn_grid = {'n_neighbors': list(range(1, 31))}
    svm_grid = {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10]}
    rf_grid = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}
    xgb_grid = {'learning_rate': [0.01, 0.1], 'n_estimators': [100, 200], 'max_depth': [3, 6]}

    # Create a list of models to iterate through
    models = [
        ('Logistic Regression', LogisticRegression(random_state=42, max_iter=1000), lr_grid),
        ('KNN', KNeighborsClassifier(), knn_grid),
        ('SVM', SVC(random_state=42, probability=True), svm_grid),
        ('Random Forest', RandomForestClassifier(random_state=42), rf_grid),
        ('XGBoost', XGBClassifier(random_state=42, use_label_encoder=False), xgb_grid)
    ]
    
    # Pre-process the scoring_metric argument
    scoring_metric = scoring_metric.title() if scoring_metric.lower() != 'roc auc' else 'ROC AUC'
    model_performance = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'])
    
    performance_metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    if scoring_metric not in performance_metrics:
        raise ValueError(f"Invalid scoring_metric. Expected one of: {performance_metrics}")
        
    best_model_obj = None
    best_model_performance = None
    
    plt.figure(figsize=(16, 9))
    ax = plt.gca()

    # Iterate through the models and evaluate their performance
    for model_name, model, grid in models:
        start_time = time.time()
        model = fit_model(search_type, model, grid, X_train, y_train, scoring_metric)
        exec_time = time.time() - start_time
        print(f"{model_name}: Done (Execution Time: {exec_time:.2f} seconds)")

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Train and evaluate the model
        accuracy, precision, recall, f1, roc_auc = train_and_evaluate_model(model, X_train, y_train, X_test, y_test)
        performance_metrics_values = [accuracy, precision, recall, f1, roc_auc]
        
        # Append model performance metrics to the model_performance DataFrame
        # Using pd.concat instead of DataFrame.append to avoid the FutureWarning
        performance_df = pd.DataFrame({
            'Model': [model_name],
            'Accuracy': [accuracy],
            'Precision': [precision],
            'Recall': [recall],
            'F1 Score': [f1],
            'ROC AUC': [roc_auc]
        })
        model_performance = pd.concat([model_performance, performance_df], ignore_index=True)
        
         # Update the best performing model
        if best_model_performance is None or performance_metrics_values[performance_metrics.index(scoring_metric)] > best_model_performance[scoring_metric]:
            best_model_obj = model
            best_model_performance = model_performance.iloc[-1]

        # Plot ROC curve
        plot_roc_curve(model_name, y_test, y_pred_proba, ax)

    # Display ROC curve comparison
    plt.plot([0, 1], [0, 1], 'k--', label="Random (AUC = 0.5)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title("ROC Curve Comparison", fontsize=16, weight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.show()

    # Display model performance metrics
    print("Model Performance Metrics:\n")
    print(model_performance)

    # Print the best performing model
    print("\nBest Performing Model:\n")
    print(best_model_performance)

    # Print the best hyperparameters for the best performing model
    if search_type in ["grid", "random"]:
        print("\nBest Hyperparameters for the Best Performing Model:\n")
        print(best_model_obj.best_params_)
        
    # Pick the best performing model
    best_model_idx = model_performance['ROC AUC'].idxmax()
    best_model = model_performance.loc[best_model_idx]
    print("\nBest Performing Model:\n")
    print(best_model)
    
    # Save the best performing model to a pickle file
    best_model_name, best_model_instance, _ = models[best_model_idx]
    if search_type in ["grid", "random"]:
        best_model_instance = best_model_obj.best_estimator_
        
    # Create a timestamped filename
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create result directory if not exists
    result_dir = "./results"
    os.makedirs(result_dir, exist_ok=True)
    
    with open(f"./{result_dir}/{best_model_name}_best_model_{current_time}.pkl", "wb") as file:
        pickle.dump(best_model_instance, file)
    print(f"\nSaved best model ({best_model_name}) to a pickle file.")
    
    # Return the best performing model object for future reuse
    return best_model_instance


def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    return accuracy, precision, recall, f1, roc_auc


def downsample_data(X, y, target_records=10000, random_state=42):
    if len(X) <= target_records:
        return X, y
    
    combined_data = pd.concat([X, y], axis=1)
    downsampled_data = combined_data.sample(n=target_records, random_state=random_state)
    X_downsampled = downsampled_data.iloc[:, :-1]
    y_downsampled = downsampled_data.iloc[:, -1]

    return X_downsampled, y_downsampled


def fit_model(search_type, model, grid, X_train, y_train, scoring_metric='ROC AUC'):
    # Replace spaces with underscores and convert to lowercase
    if scoring_metric.lower() == 'roc auc':
        scoring_metric = scoring_metric.replace(' ', '_').lower()
    elif scoring_metric.lower() == 'f1 score':
        scoring_metric = 'f1'
    else:
        scoring_metric = scoring_metric.lower()
    
    if search_type == "grid":
        model = GridSearchCV(model, grid,
        scoring=scoring_metric, cv=5, n_jobs=-1, refit=True)
        model.fit(X_train, y_train)
    elif search_type == "random":
        model = RandomizedSearchCV(model, grid, n_iter=10, 
        scoring=scoring_metric, cv=5, n_jobs=-1, refit=True, random_state=42)
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)
    return model


def plot_roc_curve(model_name, y_test, y_pred_proba, ax):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.4f})")
    
    
def plot_top_n_features(model, feature_names, selected_features, n=10):
    """
    Plot the top n features of importance for a given model.

    Parameters
    ----------
    model : object
        The model object.
    feature_names : list
        The names of features. X.columns should be used here.
    selected_feature : list
        The names of selected features. X.columns can be used here if no feature is selected.
    n : int, optional, default: 10
        The number of top features to plot.

    Returns
    -------
    None
    """
    try:
        importances = model.feature_importances_
    except AttributeError:
        try:
            # If the model is a linear model, use absolute value of coefficients as importances
            if isinstance(model, LogisticRegression) or (isinstance(model, SVC) and model.kernel == 'linear'):
                importances = np.abs(model.coef_[0])
            else:
                print("The model does not have 'feature_importances_' or 'coef_' attribute.")
                return
        except AttributeError:
            print("The model does not have 'feature_importances_' or 'coef_' attribute.")
            return
    
    n = len(selected_features) if n>len(selected_features) else n
    
    # Get feature importances and indices
    indices = np.argsort(importances)[::-1]
    top_n_indices = indices[:n]
    
    top_n_features = {}
    top_n_features = {feature_names[i]: importances[i] for i in top_n_indices}

    # Prepare colors. Create a custom colormap that goes from medium green to light blue.
    # cmap = mcolors.LinearSegmentedColormap.from_list("n",["#008000", "#0000CD"])
    colors = cm.summer(np.linspace(0, 1, n))[::-1]

    # Plot the feature importances of the model
    plt.figure(figsize=(12, 9))
    plt.title(f"Top {n} Feature Importances", fontsize=16,  weight='bold')
    plt.barh(range(n), importances[top_n_indices][::-1], color=colors, align='center')  # Only plot top n
    plt.yticks(range(n), [feature_names[i] for i in top_n_indices][::-1], fontsize=12)  # Only plot top n, reversed for aesthetics
    plt.xlabel("Relative Importance", fontsize=14)
    plt.grid(axis='x')
    plt.tight_layout()
    plt.show()
    
    return top_n_features
