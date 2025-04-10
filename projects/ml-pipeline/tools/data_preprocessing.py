import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import datetime
import logging

logging.basicConfig(level=logging.INFO)

def preprocess_data(df, label='label', missing_threshold=0.9, max_unique_values_cat=50, correlation_threshold=0.9):
    """
    Preprocesses the input data for use in a machine learning model.This function performs various preprocessing steps, including: 
    1. Encode labels: Convert the target variable column into numerical values.
    2. Drop columns with missing values above a specified threshold.
    3. Process date columns: Calculate the number of days from the date to the current date and rename the columns accordingly.
    4. Handle missing values for numerical columns: Replace missing values with the median of the respective column.
    5. Handle extreme values for numerical columns: Replace values below a certain percentile with that percentile value, and values above another percentile with that percentile value.
    6. Create dummy variables for categorical columns: Convert categorical columns with a number of unique values below a specified threshold into dummy variables.
    7. Handle high correlation among numerical columns: Drop one column from each pair of highly correlated columns based on a specified correlation threshold.
    8. Standardize numerical columns: Scale the numerical columns to have a mean of 0 and a standard deviation of 1.
    9. Clean up feature names for XgBoost model: The XgBoost model has requirement that feature names can't include specific characters, such as '[]', '<', etc.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input data to preprocess.
    label : str or None, optional, default: 'label'
        The name of the target variable column in the input DataFrame.
        If None, no target column is expected (for unsupervised learning).
    missing_threshold : float, optional, default: 0.9
        The threshold for the proportion of missing values in a column. Columns with a proportion of 
        missing values above this threshold will be dropped.
    max_unique_values_cat : int, optional, default: 50
        The maximum number of unique values allowed for a categorical column. Categorical columns with 
        more unique values than this threshold will be dropped.
    correlation_threshold : float, optional, default: 0.9
        The threshold for the correlation between numerical columns. Pairs of columns with a correlation 
        above this threshold will be handled by dropping one of the columns.

    Returns:
    --------
    result_df : pandas.DataFrame
        The preprocessed data, including the target variable column if label was provided.

    """
    
    df = df.copy()
    
    # Separate out the target column and the feature columns if label is provided
    if label is not None and label in df.columns:
        y = df[label]
        X = df.drop(columns=[label])
        
        # Encoding labels and printing out the encoding
        y, label_encoding = encode_labels(y)
        logging.info(f"Label encoding: {label_encoding}")
    else:
        X = df.copy()
        y = None
        logging.info("No label column provided or found. Proceeding with unsupervised preprocessing.")

    # Dropping columns with missing values above the threshold
    X = drop_missing_value_columns(X, missing_threshold)
    logging.info("Dropped columns with missing values above the threshold")

    # Processing date columns and renaming them
    X = process_date_columns(X)
    logging.info("Processed date columns")

    # Handling missing values for numerical columns
    X = handle_missing_values_numerical(X)
    logging.info("Handled missing values for numerical columns")

    # Handling extreme values for numerical columns
    X = handle_extreme_values_numerical(X)
    logging.info("Handled extreme values for numerical columns")

    # Creating dummy variables for categorical columns
    X = create_dummy_variables(X, max_unique_values_cat)
    logging.info("Created dummy variables for categorical columns")

    # Handling high correlation among numerical columns
    X = handle_high_correlation(X, correlation_threshold)
    logging.info("Handled high correlation among numerical columns")
    
    # Standardizing numerical columns
    X = standardize_numerical_columns(X)
    logging.info("Standardized numerical columns")

    # Clean feature names for XGBoost
    X = clean_feature_names_for_xgboost(X)
    logging.info("Cleaned feature names for XGBoost")

    # Combine the processed features and the label into a single DataFrame
    X.reset_index(drop=True, inplace=True)
    
    if y is not None:
        result_df = pd.concat([X, pd.Series(y, name=label)], axis=1)
    else:
        result_df = X

    return result_df


def encode_labels(y):
    y, unique_labels = pd.factorize(y)
    label_encoding = {label: idx for idx, label in enumerate(unique_labels)}
    return y, label_encoding

def drop_missing_value_columns(X, missing_threshold):
    return X.loc[:, X.isna().mean() < missing_threshold]

def process_date_columns(X):
    date_columns = [col for col in X.columns if 'date' in col.lower()]

    for col in date_columns:
        X[col] = (datetime.datetime.now() - pd.to_datetime(X[col])).dt.days
        X.rename(columns={col: col + '_days'}, inplace=True)
    
    return X

def handle_missing_values_numerical(X):
    numerical_columns = X.select_dtypes(include=['number']).columns.tolist()

    for col in numerical_columns:
        X[col].fillna(X[col].median(), inplace=True)
    
    return X

def handle_extreme_values_numerical(X):
    numerical_columns = X.select_dtypes(include=['number']).columns.tolist()
    lower_bound = X[numerical_columns].quantile(0.05)
    upper_bound = X[numerical_columns].quantile(0.95)

    for col in numerical_columns:
        X[col] = np.where(X[col] < lower_bound[col], lower_bound[col], X[col])
        X[col] = np.where(X[col] > upper_bound[col], upper_bound[col], X[col])
    
    return X

def create_dummy_variables(X, max_unique_values_cat):
    categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
    categorical_columns_to_drop = [col for col in categorical_columns if X[col].nunique() > max_unique_values_cat]
    X.drop(columns=categorical_columns_to_drop, inplace=True)
    remaining_categorical_columns = list(set(categorical_columns) - set(categorical_columns_to_drop))
    X = pd.get_dummies(X, columns=remaining_categorical_columns, dummy_na=True, drop_first=True)
    
    return X

def handle_high_correlation(X, correlation_threshold):
    numerical_columns = X.select_dtypes(include=['number']).columns.tolist()
    corr_matrix = X[numerical_columns].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    columns_to_drop = [col for col in upper.columns if any(upper[col] > correlation_threshold)]
    X.drop(columns=columns_to_drop, inplace=True)
    return X

def standardize_numerical_columns(X):
    numerical_columns = X.select_dtypes(include=['number']).columns.tolist()
    scaler = StandardScaler()
    X[numerical_columns] = scaler.fit_transform(X[numerical_columns])
    return X

def clean_feature_names_for_xgboost(X):
    """
    Clean column names to meet the requirements of XGBoost.
    XGBoost (at least the version used at the time of writing) does not accept feature names with special characters like '<', '[' or ']'.
    This function replaces these special characters with corresponding text representations.
    """
    X.columns = X.columns.astype(str)\
                 .str.replace(r'\[', '_replace_bracket_open_', regex=True)\
                 .str.replace(r'\]', '_replace_bracket_close_', regex=True)\
                 .str.replace('<', '_smaller_than_', regex=True)
    return X


