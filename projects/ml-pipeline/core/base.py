"""
Base module containing core components for the ML pipeline.
"""

import os
import sys
import logging
import time
import traceback
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
import yaml
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class MLPipelineError(Exception):
    """Base exception class for ML pipeline errors."""
    pass

class DataError(MLPipelineError):
    """Exception raised for errors in data loading or processing."""
    pass

class ModelError(MLPipelineError):
    """Exception raised for errors in model training or inference."""
    pass

class ParameterError(MLPipelineError):
    """Exception raised for invalid parameters."""
    pass

class ResourceError(MLPipelineError):
    """Exception raised for resource-related errors (memory, disk, etc.)."""
    pass

class MLComponent(ABC):
    """
    Abstract base class for ML pipeline components.
    
    All pipeline components (preprocessing, feature selection, modeling)
    should inherit from this class.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize component with configuration.
        
        Parameters:
        -----------
        config : dict
            Component configuration
        """
        self.config = config
        self.validate_config()
    
    @abstractmethod
    def validate_config(self):
        """Validate component configuration."""
        pass
    
    @abstractmethod
    def run(self, *args, **kwargs):
        """Run component's main functionality."""
        pass
    
    def log_execution_time(func):
        """Decorator to log execution time of component methods."""
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            try:
                logger.info(f"Starting {self.__class__.__name__}.{func.__name__}")
                result = func(self, *args, **kwargs)
                end_time = time.time()
                duration = end_time - start_time
                logger.info(f"Completed {self.__class__.__name__}.{func.__name__} in {duration:.2f} seconds")
                return result
            except Exception as e:
                logger.error(f"Error in {self.__class__.__name__}.{func.__name__}: {str(e)}")
                if hasattr(self, 'config') and self.config.get('debug', False):
                    logger.error(traceback.format_exc())
                raise
        return wrapper

class Pipeline(ABC):
    """
    Abstract base class for ML pipelines.
    
    This class defines the interface for complete ML pipelines 
    (classification, clustering, etc.)
    """
    
    def __init__(self, config_path: str = None, config: Dict[str, Any] = None):
        """
        Initialize pipeline with configuration.
        
        Parameters:
        -----------
        config_path : str, optional
            Path to YAML configuration file
        config : dict, optional
            Configuration dictionary (alternative to config_path)
        """
        if config_path:
            self.config = self._load_config(config_path)
        elif config:
            self.config = config
        else:
            raise ParameterError("Either config_path or config must be provided")
        
        self.output_dir = self._create_output_dir()
        self._setup_logging()
        self.components = {}
        self.results = {}
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Parameters:
        -----------
        config_path : str
            Path to YAML configuration file
            
        Returns:
        --------
        config : dict
            Configuration dictionary
        """
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            raise ParameterError(f"Error loading configuration from {config_path}: {str(e)}")
    
    def _create_output_dir(self) -> str:
        """
        Create output directory for results.
        
        Returns:
        --------
        output_dir : str
            Path to output directory
        """
        output_dir = self.config.get('reporting', {}).get('output_dir')
        
        if not output_dir:
            # Create timestamped directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"results_{timestamp}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save configuration to output directory
        config_path = os.path.join(output_dir, 'config.yaml')
        with open(config_path, 'w') as file:
            yaml.dump(self.config, file)
        
        return output_dir
    
    def _setup_logging(self):
        """Set up logging based on configuration."""
        log_level_str = self.config.get('reporting', {}).get('log_level', 'INFO')
        log_level = getattr(logging, log_level_str.upper())
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.output_dir, 'pipeline.log')),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    @abstractmethod
    def load_data(self) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Load data for the pipeline.
        
        Returns:
        --------
        data : tuple
            Tuple containing (X, y) where y may be None for unsupervised tasks
        """
        pass
    
    @abstractmethod
    def preprocess_data(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Preprocess data for the pipeline.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Feature matrix
        y : pandas.Series, optional
            Target variable
            
        Returns:
        --------
        processed_data : tuple
            Tuple containing (X_processed, y_processed)
        """
        pass
    
    @abstractmethod
    def select_features(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Select features for the pipeline.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Feature matrix
        y : pandas.Series, optional
            Target variable
            
        Returns:
        --------
        X_selected : pandas.DataFrame
            Selected features
        """
        pass
    
    @abstractmethod
    def train_models(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Train models for the pipeline.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Feature matrix
        y : pandas.Series, optional
            Target variable
            
        Returns:
        --------
        models : dict
            Trained models and related information
        """
        pass
    
    @abstractmethod
    def evaluate_models(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Evaluate trained models.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Feature matrix
        y : pandas.Series, optional
            Target variable
            
        Returns:
        --------
        evaluation : dict
            Evaluation results
        """
        pass
    
    @abstractmethod
    def visualize_results(self) -> Dict[str, Any]:
        """
        Create visualizations for pipeline results.
        
        Returns:
        --------
        visualizations : dict
            Created visualizations
        """
        pass
    
    @abstractmethod
    def generate_report(self) -> str:
        """
        Generate final report for the pipeline.
        
        Returns:
        --------
        report_path : str
            Path to generated report
        """
        pass
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete pipeline.
        
        Returns:
        --------
        results : dict
            Pipeline results
        """
        try:
            logger.info(f"Starting {self.__class__.__name__} pipeline")
            start_time = time.time()
            
            # Load data
            logger.info("Loading data")
            X, y = self.load_data()
            self.results['data'] = {'X': X, 'y': y}
            
            # Preprocess data
            logger.info("Preprocessing data")
            X_processed, y_processed = self.preprocess_data(X, y)
            self.results['preprocessed'] = {'X': X_processed, 'y': y_processed}
            
            # Select features
            if self.config.get('feature_selection', {}).get('enabled', True):
                logger.info("Selecting features")
                X_selected = self.select_features(X_processed, y_processed)
                self.results['features'] = {'X': X_selected}
            else:
                logger.info("Skipping feature selection")
                X_selected = X_processed
                self.results['features'] = {'X': X_selected}
            
            # Train models
            logger.info("Training models")
            model_results = self.train_models(X_selected, y_processed)
            self.results['models'] = model_results
            
            # Evaluate models
            logger.info("Evaluating models")
            evaluation_results = self.evaluate_models(X_selected, y_processed)
            self.results['evaluation'] = evaluation_results
            
            # Visualize results
            if self.config.get('reporting', {}).get('generate_visualizations', True):
                logger.info("Creating visualizations")
                visualization_results = self.visualize_results()
                self.results['visualizations'] = visualization_results
            
            # Generate report
            if self.config.get('reporting', {}).get('generate_summary', True):
                logger.info("Generating report")
                report_path = self.generate_report()
                self.results['report'] = report_path
            
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"Pipeline completed in {duration:.2f} seconds")
            
            return self.results
        
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
            if self.config.get('debug', False):
                logger.error(traceback.format_exc())
            raise

class Preprocessor(MLComponent):
    """Base class for data preprocessing components."""
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'Preprocessor':
        """
        Fit preprocessor to data.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Feature matrix
        y : pandas.Series, optional
            Target variable
            
        Returns:
        --------
        self : Preprocessor
            Fitted preprocessor
        """
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Feature matrix
            
        Returns:
        --------
        X_transformed : pandas.DataFrame
            Transformed feature matrix
        """
        pass
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit preprocessor to data and transform it.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Feature matrix
        y : pandas.Series, optional
            Target variable
            
        Returns:
        --------
        X_transformed : pandas.DataFrame
            Transformed feature matrix
        """
        return self.fit(X, y).transform(X)

class FeatureSelector(MLComponent):
    """Base class for feature selection components."""
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureSelector':
        """
        Fit feature selector to data.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Feature matrix
        y : pandas.Series, optional
            Target variable
            
        Returns:
        --------
        self : FeatureSelector
            Fitted feature selector
        """
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted feature selector.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Feature matrix
            
        Returns:
        --------
        X_selected : pandas.DataFrame
            Selected features
        """
        pass
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit feature selector to data and transform it.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Feature matrix
        y : pandas.Series, optional
            Target variable
            
        Returns:
        --------
        X_selected : pandas.DataFrame
            Selected features
        """
        return self.fit(X, y).transform(X)

class Model(MLComponent):
    """Base class for machine learning models."""
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'Model':
        """
        Fit model to data.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Feature matrix
        y : pandas.Series, optional
            Target variable
            
        Returns:
        --------
        self : Model
            Fitted model
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using fitted model.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Feature matrix
            
        Returns:
        --------
        predictions : numpy.ndarray
            Model predictions
        """
        pass
    
    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Feature matrix
        y : pandas.Series
            True target values
            
        Returns:
        --------
        metrics : dict
            Evaluation metrics
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> str:
        """
        Save model to disk.
        
        Parameters:
        -----------
        path : str
            Directory to save model
            
        Returns:
        --------
        model_path : str
            Path to saved model
        """
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'Model':
        """
        Load model from disk.
        
        Parameters:
        -----------
        path : str
            Path to saved model
            
        Returns:
        --------
        model : Model
            Loaded model
        """
        pass

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Parameters:
    -----------
    config_path : str
        Path to YAML configuration file
        
    Returns:
    --------
    config : dict
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        raise ParameterError(f"Error loading configuration from {config_path}: {str(e)}")

def create_pipeline(config_path: str = None, config: Dict[str, Any] = None) -> Pipeline:
    """
    Create pipeline instance based on configuration.
    
    Parameters:
    -----------
    config_path : str, optional
        Path to YAML configuration file
    config : dict, optional
        Configuration dictionary (alternative to config_path)
        
    Returns:
    --------
    pipeline : Pipeline
        Pipeline instance
    """
    if config_path:
        config = load_config(config_path)
    elif not config:
        raise ParameterError("Either config_path or config must be provided")
    
    task_type = config.get('task', {}).get('type', '').lower()
    
    if task_type == 'classification':
        from core.classification import ClassificationPipeline
        return ClassificationPipeline(config=config)
    elif task_type == 'clustering':
        from core.clustering import ClusteringPipeline
        return ClusteringPipeline(config=config)
    else:
        raise ParameterError(f"Unsupported task type: {task_type}")

def memory_usage_info() -> Dict[str, float]:
    """
    Get memory usage information.
    
    Returns:
    --------
    memory_info : dict
        Memory usage information
    """
    import psutil
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        'rss': memory_info.rss / 1024**2,  # RSS in MB
        'vms': memory_info.vms / 1024**2,  # VMS in MB
        'percent': process.memory_percent(),
        'available': psutil.virtual_memory().available / 1024**2  # Available memory in MB
    }

def check_available_memory(required_mb: float) -> bool:
    """
    Check if there is enough available memory.
    
    Parameters:
    -----------
    required_mb : float
        Required memory in MB
        
    Returns:
    --------
    is_available : bool
        Whether required memory is available
    """
    import psutil
    
    available_mb = psutil.virtual_memory().available / 1024**2
    return available_mb >= required_mb

def batch_process(func, data, batch_size: int, *args, **kwargs):
    """
    Process data in batches to manage memory usage.
    
    Parameters:
    -----------
    func : callable
        Function to apply to each batch
    data : pandas.DataFrame or numpy.ndarray
        Data to process
    batch_size : int
        Size of batches
    *args, **kwargs
        Additional arguments to pass to func
        
    Returns:
    --------
    results : list
        List of results from each batch
    """
    results = []
    total_size = len(data)
    
    for i in range(0, total_size, batch_size):
        batch = data[i:min(i + batch_size, total_size)]
        batch_result = func(batch, *args, **kwargs)
        results.append(batch_result)
    
    return results 