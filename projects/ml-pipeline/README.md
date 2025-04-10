# Advanced ML Pipeline

A robust, modular machine learning pipeline supporting both classification and clustering tasks. This pipeline automates the entire ML workflow from data preprocessing to model training, evaluation, and visualization with a focus on production readiness, error handling, and memory efficiency.

## Features

- **Modular Architecture**: Highly extensible component-based design
- **Configuration-Driven**: YAML-based configuration for reproducible experiments
- **Memory Efficient**: Batch processing for large datasets
- **Error Handling**: Comprehensive error handling and reporting
- **Robust Visualization**: Advanced visualization tools for model interpretation
- **Multiple Algorithms**: Automatic selection of best performing models
- **Hyperparameter Tuning**: Grid or randomized search for optimal parameters
- **Automated Reporting**: Detailed performance reports and visualizations

## Architecture

The pipeline follows a component-based architecture:

```
ml-pipeline/
│
├── core/                           # Core pipeline components
│   ├── base.py                     # Base classes and interfaces
│   ├── preprocessing.py            # Data preprocessing components
│   ├── feature_selection.py        # Feature selection implementations
│   ├── classification.py           # Classification pipeline
│   └── clustering.py               # Clustering pipeline
│
├── tools/                          # Implementation components
│   ├── data_preprocessing.py       # Data preprocessing functions
│   ├── feature_selection.py        # Feature selection techniques
│   ├── classification_models.py    # Classification algorithms
│   ├── clustering_models.py        # Clustering algorithms
│   └── visualization.py            # Visualization utilities
│
├── data/                           # Data directory
│   ├── customer_personality_analysis.csv  # Example dataset
│   └── about-dataset.txt           # Dataset description
│
├── config.yaml                     # Default configuration
├── run_pipeline.py                 # Main entry point
├── flexible_pipeline.py            # Legacy flexible pipeline
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ml-pipeline.git
cd ml-pipeline
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run a classification task:
```bash
python run_pipeline.py --task classification --data data/customer_personality_analysis.csv --target Response
```

Run a clustering task:
```bash
python run_pipeline.py --task clustering --data data/customer_personality_analysis.csv
```

### Using the Flexible Pipeline

You can also use the `flexible_pipeline.py` script for a command-line based approach:

Run a classification task:
```bash
python flexible_pipeline.py --data data/customer_personality_analysis.csv --task classification --target Response
```

Run a clustering task:
```bash
python flexible_pipeline.py --data data/customer_personality_analysis.csv --task clustering
```

### Configuration-Based Execution

Generate a default configuration:
```bash
python run_pipeline.py --generate-config --task clustering --config-output my_config.yaml
```

Edit the configuration file and run:
```bash
python run_pipeline.py --config my_config.yaml
```

### Advanced Options

Enable debug mode for detailed error information:
```bash
python run_pipeline.py --config my_config.yaml --debug
```

Profile execution for performance analysis:
```bash
python run_pipeline.py --config my_config.yaml --profile
```

Set output directory and log level:
```bash
python run_pipeline.py --config my_config.yaml --output-dir ./results --log-level DEBUG
```

## Testing the Pipeline with Sample Data

The repository includes a sample dataset for customer segmentation analysis. You can test the pipeline with this data to ensure everything is working correctly.

### Running a Test

To run a clustering test with the sample data:

```bash
python flexible_pipeline.py --data data/customer_personality_analysis.csv --task clustering
```

### Expected Output

After running the pipeline, you should see a new directory created with a name like `results_clustering_[timestamp]`. This directory contains:

1. `clustering_summary.txt` - A summary of the clustering results
2. `data_with_clusters.csv` - The original data with cluster assignments
3. `preprocessed_data.csv` - The preprocessed data used for modeling
4. `selected_features.txt` - List of features used in the clustering
5. Visualization directories:
   - `clusters/` - Cluster visualizations including PCA plots
   - `dashboard/` - Customer segment analysis dashboard

The clustering summary will show:
- The original and preprocessed data shapes
- Selected features for clustering
- Best clustering method and optimal number of clusters
- Distribution of samples across clusters

## Configuration

The pipeline is highly configurable through YAML configuration files. Example configuration:

```yaml
# Data settings
data:
  path: "data/customer_personality_analysis.csv"
  target_column: "Response"  # Set to null for clustering tasks
  test_size: 0.2
  random_state: 42
  memory_efficient: true

# Task settings
task:
  type: "classification"  # Options: "classification", "clustering"
  
# Preprocessing settings
preprocessing:
  missing_threshold: 0.9
  max_unique_values_cat: 50
  correlation_threshold: 0.9
  handle_outliers: true
  scale_data: true

# Feature selection settings
feature_selection:
  enabled: true
  n_features: null  # null means auto-select optimal number

# Classification settings
classification:
  algorithms: ["logistic_regression", "random_forest", "xgboost", "svm"]
  metrics: ["accuracy", "precision", "recall", "f1", "roc_auc"]
  cv_folds: 5
  use_smote: true
  tune_hyperparams: true

# Reporting settings
reporting:
  save_models: true
  generate_visualizations: true
  generate_summary: true
  output_dir: null  # null means a timestamped directory will be created
  log_level: "INFO"
```

## Error Handling

The pipeline includes robust error handling with specific exception types:

- `MLPipelineError`: Base exception for all pipeline errors
- `DataError`: For data loading and processing errors
- `ModelError`: For model training and inference errors
- `ParameterError`: For invalid parameters
- `ResourceError`: For resource-related issues (memory, disk, etc.)

Errors are logged with detailed information, and in debug mode, full stack traces are provided.

## Memory Management

For large datasets, the pipeline includes memory-efficient processing:

- Batch processing for data loading and transformation
- Memory monitoring to prevent out-of-memory errors
- Configurable batch sizes for different operations

## Extending the Pipeline

The component-based architecture makes it easy to extend the pipeline:

1. Create a new component by inheriting from appropriate base classes
2. Register the component in the appropriate factory
3. Update configuration schema to include new component parameters

Example: Adding a new classification algorithm:

```python
from core.base import Model

class MyCustomModel(Model):
    def __init__(self, config):
        super().__init__(config)
        # Initialize model
        
    def fit(self, X, y):
        # Train model
        return self
        
    def predict(self, X):
        # Make predictions
        return predictions
        
    def evaluate(self, X, y):
        # Evaluate model
        return metrics
        
    def save(self, path):
        # Save model
        return model_path
        
    @classmethod
    def load(cls, path):
        # Load model
        return model
```

## Dependencies

All dependencies are listed in `requirements.txt` with specific versions for reproducibility. The main dependencies include:

- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- xgboost
- joblib
- pyyaml

## License

This project is open source and available under the MIT License. 