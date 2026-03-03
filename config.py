"""
Configuration module for Social Media Analysis project.

This module contains configuration settings, file paths, and model parameters
used throughout the project. Update these settings based on your environment
and data locations.
"""

import os
from pathlib import Path

# =============================================================================
# PROJECT STRUCTURE
# =============================================================================

# Base project directory (adjust this to your actual project root)
PROJECT_ROOT = Path(__file__).parent.absolute()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

# Output directories
PLOTS_DIR = PROJECT_ROOT / "plots"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, SYNTHETIC_DATA_DIR, 
                  MODELS_DIR, CHECKPOINTS_DIR, RESULTS_DIR, PLOTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

# Pre-trained model names
MODEL_NAMES = {
    "bert_dutch": "GRoNLP/bert-base-dutch-cased",
    "deberta": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    "roberta": "DTAI-KULeuven/robbert-2023-dutch-large",
}

# Model hyperparameters
MODEL_CONFIG = {
    "max_length": 384,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "num_epochs": 5,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "seed": 42
}

# Cross-validation settings
CV_CONFIG = {
    "n_splits": 2,
    "shuffle": True,
    "random_state": 42,
    "stratify": True
}

# =============================================================================
# DATA PROCESSING CONFIGURATION
# =============================================================================

# Column names (standardize across datasets)
COLUMN_NAMES = {
    "text": "text",
    "intent": "Intent", 
    "sentiment": "sentiment",
    "timestamp": "Comments_time",
    "author": "author",
    "synthetic_data": "Synthetic Data",
    "cluster": "Cluster_KMeans"
}

# Text preprocessing settings
PREPROCESSING_CONFIG = {
    "remove_html": True,
    "normalize_unicode": True,
    "standardize_urls": True,
    "standardize_mentions": True,
    "standardize_numbers": True,
    "max_length": 512,
    "min_length": 10
}

# =============================================================================
# ANALYSIS SETTINGS
# =============================================================================

# Time period for discourse analysis
TIME_PERIOD = {
    "start_date": "2018-01-01",
    "end_date": "2023-03-31",
    "aggregation": "Q"  # Quarterly
}

# Clustering settings
CLUSTERING_CONFIG = {
    "algorithm": "kmeans",
    "n_clusters": 6,
    "random_state": 42,
    "max_iter": 300
}

# Visualization settings
PLOT_CONFIG = {
    "figure_size": (12, 8),
    "dpi": 300,
    "style": "seaborn",
    "color_palette": "Set2",
    "font_size": 12
}

# =============================================================================
# FILE PATHS (UPDATE THESE FOR YOUR ENVIRONMENT)
# =============================================================================

# Data file paths - UPDATE THESE PATHS FOR YOUR SYSTEM
DEFAULT_DATA_PATHS = {
    "synthetic_data": SYNTHETIC_DATA_DIR / "New_synthethic_data_generation.xlsx",
    "clustered_comments": PROCESSED_DATA_DIR / "Clustered_Comments_Probabilities_KMeans.csv",
    "raw_comments": RAW_DATA_DIR / "comments_raw.csv"
}

# Model save paths
MODEL_SAVE_PATHS = {
    "bert_dutch": MODELS_DIR / "bert_dutch_intent",
    "deberta": MODELS_DIR / "deberta_intent", 
    "roberta": MODELS_DIR / "roberta_intent"
}

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

import logging

def setup_logging(log_level=logging.INFO):
    """Set up logging configuration for the project."""
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logs directory if it doesn't exist
    LOGS_DIR.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(LOGS_DIR / 'social_media_analysis.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_data_path(data_key: str) -> Path:
    """Get the path for a specific data file."""
    if data_key in DEFAULT_DATA_PATHS:
        return DEFAULT_DATA_PATHS[data_key]
    else:
        raise KeyError(f"Unknown data key: {data_key}")

def get_model_path(model_key: str) -> Path:
    """Get the path for a specific model."""
    if model_key in MODEL_SAVE_PATHS:
        return MODEL_SAVE_PATHS[model_key]
    else:
        raise KeyError(f"Unknown model key: {model_key}")

def update_data_path(data_key: str, new_path: str):
    """Update a data path for your specific environment."""
    DEFAULT_DATA_PATHS[data_key] = Path(new_path)

# =============================================================================
# ENVIRONMENT CHECKS
# =============================================================================

def check_environment():
    """Check if the environment is properly set up."""
    import torch
    import transformers
    
    print("=== Environment Check ===")
    print(f"Python version: {os.sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Transformers version: {transformers.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    print(f"Project root: {PROJECT_ROOT}")
    print("=========================")

if __name__ == "__main__":
    # Run environment check when this module is executed directly
    check_environment()