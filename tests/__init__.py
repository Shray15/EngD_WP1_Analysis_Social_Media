"""
Test utilities for the Social Media Analysis project.

This module provides common testing utilities and fixtures used across
different test modules in the project.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_text_data():
    """Sample text data for testing preprocessing functions."""
    return [
        "Hello world! How are you?",
        "Visit https://example.com for more info",
        "Contact us at test@example.com",
        "@user123 thanks for sharing!",
        "The price is $29.99 or €25.50",
        "<p>HTML content</p> with &amp; entities",
        "Multiple    spaces   and\ttabs",
        "Emoji test 😀 🎉 ❤️",
        "Dutch text: Hoe gaat het met je?",
        ""  # Empty string edge case
    ]


@pytest.fixture
def sample_intent_data():
    """Sample intent classification data for testing."""
    data = {
        "text": [
            "I want to buy a new phone",
            "What time does the store close?", 
            "Thank you for your help!",
            "I'm having trouble with my order",
            "Where can I find more information?",
            "This product is amazing!",
            "I need to return this item",
            "How much does shipping cost?",
            "I love this service",
            "Can you help me with setup?"
        ],
        "intent": [
            "purchase", "information", "gratitude", "complaint", "information",
            "praise", "return", "information", "praise", "support"
        ]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_sentiment_data():
    """Sample sentiment classification data for testing."""
    data = {
        "text": [
            "I love this product! It's amazing!",
            "This is terrible, worst experience ever",
            "It's okay, nothing special",
            "Absolutely fantastic, highly recommend!",
            "Not good, disappointed with quality",
            "Average product, meets expectations",
            "Outstanding service, very impressed",
            "Poor quality, waste of money",
            "Decent value for the price",
            "Excellent customer support team"
        ],
        "sentiment": [
            "positive", "negative", "neutral", "positive", "negative",
            "neutral", "positive", "negative", "neutral", "positive"
        ]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_temporal_data():
    """Sample temporal data for discourse analysis testing."""
    dates = pd.date_range("2018-01-01", "2023-03-31", freq="M")
    data = {
        "Comments_time": np.random.choice(dates, 100),
        "text": [f"Comment {i}" for i in range(100)],
        "Cluster_KMeans": np.random.choice([0, 1, 2, 3, 4], 100)
    }
    return pd.DataFrame(data)


class MockModel:
    """Mock model class for testing purposes."""
    
    def __init__(self, model_name="test-model"):
        self.model_name = model_name
        self.is_trained = False
    
    def fit(self, X, y):
        """Mock training method."""
        self.is_trained = True
        return self
    
    def predict(self, X):
        """Mock prediction method."""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        # Return random predictions for testing
        return np.random.choice([0, 1, 2], len(X))
    
    def predict_proba(self, X):
        """Mock probability prediction method.""" 
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        n_samples = len(X)
        n_classes = 3
        # Return random probabilities that sum to 1
        probs = np.random.random((n_samples, n_classes))
        return probs / probs.sum(axis=1, keepdims=True)


def assert_valid_preprocessing(original_text, processed_text):
    """Assert that text preprocessing produces valid results."""
    # Basic validity checks
    assert isinstance(processed_text, str), "Processed text should be a string"
    assert len(processed_text.strip()) >= 0, "Processed text should not be negative length"
    
    # Check that extreme whitespace is cleaned
    assert "  " not in processed_text, "Multiple spaces should be cleaned"
    assert not processed_text.startswith(" "), "Should not start with space"
    assert not processed_text.endswith(" "), "Should not end with space"


def create_temp_data_file(tmp_path, data, filename="test_data.csv"):
    """Create a temporary data file for testing."""
    file_path = tmp_path / filename
    if filename.endswith('.csv'):
        data.to_csv(file_path, index=False)
    elif filename.endswith('.xlsx'):
        data.to_excel(file_path, index=False)
    return file_path


def assert_model_performance(y_true, y_pred, min_accuracy=0.5):
    """Assert that model performance meets minimum standards."""
    from sklearn.metrics import accuracy_score
    
    accuracy = accuracy_score(y_true, y_pred)
    assert accuracy >= min_accuracy, f"Model accuracy {accuracy:.3f} below minimum {min_accuracy}"


@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary config file for testing."""
    config_content = '''
# Test configuration
PROJECT_ROOT = r"{}"
DATA_DIR = PROJECT_ROOT / "data"
MODEL_CONFIG = {{
    "max_length": 128,
    "batch_size": 8, 
    "seed": 42
}}
'''.format(str(tmp_path))
    
    config_file = tmp_path / "test_config.py"
    config_file.write_text(config_content)
    return config_file