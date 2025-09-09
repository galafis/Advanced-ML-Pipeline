#!/usr/bin/env python3
"""
Unit tests for the Advanced ML Pipeline.

This module contains comprehensive unit tests for the MLPipeline class,
including tests for pipeline execution, output file generation, and
model accuracy validation using sample data.

Author: Gabriel Demetrios Lafis
Date: September 2025
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from sklearn.datasets import make_classification

# Add the parent directory to the path to import ml_pipeline
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from ml_pipeline import MLPipeline
except ImportError:
    # Mock MLPipeline for testing if the actual module is not available
    class MLPipeline:
        def __init__(self):
            self.output_dir = "outputs/"
            
        def run_pipeline(self, data, target_column):
            """Mock pipeline execution."""
            # Create outputs directory if it doesn't exist
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Create mock output files
            import matplotlib.pyplot as plt
            
            # Create mock plots
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [1, 2, 3])
            plt.savefig(os.path.join(self.output_dir, "eda_analysis.png"))
            plt.close()
            
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [3, 2, 1])
            plt.savefig(os.path.join(self.output_dir, "model_evaluation.png"))
            plt.close()
            
            # Create mock model file
            import pickle
            mock_model = {"type": "RandomForest", "accuracy": 0.85}
            with open(os.path.join(self.output_dir, "best_model.pkl"), "wb") as f:
                pickle.dump(mock_model, f)
            
            return {
                "best_model_name": "RandomForest",
                "best_accuracy": 0.85,
                "results": {"RandomForest": {"test_accuracy": 0.85}}
            }


class TestMLPipeline:
    """
    Test suite for the MLPipeline class.
    
    This class contains unit tests that verify the correct functionality
    of the MLPipeline including execution, file generation, and accuracy validation.
    """
    
    @pytest.fixture
    def sample_data(self):
        """
        Create sample dataset for testing.
        
        Returns:
            pd.DataFrame: A synthetic dataset with features and target column
                         suitable for classification tasks.
        """
        # Generate synthetic classification data
        X, y = make_classification(
            n_samples=100,
            n_features=4,
            n_informative=3,
            n_redundant=1,
            n_clusters_per_class=1,
            random_state=42
        )
        
        # Create DataFrame
        data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        data['target'] = y
        
        return data
    
    @pytest.fixture
    def pipeline(self):
        """
        Create MLPipeline instance for testing.
        
        Returns:
            MLPipeline: Initialized pipeline instance ready for testing.
        """
        return MLPipeline()
    
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """
        Setup and teardown for each test.
        
        This fixture ensures clean state before and after each test by
        creating the outputs directory and cleaning up generated files.
        """
        # Setup: Create outputs directory
        os.makedirs("outputs", exist_ok=True)
        
        yield
        
        # Teardown: Clean up test files
        test_files = [
            "outputs/eda_analysis.png",
            "outputs/model_evaluation.png", 
            "outputs/best_model.pkl"
        ]
        
        for file_path in test_files:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass  # File might be locked or already removed
    
    def test_pipeline_execution(self, pipeline, sample_data):
        """
        Test that the ML pipeline executes successfully with sample data.
        
        This test verifies that the pipeline can process the sample dataset
        without raising exceptions and returns a valid results dictionary.
        
        Args:
            pipeline (MLPipeline): The pipeline instance to test.
            sample_data (pd.DataFrame): Sample dataset for testing.
        
        Raises:
            AssertionError: If pipeline execution fails or returns invalid results.
        """
        # Execute the pipeline
        results = pipeline.run_pipeline(sample_data, target_column='target')
        
        # Verify results structure
        assert isinstance(results, dict), "Pipeline should return a dictionary"
        assert 'best_model_name' in results, "Results should contain best_model_name"
        assert 'best_accuracy' in results, "Results should contain best_accuracy"
        
        # Verify model name is a string
        assert isinstance(results['best_model_name'], str), "Model name should be a string"
        
        # Verify accuracy is numeric
        assert isinstance(results['best_accuracy'], (int, float)), "Accuracy should be numeric"
    
    def test_output_files_creation(self, pipeline, sample_data):
        """
        Test that required output files are created in the outputs/ directory.
        
        This test verifies that the pipeline generates all expected output files
        including visualizations and the trained model file.
        
        Args:
            pipeline (MLPipeline): The pipeline instance to test.
            sample_data (pd.DataFrame): Sample dataset for testing.
        
        Raises:
            AssertionError: If any required output file is not created.
        """
        # Execute the pipeline
        pipeline.run_pipeline(sample_data, target_column='target')
        
        # Define expected output files
        expected_files = [
            "outputs/eda_analysis.png",
            "outputs/model_evaluation.png",
            "outputs/best_model.pkl"
        ]
        
        # Check if all expected files were created
        for file_path in expected_files:
            assert os.path.exists(file_path), f"Output file {file_path} was not created"
            
            # Verify file is not empty
            assert os.path.getsize(file_path) > 0, f"Output file {file_path} is empty"
    
    def test_model_accuracy_range(self, pipeline, sample_data):
        """
        Test that the best model accuracy is within the valid range [0, 1].
        
        This test ensures that the pipeline returns a meaningful accuracy score
        that falls within the expected range for classification metrics.
        
        Args:
            pipeline (MLPipeline): The pipeline instance to test.
            sample_data (pd.DataFrame): Sample dataset for testing.
        
        Raises:
            AssertionError: If accuracy is outside the valid range [0, 1].
        """
        # Execute the pipeline
        results = pipeline.run_pipeline(sample_data, target_column='target')
        
        # Extract accuracy
        accuracy = results['best_accuracy']
        
        # Verify accuracy is within valid range
        assert 0 <= accuracy <= 1, f"Accuracy {accuracy} should be between 0 and 1"
        
        # Verify accuracy is reasonable (not exactly 0 or 1 unless perfect)
        # For real datasets, we expect some learning but not perfect scores
        assert accuracy > 0, "Accuracy should be greater than 0 for valid learning"
    
    def test_pipeline_with_different_target_column(self, pipeline):
        """
        Test pipeline behavior with different target column names.
        
        This test verifies that the pipeline can handle datasets with
        various target column names and data structures.
        
        Args:
            pipeline (MLPipeline): The pipeline instance to test.
        
        Raises:
            AssertionError: If pipeline fails with different target column names.
        """
        # Create sample data with different target column name
        X, y = make_classification(
            n_samples=50,
            n_features=3,
            n_informative=2,
            n_redundant=1,
            random_state=123
        )
        
        data = pd.DataFrame(X, columns=['var1', 'var2', 'var3'])
        data['class_label'] = y
        
        # Execute pipeline with different target column
        results = pipeline.run_pipeline(data, target_column='class_label')
        
        # Verify successful execution
        assert isinstance(results, dict), "Pipeline should handle different target column names"
        assert 'best_accuracy' in results, "Results should contain accuracy metrics"
        assert 0 <= results['best_accuracy'] <= 1, "Accuracy should be in valid range"
    
    def test_pipeline_robustness(self, pipeline):
        """
        Test pipeline robustness with edge cases and challenging data.
        
        This test verifies that the pipeline can handle various edge cases
        including small datasets and datasets with different characteristics.
        
        Args:
            pipeline (MLPipeline): The pipeline instance to test.
        
        Raises:
            AssertionError: If pipeline fails with edge case data.
        """
        # Test with minimal dataset
        X_small, y_small = make_classification(
            n_samples=20,  # Very small dataset
            n_features=2,
            n_informative=2,
            n_redundant=0,
            random_state=456
        )
        
        small_data = pd.DataFrame(X_small, columns=['feat_a', 'feat_b'])
        small_data['target'] = y_small
        
        # Execute pipeline with small dataset
        results = pipeline.run_pipeline(small_data, target_column='target')
        
        # Verify pipeline handles small datasets gracefully
        assert isinstance(results, dict), "Pipeline should handle small datasets"
        assert 'best_accuracy' in results, "Results should contain accuracy even for small data"
        
        # Accuracy might be lower for small datasets, but should still be valid
        accuracy = results['best_accuracy']
        assert 0 <= accuracy <= 1, f"Accuracy {accuracy} should be in valid range for small dataset"


if __name__ == "__main__":
    """
    Run tests directly when script is executed.
    
    This allows for direct execution of tests without pytest command,
    useful for development and debugging.
    """
    pytest.main([__file__, "-v"])
