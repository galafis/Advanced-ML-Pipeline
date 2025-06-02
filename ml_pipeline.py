#!/usr/bin/env python3
"""
Advanced Machine Learning Pipeline
A comprehensive ML pipeline with preprocessing, feature engineering, and model comparison.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

class MLPipeline:
    """Advanced Machine Learning Pipeline for classification tasks."""
    
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42)
        }
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(f_classif, k=10)
        self.best_model = None
        self.best_score = 0
        
    def load_data(self, file_path=None, sample_data=True):
        """Load dataset for training."""
        if sample_data:
            # Generate sample dataset for demonstration
            np.random.seed(42)
            n_samples = 1000
            n_features = 15
            
            X = np.random.randn(n_samples, n_features)
            # Create some correlation between features and target
            y = (X[:, 0] + X[:, 1] * 0.5 + X[:, 2] * 0.3 + np.random.randn(n_samples) * 0.1 > 0).astype(int)
            
            feature_names = [f'feature_{i}' for i in range(n_features)]
            self.data = pd.DataFrame(X, columns=feature_names)
            self.data['target'] = y
            
            print(f"Generated sample dataset with {n_samples} samples and {n_features} features")
        else:
            self.data = pd.read_csv(file_path)
            print(f"Loaded dataset from {file_path}")
            
        return self.data
    
    def exploratory_analysis(self):
        """Perform exploratory data analysis."""
        print("=== Exploratory Data Analysis ===")
        print(f"Dataset shape: {self.data.shape}")
        print(f"Missing values: {self.data.isnull().sum().sum()}")
        print(f"Target distribution:\n{self.data['target'].value_counts()}")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Target distribution
        self.data['target'].value_counts().plot(kind='bar', ax=axes[0,0], title='Target Distribution')
        
        # Correlation heatmap
        corr_matrix = self.data.corr()
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', ax=axes[0,1], title='Feature Correlation')
        
        # Feature distributions
        self.data.iloc[:, :5].hist(ax=axes[1,0], bins=20, alpha=0.7)
        axes[1,0].set_title('Feature Distributions (First 5)')
        
        # Box plot for outlier detection
        self.data.iloc[:, :5].boxplot(ax=axes[1,1])
        axes[1,1].set_title('Outlier Detection (First 5 Features)')
        
        plt.tight_layout()
        plt.savefig('eda_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("EDA visualizations saved as 'eda_analysis.png'")
    
    def preprocess_data(self):
        """Preprocess the data for training."""
        print("=== Data Preprocessing ===")
        
        # Separate features and target
        X = self.data.drop('target', axis=1)
        y = self.data['target']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Feature selection
        self.X_train_selected = self.feature_selector.fit_transform(self.X_train_scaled, self.y_train)
        self.X_test_selected = self.feature_selector.transform(self.X_test_scaled)
        
        selected_features = self.feature_selector.get_support()
        feature_names = X.columns[selected_features]
        
        print(f"Selected {len(feature_names)} best features: {list(feature_names)}")
        print(f"Training set shape: {self.X_train_selected.shape}")
        print(f"Test set shape: {self.X_test_selected.shape}")
    
    def train_models(self):
        """Train multiple models and compare performance."""
        print("=== Model Training and Comparison ===")
        
        self.results = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train_selected, self.y_train, cv=5, scoring='accuracy')
            
            # Train on full training set
            model.fit(self.X_train_selected, self.y_train)
            
            # Predictions
            train_pred = model.predict(self.X_train_selected)
            test_pred = model.predict(self.X_test_selected)
            
            # Calculate metrics
            train_accuracy = accuracy_score(self.y_train, train_pred)
            test_accuracy = accuracy_score(self.y_test, test_pred)
            
            self.results[name] = {
                'model': model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'test_predictions': test_pred
            }
            
            print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"  Test Accuracy: {test_accuracy:.4f}")
            
            # Update best model
            if test_accuracy > self.best_score:
                self.best_score = test_accuracy
                self.best_model = model
                self.best_model_name = name
    
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning on the best model."""
        print(f"=== Hyperparameter Tuning for {self.best_model_name} ===")
        
        if self.best_model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }
        elif self.best_model_name == 'Gradient Boosting':
            param_grid = {
                'n_estimators': [50, 100],
                'learning_rate': [0.1, 0.2],
                'max_depth': [3, 5]
            }
        else:
            print("Hyperparameter tuning not implemented for this model.")
            return
        
        grid_search = GridSearchCV(
            self.best_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1
        )
        
        grid_search.fit(self.X_train_selected, self.y_train)
        
        self.tuned_model = grid_search.best_estimator_
        tuned_pred = self.tuned_model.predict(self.X_test_selected)
        tuned_accuracy = accuracy_score(self.y_test, tuned_pred)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Tuned model accuracy: {tuned_accuracy:.4f}")
        print(f"Improvement: {tuned_accuracy - self.best_score:.4f}")
    
    def generate_report(self):
        """Generate comprehensive model evaluation report."""
        print("=== Model Evaluation Report ===")
        
        # Create comparison plot
        model_names = list(self.results.keys())
        test_scores = [self.results[name]['test_accuracy'] for name in model_names]
        cv_scores = [self.results[name]['cv_mean'] for name in model_names]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Model comparison
        x_pos = np.arange(len(model_names))
        ax1.bar(x_pos, test_scores, alpha=0.7, label='Test Accuracy')
        ax1.bar(x_pos, cv_scores, alpha=0.7, label='CV Mean')
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Performance Comparison')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(model_names, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Confusion matrix for best model
        best_pred = self.results[self.best_model_name]['test_predictions']
        cm = confusion_matrix(self.y_test, best_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
        ax2.set_title(f'Confusion Matrix - {self.best_model_name}')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print detailed report
        print(f"\nBest Model: {self.best_model_name}")
        print(f"Best Test Accuracy: {self.best_score:.4f}")
        print("\nDetailed Classification Report:")
        best_pred = self.results[self.best_model_name]['test_predictions']
        print(classification_report(self.y_test, best_pred))
        
        print("Model evaluation plots saved as 'model_evaluation.png'")
    
    def save_model(self, filename='best_model.pkl'):
        """Save the best model to disk."""
        import pickle
        
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'model_name': self.best_model_name,
            'accuracy': self.best_score
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Best model saved as '{filename}'")

def main():
    """Main execution function."""
    print("Advanced Machine Learning Pipeline")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = MLPipeline()
    
    # Load data
    data = pipeline.load_data(sample_data=True)
    
    # Perform EDA
    pipeline.exploratory_analysis()
    
    # Preprocess data
    pipeline.preprocess_data()
    
    # Train models
    pipeline.train_models()
    
    # Hyperparameter tuning
    pipeline.hyperparameter_tuning()
    
    # Generate report
    pipeline.generate_report()
    
    # Save model
    pipeline.save_model()
    
    print("\nPipeline execution completed successfully!")

if __name__ == "__main__":
    main()

