"""
Advanced Machine Learning Pipeline
==================================
Pipeline profissional para automação total de Machine Learning, incluindo EDA automatizada,
feature engineering, comparação e tuning de modelos, outputs em 'outputs/' e salvamento robusto.
"""

from config import ML_CONFIG, VISUALIZATION_CONFIG, HYPERPARAMETER_GRIDS
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import warnings

warnings.filterwarnings('ignore')


class MLPipeline:
    """
    Pipeline completo para tarefas de classificação em Machine Learning.
    Automatiza EDA, pré-processamento, treinamento, avaliação, tuning e geração de outputs estruturados.
    """

    def __init__(self):
        """
        Inicializa os modelos, scaler, seletor de features e variáveis de controle.
        """
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=ML_CONFIG['random_state']),
            'Gradient Boosting': GradientBoostingClassifier(random_state=ML_CONFIG['random_state']),
            'Logistic Regression': LogisticRegression(random_state=ML_CONFIG['random_state'], max_iter=1000),
            'SVM': SVC(random_state=ML_CONFIG['random_state'], probability=True) # probability=True for ROC curve
        }
        self.scaler = StandardScaler()
        self.feature_selector = None # Will be initialized in preprocess_data
        self.best_model = None
        self.best_score = 0
        self.best_model_name = None
        self.results = {}
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.X_train_selected = None
        self.X_test_selected = None
        self.feature_names_selected = None

    def load_data(self, data_frame=None, file_path=None, target_column=None, sample_data=False):
        """
        Carrega dataset para treino. Pode receber um DataFrame, um caminho de arquivo ou gerar dados de amostra.
        Define a coluna alvo.
        """
        if data_frame is not None:
            self.data = data_frame.copy()
            print(f"Loaded data from provided DataFrame with shape: {self.data.shape}")
        elif file_path:
            self.data = pd.read_csv(file_path)
            print(f"Loaded dataset from {file_path} with shape: {self.data.shape}")
        elif sample_data:
            np.random.seed(ML_CONFIG['random_state'])
            n_samples = 1000
            n_features = 15
            X = np.random.randn(n_samples, n_features)
            y = (X[:, 0] + X[:, 1] * 0.5 + X[:, 2] * 0.3 + np.random.randn(n_samples) * 0.1 > 0).astype(int)
            feature_names = [f'feature_{i}' for i in range(n_features)]
            self.data = pd.DataFrame(X, columns=feature_names)
            self.data['target'] = y
            target_column = 'target'
            print(f"Generated sample dataset with {n_samples} samples and {n_features} features")
        else:
            raise ValueError("Must provide either a DataFrame, a file_path, or set sample_data=True.")

        if target_column is None or target_column not in self.data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data.")
        self.target_column = target_column
        print(f"Target column set to: {self.target_column}")
        return self.data

    def exploratory_analysis(self):
        """
        Executa análise exploratória nos dados, gera gráficos EDA e salva em outputs/.
        """
        print("=== Exploratory Data Analysis ===")
        if self.data is None: raise ValueError("Data not loaded. Call load_data() first.")

        os.makedirs(ML_CONFIG['output_dir'], exist_ok=True)
        plt.figure(figsize=VISUALIZATION_CONFIG["figure_size"], dpi=VISUALIZATION_CONFIG["dpi"])
        sns.set_style(VISUALIZATION_CONFIG["style"])

        fig, axes = plt.subplots(2, 2, figsize=VISUALIZATION_CONFIG['figure_size'])
        
        # Target distribution
        self.data[self.target_column].value_counts().plot(kind='bar', ax=axes[0, 0], title='Target Distribution')
        
        # Correlation heatmap
        numeric_cols = self.data.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 1:
            corr_matrix = self.data[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', ax=axes[0, 1])
            axes[0, 1].set_title('Feature Correlation')
        else:
            axes[0, 1].set_title('Not enough numeric features for correlation')
            axes[0, 1].axis('off')

        # Feature distributions (first 5 numeric features)
        if len(numeric_cols) > 0:
            self.data[numeric_cols[:5]].hist(ax=axes[1, 0], bins=20, alpha=0.7)
            axes[1, 0].set_title('Feature Distributions (First 5 Numeric)')
        else:
            axes[1, 0].set_title('No numeric features for distribution')
            axes[1, 0].axis('off')

        # Box plot for outlier detection (first 5 numeric features)
        if len(numeric_cols) > 0:
            self.data[numeric_cols[:5]].boxplot(ax=axes[1, 1])
            axes[1, 1].set_title('Outlier Detection (First 5 Numeric Features)')
        else:
            axes[1, 1].set_title('No numeric features for outlier detection')
            axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(ML_CONFIG['output_dir'], 'eda_analysis.png'), dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
        print("EDA visualizations saved as 'outputs/eda_analysis.png'")

    def create_polynomial_features(self, X):
        """
        Cria features polinomiais a partir dos dados de entrada.
        """
        poly = PolynomialFeatures(degree=2, include_bias=False)
        return poly.fit_transform(X)

    def preprocess_data(self):
        """
        Executa separação, normalização e seleção de features nos dados.
        """
        print("=== Data Preprocessing ===")
        if self.data is None: raise ValueError("Data not loaded. Call load_data() first.")

        X = self.data.drop(self.target_column, axis=1)
        y = self.data[self.target_column]
        
        # Handle categorical features if any (simple one-hot encoding for now)
        X = pd.get_dummies(X, drop_first=True)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=ML_CONFIG['test_size'],
            random_state=ML_CONFIG['random_state'],
            stratify=y
        )
        
        # Scaling
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        # Feature selection
        # Initialize SelectKBest here, after potential one-hot encoding
        self.feature_selector = SelectKBest(f_classif, k=min(10, self.X_train_scaled.shape[1]))
        self.X_train_selected = self.feature_selector.fit_transform(self.X_train_scaled, self.y_train)
        self.X_test_selected = self.feature_selector.transform(self.X_test_scaled)
        
        selected_feature_indices = self.feature_selector.get_support(indices=True)
        self.feature_names_selected = X.columns[selected_feature_indices]

        print(f"Selected {len(self.feature_names_selected)} best features: {list(self.feature_names_selected)}")
        print(f"Training set shape: {self.X_train_selected.shape}")
        print(f"Test set shape: {self.X_test_selected.shape}")

    def train_models(self):
        """
        Treina múltiplos modelos, avalia performance e computa scores.
        """
        print("=== Model Training and Comparison ===")
        if self.X_train_selected is None: raise ValueError("Data not preprocessed. Call preprocess_data() first.")

        self.results = {}
        for name, model in self.models.items():
            print(f"Training {name}...")
            cv_scores = cross_val_score(model, self.X_train_selected, self.y_train, cv=ML_CONFIG['cv_folds'], scoring='accuracy')
            model.fit(self.X_train_selected, self.y_train)
            train_pred = model.predict(self.X_train_selected)
            test_pred = model.predict(self.X_test_selected)
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
            print(f"   CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"   Test Accuracy: {test_accuracy:.4f}")
            if test_accuracy > self.best_score:
                self.best_score = test_accuracy
                self.best_model = model
                self.best_model_name = name

    def hyperparameter_tuning(self):
        """
        Executa tuning automatizado de hiperparâmetros nos modelos suportados.
        """
        print(f"=== Hyperparameter Tuning for {self.best_model_name} ===")
        if self.best_model is None: raise ValueError("No best model found. Call train_models() first.")

        if self.best_model_name in HYPERPARAMETER_GRIDS:
            param_grid = HYPERPARAMETER_GRIDS[self.best_model_name]
            grid_search = GridSearchCV(
                self.best_model, param_grid,
                cv=3, scoring='accuracy',
                n_jobs=ML_CONFIG['n_jobs'])
            grid_search.fit(self.X_train_selected, self.y_train)
            self.tuned_model = grid_search.best_estimator_
            tuned_pred = self.tuned_model.predict(self.X_test_selected)
            tuned_accuracy = accuracy_score(self.y_test, tuned_pred)
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Tuned model accuracy: {tuned_accuracy:.4f}")
            print(f"Improvement: {tuned_accuracy - self.best_score:.4f}")
            # Update best model if tuning improved it
            if tuned_accuracy > self.best_score:
                self.best_score = tuned_accuracy
                self.best_model = self.tuned_model
        else:
            print(f"Hyperparameter tuning not implemented for {self.best_model_name}.")

    def generate_report(self):
        """
        Gera gráficos de comparação de performance e matriz de confusão, salva em outputs/.
        """
        print("=== Model Evaluation Report ===")
        if self.best_model is None: raise ValueError("No best model found. Call train_models() first.")

        os.makedirs(ML_CONFIG['output_dir'], exist_ok=True)
        plt.figure(figsize=VISUALIZATION_CONFIG['figure_size'], dpi=VISUALIZATION_CONFIG['dpi'])
        sns.set_style(VISUALIZATION_CONFIG["style"])
        
        model_names = list(self.results.keys())
        test_scores = [self.results[name]['test_accuracy'] for name in model_names]
        cv_scores = [self.results[name]['cv_mean'] for name in model_names]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        x_pos = np.arange(len(model_names))
        ax1.bar(x_pos - 0.2, test_scores, width=0.4, alpha=0.7, label='Test Accuracy')
        ax1.bar(x_pos + 0.2, cv_scores, width=0.4, alpha=0.7, label='CV Mean')
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Performance Comparison')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        best_pred = self.results[self.best_model_name]['test_predictions']
        cm = confusion_matrix(self.y_test, best_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
        ax2.set_title(f'Confusion Matrix - {self.best_model_name}')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(os.path.join(ML_CONFIG['output_dir'], 'model_evaluation.png'), dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
        
        print(f"Best Model: {self.best_model_name}")
        print(f"Best Test Accuracy: {self.best_score:.4f}")
        print("\nDetailed Classification Report:")
        print(classification_report(self.y_test, best_pred))
        print("Model evaluation plots saved as 'outputs/model_evaluation.png'")

    def get_feature_importance(self):
        """
        Retorna a importância das features do melhor modelo, se disponível.
        """
        if self.best_model and hasattr(self.best_model, 'feature_importances_'):
            return pd.Series(self.best_model.feature_importances_, index=self.feature_names_selected)
        elif self.best_model and hasattr(self.best_model, 'coef_'):
            # For linear models, use absolute coefficients as importance
            return pd.Series(np.abs(self.best_model.coef_[0]), index=self.feature_names_selected)
        return None

    def plot_feature_importance(self, feature_importance):
        """
        Plota a importância das features e salva em outputs/.
        """
        if feature_importance is None: 
            print("Feature importance not available for the best model.")
            return

        print("=== Feature Importance Analysis ===")
        os.makedirs(ML_CONFIG['output_dir'], exist_ok=True)
        plt.figure(figsize=VISUALIZATION_CONFIG['figure_size'], dpi=VISUALIZATION_CONFIG['dpi'])
        sns.set_style(VISUALIZATION_CONFIG["style"])

        feature_importance.sort_values(ascending=False).plot(kind='bar')
        plt.title('Feature Importance')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(ML_CONFIG['output_dir'], 'feature_importance.png'), dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
        print("Feature importance plot saved as 'outputs/feature_importance.png'")

    def save_model(self, filename='best_model.pkl'):
        """
        Salva o melhor modelo, scaler e seletor de features para recuperação posterior.
        """
        if self.best_model is None: raise ValueError("No best model to save.")

        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'model_name': self.best_model_name,
            'accuracy': self.best_score,
            'target_column': self.target_column,
            'feature_names_selected': self.feature_names_selected.tolist() if self.feature_names_selected is not None else []
        }
        path = os.path.join(ML_CONFIG['output_dir'], filename)
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Best model saved as '{path}'")

    def load_model(self, filename='best_model.pkl'):
        """
        Carrega um modelo salvo, scaler e seletor de features.
        """
        path = os.path.join(ML_CONFIG['output_dir'], filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        print(f"Model '{model_data['model_name']}' loaded from '{path}'")
        return model_data

    def run_pipeline(self, data_frame=None, file_path=None, target_column=None):
        """
        Executa o pipeline completo de Machine Learning.
        """
        print("\nStarting Advanced Machine Learning Pipeline execution...")
        self.load_data(data_frame=data_frame, file_path=file_path, target_column=target_column, sample_data=(data_frame is None and file_path is None))
        self.exploratory_analysis()
        self.preprocess_data()
        self.train_models()
        self.hyperparameter_tuning()
        self.generate_report()
        
        feature_importance = self.get_feature_importance()
        if feature_importance is not None:
            self.plot_feature_importance(feature_importance)

        self.save_model()
        print("\nPipeline execution completed successfully!")
        return {
            "best_model_name": self.best_model_name,
            "best_accuracy": self.best_score,
            "results": self.results
        }

def main():
    """
    Executa o pipeline completo de Machine Learning com dataset de demonstração.
    Todos os outputs e modelos são salvos em 'outputs/'.
    """
    print("Advanced Machine Learning Pipeline")
    print("=" * 50)
    os.makedirs(ML_CONFIG['output_dir'], exist_ok=True)
    pipeline = MLPipeline()
    
    # Example with sample data
    pipeline.run_pipeline(sample_data=True)

    print("\nPipeline execution completed successfully!")


if __name__ == "__main__":
    main()

