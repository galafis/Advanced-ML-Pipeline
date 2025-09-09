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
from sklearn.preprocessing import StandardScaler
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
            'SVM': SVC(random_state=ML_CONFIG['random_state'])
        }
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(f_classif, k=10)
        self.best_model = None
        self.best_score = 0
        self.best_model_name = None
        self.results = {}

    def load_data(self, file_path=None, sample_data=True):
        """
        Carrega dataset para treino. Se sample_data=True, gera um conjunto sintético.
        """
        if sample_data:
            np.random.seed(ML_CONFIG['random_state'])
            n_samples = 1000
            n_features = 15
            X = np.random.randn(n_samples, n_features)
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
        """
        Executa análise exploratória nos dados, gera gráficos EDA e salva em outputs/.
        """
        print("=== Exploratory Data Analysis ===")
        print(f"Dataset shape: {self.data.shape}")
        print(f"Missing values: {self.data.isnull().sum().sum()}")
        print(f"Target distribution:\n{self.data['target'].value_counts()}")
        os.makedirs(ML_CONFIG['output_dir'], exist_ok=True)
        plt.figure(figsize=VISUALIZATION_CONFIG["figure_size"], dpi=VISUALIZATION_CONFIG["dpi"])
        sns.set_style(VISUALIZATION_CONFIG["style"])

        fig, axes = plt.subplots(2, 2, figsize=VISUALIZATION_CONFIG['figure_size'])
        # Target distribution
        self.data['target'].value_counts().plot(kind='bar', ax=axes[0, 0], title='Target Distribution')
        # Correlation heatmap
        corr_matrix = self.data.corr()
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', ax=axes[0, 1])
        axes[0, 1].set_title('Feature Correlation')
        # Feature distributions
        self.data.iloc[:, :5].hist(ax=axes[1, 0], bins=20, alpha=0.7)
        axes[1, 0].set_title('Feature Distributions (First 5)')
        # Box plot for outlier detection
        self.data.iloc[:, :5].boxplot(ax=axes[1, 1])
        axes[1, 1].set_title('Outlier Detection (First 5 Features)')
        plt.tight_layout()
        plt.savefig(os.path.join(ML_CONFIG['output_dir'], 'eda_analysis.png'), dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
        print("EDA visualizations saved as 'outputs/eda_analysis.png'")

    def preprocess_data(self):
        """
        Executa separação, normalização e seleção de features nos dados.
        """
        print("=== Data Preprocessing ===")
        X = self.data.drop('target', axis=1)
        y = self.data['target']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=ML_CONFIG['test_size'],
            random_state=ML_CONFIG['random_state'],
            stratify=y
        )
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        self.X_train_selected = self.feature_selector.fit_transform(self.X_train_scaled, self.y_train)
        self.X_test_selected = self.feature_selector.transform(self.X_test_scaled)
        selected_features = self.feature_selector.get_support()
        feature_names = X.columns[selected_features]
        print(f"Selected {len(feature_names)} best features: {list(feature_names)}")
        print(f"Training set shape: {self.X_train_selected.shape}")
        print(f"Test set shape: {self.X_test_selected.shape}")

    def train_models(self):
        """
        Treina múltiplos modelos, avalia performance e computa scores.
        """
        print("=== Model Training and Comparison ===")
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
        else:
            print("Hyperparameter tuning not implemented for this model.")

    def generate_report(self):
        """
        Gera gráficos de comparação de performance e matriz de confusão, salva em outputs/.
        """
        print("=== Model Evaluation Report ===")
        os.makedirs(ML_CONFIG['output_dir'], exist_ok=True)
        plt.figure(figsize=VISUALIZATION_CONFIG['figure_size'], dpi=VISUALIZATION_CONFIG['dpi'])
        sns.set_style(VISUALIZATION_CONFIG["style"])
        model_names = list(self.results.keys())
        test_scores = [self.results[name]['test_accuracy'] for name in model_names]
        cv_scores = [self.results[name]['cv_mean'] for name in model_names]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
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

    def save_model(self, filename='best_model.pkl'):
        """
        Salva o melhor modelo, scaler e seletor de features para recuperação posterior.
        """
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'model_name': self.best_model_name,
            'accuracy': self.best_score
        }
        path = os.path.join(ML_CONFIG['output_dir'], filename)
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Best model saved as '{path}'")


def main():
    """
    Executa o pipeline completo de Machine Learning com dataset de demonstração.
    Todos os outputs e modelos são salvos em 'outputs/'.
    """
    print("Advanced Machine Learning Pipeline")
    print("=" * 50)
    os.makedirs(ML_CONFIG['output_dir'], exist_ok=True)
    pipeline = MLPipeline()
    data = pipeline.load_data(sample_data=True)
    pipeline.exploratory_analysis()
    pipeline.preprocess_data()
    pipeline.train_models()
    pipeline.hyperparameter_tuning()
    pipeline.generate_report()
    pipeline.save_model()
    print("\nPipeline execution completed successfully!")


if __name__ == "__main__":
    main()

