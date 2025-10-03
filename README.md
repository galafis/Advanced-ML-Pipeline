# Advanced Machine Learning Pipeline

## 🖼️ Hero Image

![Hero Image](outputs/hero_image.png)


![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat&logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

An advanced Machine Learning Pipeline that automates the entire ML workflow, from data preprocessing to model evaluation. This project demonstrates advanced data science techniques including automated exploratory analysis, feature engineering, model comparison, and hyperparameter optimization.

Pipeline avançado de Machine Learning que automatiza todo o fluxo de trabalho ML, desde o pré-processamento de dados até a avaliação de modelos. Este projeto demonstra técnicas avançadas de ciência de dados incluindo análise exploratória automatizada, engenharia de features, comparação de modelos e otimização de hiperparâmetros.

## 🎯 Overview / Visão Geral

A complete Machine Learning system that implements industry best practices for predictive model development, offering end-to-end automation with exploratory analysis, feature engineering, training of multiple algorithms, and robust performance evaluation.

Sistema completo de Machine Learning que implementa as melhores práticas da indústria para desenvolvimento de modelos preditivos, oferecendo automação end-to-end com análise exploratória, feature engineering, treinamento de múltiplos algoritmos e avaliação robusta de performance.

### ✨ Key Features / Características Principais

- **🔍 Automated EDA**: Comprehensive exploratory analysis with professional visualizations
- **⚙️ Feature Engineering**: Automatic feature selection and transformation
- **🤖 Model Comparison**: Multiple algorithms (Random Forest, Gradient Boosting, Logistic Regression, SVM)
- **📊 Cross-Validation**: Robust evaluation with k-fold cross-validation
- **🎛️ Hyperparameter Optimization**: Automatic search with GridSearchCV
- **📈 Professional Visualizations**: Performance graphs and data insights
- **💾 Model Persistence**: Saving and loading of trained models

- **🔍 EDA Automatizada**: Análise exploratória abrangente com visualizações profissionais
- **⚙️ Feature Engineering**: Seleção e transformação automática de features
- **🤖 Comparação de Modelos**: Múltiplos algoritmos (Random Forest, Gradient Boosting, Logistic Regression, SVM)
- **📊 Validação Cruzada**: Avaliação robusta com k-fold cross-validation
- **🎛️ Otimização de Hiperparâmetros**: Busca automática com GridSearchCV
- **📈 Visualizações Profissionais**: Gráficos de performance e insights dos dados
- **💾 Persistência de Modelos**: Salvamento e carregamento de modelos treinados

## 🛠️ Technology Stack / Stack Tecnológico

### Core Libraries / Bibliotecas Principais
- **Python 3.11**: Main language / Linguagem principal
- **Scikit-learn**: Machine Learning Framework / Framework de Machine Learning
- **Pandas**: Data manipulation and analysis / Manipulação e análise de dados
- **NumPy**: Numerical computation / Computação numérica

### Visualization & Analysis / Visualização e Análise
- **Matplotlib**: Static visualizations / Visualizações estáticas
- **Seaborn**: Statistical visualizations / Visualizações estatísticas
- **Plotly**: Interactive charts (optional) / Gráficos interativos (opcional)

### Model Development / Desenvolvimento de Modelos
- **Random Forest**: Ensemble of decision trees / Ensemble de árvores de decisão
- **Gradient Boosting**: Sequential boosting / Boosting sequencial
- **Logistic Regression**: Logistic regression / Regressão logística
- **Support Vector Machine**: Support vector machines / Máquinas de vetores de suporte

## 📁 Project Structure / Estrutura do Projeto

```
Advanced-ML-Pipeline/
├── ml_pipeline.py              # Main pipeline / Pipeline principal
├── requirements.txt            # Project dependencies / Dependências do projeto
├── README.md                   # Documentation / Documentação
├── .gitignore                  # Git ignored files / Arquivos ignorados pelo Git
├── data/                       # Input data (optional) / Dados de entrada (opcional)
├── outputs/                    # Generated results / Resultados gerados
│   ├── eda_analysis.png        # EDA visualizations / Visualizações EDA
│   ├── model_evaluation.png    # Model comparison / Comparação de modelos
│   ├── feature_importance.png  # Feature importance / Importância das features
│   └── best_model.pkl          # Best saved model / Melhor modelo salvo
├── notebooks/                  # Jupyter notebooks (optional) / Jupyter notebooks (opcional)
└── tests/                      # Unit tests / Testes unitários
```

## 🚀 Quick Start / Início Rápido

### Prerequisites / Pré-requisitos

- Python 3.11 or higher / Python 3.11 ou superior
- pip (Python package manager) / pip (gerenciador de pacotes Python)

### Installation / Instalação

1. **Clone the repository:** / **Clone o repositório:**
```bash
git clone https://github.com/galafis/Advanced-ML-Pipeline.git
cd Advanced-ML-Pipeline
```

2. **Install dependencies:** / **Instale as dependências:**
```bash
pip install -r requirements.txt
```

3. **Execute the pipeline:** / **Execute o pipeline:**
```bash
python ml_pipeline.py
```

### Basic Usage / Uso Básico

```python
from ml_pipeline import MLPipeline
import pandas as pd

# Load your data / Carregue seus dados
data = pd.read_csv(\'your_dataset.csv\')

# Initialize the pipeline / Inicialize o pipeline
pipeline = MLPipeline()

# Execute the complete pipeline / Execute o pipeline completo
results = pipeline.run_pipeline(data, target_column=\'target\')

# Visualize the results / Visualize os resultados
pipeline.plot_results()
```

## 🔍 Detailed Functionalities / Funcionalidades Detalhadas

### 📊 Automated Exploratory Data Analysis / Análise Exploratória Automatizada

```python
def exploratory_data_analysis(self, data):
    """
    Performs comprehensive exploratory data analysis
    Realiza análise exploratória abrangente dos dados
    """
    # Descriptive statistics / Estatísticas descritivas
    summary_stats = data.describe()
    
    # Missing values analysis / Análise de valores ausentes
    missing_analysis = data.isnull().sum()
    
    # Variable distributions / Distribuições das variáveis
    self.plot_distributions(data)
    
    # Correlation matrix / Matriz de correlação
    self.plot_correlation_matrix(data)
    
    # Outlier analysis / Análise de outliers
    outliers = self.detect_outliers(data)
    
    return {
        \'summary_stats\': summary_stats,
        \'missing_values\': missing_analysis,
        \'outliers\': outliers
    }
```

### ⚙️ Advanced Feature Engineering / Engenharia de Features Avançada

```python
def feature_engineering(self, X, y):
    """
    Automated feature engineering
    Engenharia de features automatizada
    """
    # Statistical-based feature selection / Seleção de features baseada em estatísticas
    selector = SelectKBest(score_func=f_classif, k=\'all\')
    X_selected = selector.fit_transform(X, y)
    
    # Normalization/Standardization / Normalização/Padronização
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    # Polynomial feature creation (if applicable) / Criação de features polinomiais (se aplicável)
    if X.shape[1] <= 10:  # Avoid dimensional explosion / Evitar explosão dimensional
        poly_features = self.create_polynomial_features(X_scaled)
        X_scaled = np.hstack([X_scaled, poly_features])
    
    return X_scaled, selector, scaler
```

### 🤖 Model Comparison / Comparação de Modelos

```python
def compare_models(self, X, y):
    """
    Compares multiple ML algorithms
    Compara múltiplos algoritmos de ML
    """
    results = {}
    
    for name, model in self.models.items():
        # Cross-validation / Validação cruzada
        cv_scores = cross_val_score(model, X, y, cv=5, scoring=\'accuracy\')
        
        # Training and evaluation / Treinamento e avaliação
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        results[name] = {
            \'cv_mean\': cv_scores.mean(),
            \'cv_std\': cv_scores.std(),
            \'test_accuracy\': accuracy_score(y_test, y_pred),
            \'classification_report\': classification_report(y_test, y_pred),
            \'confusion_matrix\': confusion_matrix(y_test, y_pred)
        }
    
    return results
```

### 🎛️ Hyperparameter Optimization / Otimização de Hiperparâmetros

```python
def hyperparameter_tuning(self, model, param_grid, X, y):
    """
    Automatic hyperparameter optimization
    Otimização automática de hiperparâmetros
    """
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring=\'accuracy\',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    return {
        \'best_params\': grid_search.best_params_,
        \'best_score\': grid_search.best_score_,
        \'best_estimator\': grid_search.best_estimator_
    }
```

## 📊 Usage Examples / Exemplos de Uso

### 1. Complete Pipeline with Iris Dataset / Pipeline Completo com Dataset Iris

```python
from sklearn.datasets import load_iris
from ml_pipeline import MLPipeline

# Load the dataset / Carregue o dataset
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data["target"] = iris.target

# Execute the pipeline / Execute o pipeline
pipeline = MLPipeline()
results = pipeline.run_complete_pipeline(data, "target")

# Results / Resultados
print(f"Best model: {results["best_model_name"]}")
print(f"Accuracy: {results["best_accuracy"]:.4f}")
```

### 2. Feature Importance Analysis / Análise de Importância de Features

```python
# Get feature importance / Obter importância das features
feature_importance = pipeline.get_feature_importance()

# Plot importance / Plotar importância
pipeline.plot_feature_importance(feature_importance)
```

### 3. Predictions on New Data / Predições em Novos Dados

```python
# Load saved model / Carregar modelo salvo
best_model = pipeline.load_model("outputs/best_model.pkl")

# Make predictions / Fazer predições
new_data = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], 
                       columns=["sepal_length", "sepal_width", 
                               "petal_length", "petal_width"])
prediction = best_model.predict(new_data)
probability = best_model.predict_proba(new_data)

print(f"Prediction: {prediction[0]}")
print(f"Probabilities: {probability[0]}")
```

## 📈 Generated Visualizations / Visualizações Geradas

### 1. Exploratory Analysis / Análise Exploratória
- Variable distributions / Distribuições das variáveis
- Correlation matrix / Matriz de correlação
- Box plots for outlier detection / Box plots para detecção de outliers
- Scatter plots for relationships between variables / Gráficos de dispersão para relações entre variáveis

### 2. Model Evaluation / Avaliação de Modelos
- Accuracy comparison / Comparação de acurácias
- ROC curves (for binary classification) / Curvas ROC (para classificação binária)
- Confusion matrices / Matrizes de confusão
- Cross-validation plots / Gráficos de validação cruzada

### 3. Feature Analysis / Análise de Features
- Feature importance / Importância das features
- Feature selection / Seleção de features
- Correlation analysis with target / Análise de correlação com target

## ⚡ Performance and Optimization / Performance e Otimização

### Performance Metrics / Métricas de Performance

```python
def performance_metrics(self):
    """
    Calculates comprehensive performance metrics
    Calcula métricas abrangentes de performance
    """
    return {
        \'accuracy\': self.accuracy_score,
        \'precision\': self.precision_score,
        \'recall\': self.recall_score,
        \'f1_score\': self.f1_score,
        \'roc_auc\': self.roc_auc_score,
        \'training_time\': self.training_time,
        \'prediction_time\': self.prediction_time
    }
```

### Implemented Optimizations / Otimizações Implementadas

- **Parallelization**: Use of `n_jobs=-1` in supported operations / Uso de `n_jobs=-1` em operações que suportam
- **Efficient Validation**: Optimized cross-validation / Cross-validation otimizada
- **Memory Management**: Automatic temporary variable cleanup / Limpeza automática de variáveis temporárias
- **Caching**: Caching of intermediate results / Cache de resultados intermediários

## 🧪 Tests and Validation / Testes e Validação

### Run Tests / Executar Testes

```bash
# Unit tests / Testes unitários
python -m pytest tests/

# Integration test / Teste de integração
python tests/test_integration.py

# Performance test / Teste de performance
python tests/test_performance.py
```

### Data Validation / Validação de Dados

```python
def validate_data(self, data):
    """
    Comprehensive validation of input data
    Validação abrangente dos dados de entrada
    """
    validations = {
        'shape_check': data.shape[0] > 0 and data.shape[1] > 0,
        'missing_values': data.isnull().sum().sum(),
        'data_types': data.dtypes.to_dict(),
        'duplicates': data.duplicated().sum(),
        'memory_usage': data.memory_usage(deep=True).sum()
    }
    
    return validations
```

## 📊 Use Cases / Casos de Uso

### 1. Customer Classification / Classificação de Clientes
- Customer segmentation by behavior / Segmentação de clientes por comportamento
- Churn prediction / Predição de churn
- Lifetime value analysis / Análise de lifetime value

### 2. Medical Analysis / Análise Médica
- ML-assisted diagnosis / Diagnóstico assistido por ML
- Laboratory exam analysis / Análise de exames laboratoriais
- Risk prediction / Predição de riscos

### 3. Financial Analysis / Análise Financeira
- Fraud detection / Detecção de fraudes
- Credit analysis / Análise de crédito
- Market prediction / Predição de mercado

## 🔧 Advanced Configuration / Configuração Avançada

### Configuration File / Arquivo de Configuração

```python
# config.py
ML_CONFIG = {
    'random_state': 42,
    'test_size': 0.2,
    'cv_folds': 5,
    'n_jobs': -1,
    'verbose': True,
    'save_models': True,
    'output_dir': 'outputs/'
}

VISUALIZATION_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'style': 'seaborn-v0_8',
    'color_palette': 'viridis'
}
```

### Model Parameters / Parâmetros de Modelos

```python
HYPERPARAMETER_GRIDS = {
    'Random Forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'Gradient Boosting': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
}
```

## 📄 License / Licença

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 👨‍💻 Autor

**Gabriel Demetrios Lafis**

- GitHub: [@galafis](https://github.com/galafis)
- Email: gabrieldemetrios@gmail.com

---

⭐ Se este projeto foi útil, considere deixar uma estrela!



## 🙏 Acknowledgments / Agradecimentos

Special thanks to all contributors and the open-source community for their invaluable support and resources.
Um agradecimento especial a todos os contribuidores e à comunidade open-source pelo seu inestimável apoio e recursos.

