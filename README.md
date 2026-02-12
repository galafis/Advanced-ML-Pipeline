# Advanced Machine Learning Pipeline

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat&logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

Um pipeline avançado de Machine Learning que automatiza todo o fluxo de trabalho de ML, desde o pré-processamento de dados até a avaliação de modelos. Este projeto demonstra técnicas avançadas de ciência de dados, incluindo análise exploratória automatizada, engenharia de features, comparação de modelos e otimização de hiperparâmetros.

An advanced Machine Learning Pipeline that automates the entire ML workflow, from data preprocessing to model evaluation. This project demonstrates advanced data science techniques including automated exploratory analysis, feature engineering, model comparison, and hyperparameter optimization.

## 🎯 Overview / Visão Geral

Um sistema completo de Machine Learning que implementa as melhores práticas da indústria para o desenvolvimento de modelos preditivos, oferecendo automação de ponta a ponta com análise exploratória, engenharia de features, treinamento de múltiplos algoritmos e avaliação robusta de performance.

A complete Machine Learning system that implements industry best practices for predictive model development, offering end-to-end automation with exploratory analysis, feature engineering, training of multiple algorithms, and robust performance evaluation.

### ✨ Key Features / Características Principais

- **🔍 Automated EDA / EDA Automatizada**: Análise exploratória abrangente com visualizações profissionais.
- **⚙️ Feature Engineering / Engenharia de Features**: Seleção e transformação automática de features.
- **🤖 Model Comparison / Comparação de Modelos**: Múltiplos algoritmos (Random Forest, Gradient Boosting, Logistic Regression, SVM).
- **📊 Cross-Validation / Validação Cruzada**: Avaliação robusta com k-fold cross-validation.
- **🎛️ Hyperparameter Optimization / Otimização de Hiperparâmetros**: Busca automática com GridSearchCV.
- **📈 Professional Visualizations / Visualizações Profissionais**: Gráficos de performance e insights dos dados.
- **💾 Model Persistence / Persistência de Modelos**: Salvamento e carregamento de modelos treinados.

## 🛠️ Technology Stack / Stack Tecnológico

### Core Libraries / Bibliotecas Principais
- **Python 3.11+**: Main language / Linguagem principal
- **Scikit-learn**: Machine Learning Framework / Framework de Machine Learning
- **Pandas**: Data manipulation and analysis / Manipulação e análise de dados
- **NumPy**: Numerical computation / Computação numérica

### Visualization & Analysis / Visualização e Análise
- **Matplotlib**: Static visualizations / Visualizações estáticas
- **Seaborn**: Statistical visualizations / Visualizações estatísticas

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
├── config.py                   # Configuration settings / Configurações do projeto
├── data/                       # Input data (optional) / Dados de entrada (opcional)
├── outputs/                    # Generated results / Resultados gerados
│   ├── eda_analysis.png        # EDA visualizations / Visualizações EDA
│   ├── model_evaluation.png    # Model comparison / Comparação de modelos
│   ├── feature_importance.png  # Feature importance / Importância das features
│   └── best_model.pkl          # Best saved model / Melhor modelo salvo
├── notebooks/                  # Jupyter notebooks (optional) / Jupyter notebooks (opcional)
└── tests/                      # Unit and integration tests / Testes unitários e de integração
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
# Exemplo com dados de demonstração (Iris dataset)
from sklearn.datasets import load_iris
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data["target"] = iris.target

# Initialize the pipeline / Inicialize o pipeline
pipeline = MLPipeline()

# Execute the complete pipeline / Execute o pipeline completo
# Certifique-se de que a coluna 'target' existe em seus dados
results = pipeline.run_pipeline(data_frame=data, target_column='target')
```

## 🔍 Detailed Functionalities / Funcionalidades Detalhadas

### 📊 Automated Exploratory Data Analysis / Análise Exploratória Automatizada

O pipeline inclui uma fase de Análise Exploratória de Dados (EDA) automatizada, que gera estatísticas descritivas, analisa valores ausentes, visualiza distribuições de variáveis, matrizes de correlação e detecta outliers. Isso fornece uma compreensão profunda dos dados antes do treinamento do modelo.

The pipeline includes an automated Exploratory Data Analysis (EDA) phase, which generates descriptive statistics, analyzes missing values, visualizes variable distributions, correlation matrices, and detects outliers. This provides a deep understanding of the data before model training.

### ⚙️ Advanced Feature Engineering / Engenharia de Features Avançada

Esta seção do pipeline lida com a seleção e transformação automática de features. Inclui seleção de features baseada em estatísticas (e.g., `SelectKBest`), normalização/padronização de dados (`StandardScaler`) e, se aplicável, criação de features polinomiais para capturar relações não lineares.

This section of the pipeline handles automatic feature selection and transformation. It includes statistical-based feature selection (e.g., `SelectKBest`), data normalization/standardization (`StandardScaler`), and, if applicable, polynomial feature creation to capture non-linear relationships.

### 🤖 Model Comparison / Comparação de Modelos

O pipeline compara múltiplos algoritmos de Machine Learning, como Random Forest, Gradient Boosting, Regressão Logística e SVM. Cada modelo é avaliado usando validação cruzada (k-fold) e métricas de desempenho robustas para identificar o melhor modelo para o conjunto de dados.

The pipeline compares multiple Machine Learning algorithms, such as Random Forest, Gradient Boosting, Logistic Regression, and SVM. Each model is evaluated using k-fold cross-validation and robust performance metrics to identify the best model for the dataset.

### 🎛️ Hyperparameter Optimization / Otimização de Hiperparâmetros

Para garantir o desempenho ideal do modelo, o pipeline incorpora otimização automática de hiperparâmetros usando `GridSearchCV`. Isso busca sistematicamente a melhor combinação de hiperparâmetros para o modelo selecionado, melhorando sua acurácia e generalização.

To ensure optimal model performance, the pipeline incorporates automatic hyperparameter optimization using `GridSearchCV`. This systematically searches for the best combination of hyperparameters for the selected model, improving its accuracy and generalization.

## 📊 Usage Examples / Exemplos de Uso

### 1. Complete Pipeline with Iris Dataset / Pipeline Completo com Dataset Iris

Este exemplo demonstra como executar o pipeline completo usando o popular dataset Iris. Ele carrega os dados, inicializa o pipeline, executa todas as etapas (EDA, pré-processamento, treinamento, otimização) e gera um relatório de avaliação.

This example demonstrates how to run the complete pipeline using the popular Iris dataset. It loads the data, initializes the pipeline, executes all steps (EDA, preprocessing, training, optimization), and generates an evaluation report.

```python
from sklearn.datasets import load_iris
from ml_pipeline import MLPipeline
import pandas as pd

# Load the dataset / Carregue o dataset
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data["target"] = iris.target

# Execute the pipeline / Execute o pipeline
pipeline = MLPipeline()
results = pipeline.run_pipeline(data_frame=data, target_column="target")

# Results / Resultados
print(f"Best model: {pipeline.best_model_name}")
print(f"Accuracy: {pipeline.best_score:.4f}")
```

### 2. Feature Importance Analysis / Análise de Importância de Features

Após o treinamento do modelo, é possível analisar a importância das features para entender quais variáveis contribuem mais para as previsões do modelo. O pipeline pode gerar visualizações para essa análise.

After model training, it's possible to analyze feature importance to understand which variables contribute most to the model's predictions. The pipeline can generate visualizations for this analysis.

```python
# Assuming pipeline.run_pipeline has been executed
# Supondo que pipeline.run_pipeline tenha sido executado

# Get feature importance / Obter importância das features
feature_importance = pipeline.get_feature_importance()

# Plot importance / Plotar importância
pipeline.plot_feature_importance(feature_importance)
```

### 3. Predictions on New Data / Predições em Novos Dados

O modelo treinado e otimizado pode ser salvo e posteriormente carregado para fazer previsões em novos dados. Isso demonstra a capacidade de implantação do pipeline.

The trained and optimized model can be saved and later loaded to make predictions on new data. This demonstrates the deployment capability of the pipeline.

```python
import pandas as pd
from ml_pipeline import MLPipeline

# Initialize pipeline to load model / Inicializar pipeline para carregar modelo
pipeline = MLPipeline()

# Load saved model / Carregar modelo salvo
best_model_data = pipeline.load_model("best_model.pkl")
best_model = best_model_data["model"]
scaler = best_model_data["scaler"]
feature_selector = best_model_data["feature_selector"]

# Make predictions / Fazer predições
new_data = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], 
                       columns=["sepal length (cm)", "sepal width (cm)", 
                               "petal length (cm)", "petal width (cm)"])

# Apply the same preprocessing transformations (scale, then select)
new_data_scaled = scaler.transform(new_data)
new_data_selected = feature_selector.transform(new_data_scaled)

prediction = best_model.predict(new_data_selected)
probability = best_model.predict_proba(new_data_selected)

print(f"Prediction: {prediction[0]}")
print(f"Probabilities: {probability[0]}")
```

## 📈 Generated Visualizations / Visualizações Geradas

O pipeline gera automaticamente várias visualizações para auxiliar na compreensão dos dados e na avaliação do modelo. Essas visualizações são salvas no diretório `outputs/`.

The pipeline automatically generates various visualizations to aid in data understanding and model evaluation. These visualizations are saved in the `outputs/` directory.

### 1. Exploratory Analysis / Análise Exploratória
- **Variable distributions / Distribuições das variáveis**: Histogramas e gráficos de densidade para entender a forma dos dados.
- **Correlation matrix / Matriz de correlação**: Mapa de calor mostrando a correlação entre as features.
- **Box plots for outlier detection / Box plots para detecção de outliers**: Identificação visual de valores atípicos.

### 2. Model Evaluation / Avaliação de Modelos
- **Accuracy comparison / Comparação de acurácias**: Gráficos de barras comparando o desempenho de diferentes modelos.
- **Confusion matrices / Matrizes de confusão**: Detalhamento dos verdadeiros positivos, verdadeiros negativos, falsos positivos e falsos negativos.

### 3. Feature Analysis / Análise de Features
- **Feature importance / Importância das features**: Gráficos mostrando a contribuição de cada feature para o modelo.

## ⚡ Performance and Optimization / Performance e Otimização

O pipeline é projetado com foco em performance e otimização, utilizando técnicas como paralelização e gerenciamento eficiente de memória.

The pipeline is designed with a focus on performance and optimization, utilizing techniques such as parallelization and efficient memory management.

### Performance Metrics / Métricas de Performance

O pipeline calcula e reporta métricas de desempenho para cada modelo, incluindo acurácia, precisão, recall e F1-score.

The pipeline calculates and reports performance metrics for each model, including accuracy, precision, recall, and F1-score.

### Implemented Optimizations / Otimizações Implementadas

- **Parallelization / Paralelização**: Uso de `n_jobs=-1` em operações suportadas para aproveitar múltiplos núcleos de CPU.
- **Efficient Validation / Validação Eficiente**: Validação cruzada otimizada para reduzir o tempo de computação.

## 🧪 Tests and Validation / Testes e Validação

O projeto inclui um conjunto de testes para garantir a funcionalidade e a robustez do pipeline.

The project includes a suite of tests to ensure the functionality and robustness of the pipeline.

### Run Tests / Executar Testes

Para executar os testes, navegue até o diretório raiz do projeto e use os seguintes comandos:

To run the tests, navigate to the project root directory and use the following commands:

```bash
# Unit tests / Testes unitários
python -m pytest tests/test_pipeline.py

# Integration test / Teste de integração
python -m pytest tests/test_integration.py

# Performance test / Teste de performance
python -m pytest tests/test_performance.py
```

### Data Validation / Validação de Dados

O pipeline valida os dados de entrada verificando se a coluna alvo existe e se o DataFrame contém dados válidos antes de iniciar o processamento.

The pipeline validates input data by checking that the target column exists and that the DataFrame contains valid data before starting processing.

##  Advanced Configuration / Configuração Avançada

### Configuration File / Arquivo de Configuração

O arquivo `config.py` centraliza todas as configurações do pipeline, permitindo fácil personalização de parâmetros como `random_state`, `test_size`, `cv_folds`, diretórios de saída e opções de verbosidade.

The `config.py` file centralizes all pipeline configurations, allowing easy customization of parameters such as `random_state`, `test_size`, `cv_folds`, output directories, and verbosity options.

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
    'style': 'whitegrid',
    'color_palette': 'viridis'
}
```

### Model Parameters / Parâmetros de Modelos

As grades de hiperparâmetros para otimização são definidas em `config.py`, permitindo que os usuários especifiquem os intervalos de busca para diferentes modelos de ML, como Random Forest e Gradient Boosting.

Hyperparameter grids for optimization are defined in `config.py`, allowing users to specify search ranges for different ML models, such as Random Forest and Gradient Boosting.

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
    },
    'Logistic Regression': {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
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

