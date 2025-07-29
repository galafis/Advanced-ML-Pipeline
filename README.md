# Advanced Machine Learning Pipeline

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat&logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

Pipeline avançado de Machine Learning que automatiza todo o fluxo de trabalho ML, desde o pré-processamento de dados até a avaliação de modelos. Este projeto demonstra técnicas avançadas de ciência de dados incluindo análise exploratória automatizada, engenharia de features, comparação de modelos e otimização de hiperparâmetros.

## 🎯 Visão Geral

Sistema completo de Machine Learning que implementa as melhores práticas da indústria para desenvolvimento de modelos preditivos, oferecendo automação end-to-end com análise exploratória, feature engineering, treinamento de múltiplos algoritmos e avaliação robusta de performance.

### ✨ Características Principais

- **🔍 EDA Automatizada**: Análise exploratória abrangente com visualizações profissionais
- **⚙️ Feature Engineering**: Seleção e transformação automática de features
- **🤖 Comparação de Modelos**: Múltiplos algoritmos (Random Forest, Gradient Boosting, Logistic Regression, SVM)
- **📊 Validação Cruzada**: Avaliação robusta com k-fold cross-validation
- **🎛️ Otimização de Hiperparâmetros**: Busca automática com GridSearchCV
- **📈 Visualizações Profissionais**: Gráficos de performance e insights dos dados
- **💾 Persistência de Modelos**: Salvamento e carregamento de modelos treinados

## 🛠️ Stack Tecnológico

### Core Libraries
- **Python 3.11**: Linguagem principal
- **Scikit-learn**: Framework de Machine Learning
- **Pandas**: Manipulação e análise de dados
- **NumPy**: Computação numérica

### Visualization & Analysis
- **Matplotlib**: Visualizações estáticas
- **Seaborn**: Visualizações estatísticas
- **Plotly**: Gráficos interativos (opcional)

### Model Development
- **Random Forest**: Ensemble de árvores de decisão
- **Gradient Boosting**: Boosting sequencial
- **Logistic Regression**: Regressão logística
- **Support Vector Machine**: Máquinas de vetores de suporte

## 📁 Estrutura do Projeto

```
Advanced-ML-Pipeline/
├── ml_pipeline.py              # Pipeline principal
├── requirements.txt            # Dependências do projeto
├── README.md                   # Documentação
├── .gitignore                  # Arquivos ignorados pelo Git
├── data/                       # Dados de entrada (opcional)
├── outputs/                    # Resultados gerados
│   ├── eda_analysis.png        # Visualizações EDA
│   ├── model_evaluation.png    # Comparação de modelos
│   ├── feature_importance.png  # Importância das features
│   └── best_model.pkl          # Melhor modelo salvo
├── notebooks/                  # Jupyter notebooks (opcional)
└── tests/                      # Testes unitários
```

## 🚀 Quick Start

### Pré-requisitos

- Python 3.11 ou superior
- pip (gerenciador de pacotes Python)

### Instalação

1. **Clone o repositório:**
```bash
git clone https://github.com/galafis/Advanced-ML-Pipeline.git
cd Advanced-ML-Pipeline
```

2. **Instale as dependências:**
```bash
pip install -r requirements.txt
```

3. **Execute o pipeline:**
```bash
python ml_pipeline.py
```

### Uso Básico

```python
from ml_pipeline import MLPipeline
import pandas as pd

# Carregue seus dados
data = pd.read_csv('your_dataset.csv')

# Inicialize o pipeline
pipeline = MLPipeline()

# Execute o pipeline completo
results = pipeline.run_pipeline(data, target_column='target')

# Visualize os resultados
pipeline.plot_results()
```

## 🔍 Funcionalidades Detalhadas

### 📊 Análise Exploratória Automatizada

```python
def exploratory_data_analysis(self, data):
    """
    Realiza análise exploratória abrangente dos dados
    """
    # Estatísticas descritivas
    summary_stats = data.describe()
    
    # Análise de valores ausentes
    missing_analysis = data.isnull().sum()
    
    # Distribuições das variáveis
    self.plot_distributions(data)
    
    # Matriz de correlação
    self.plot_correlation_matrix(data)
    
    # Análise de outliers
    outliers = self.detect_outliers(data)
    
    return {
        'summary_stats': summary_stats,
        'missing_values': missing_analysis,
        'outliers': outliers
    }
```

### ⚙️ Feature Engineering Avançado

```python
def feature_engineering(self, X, y):
    """
    Engenharia de features automatizada
    """
    # Seleção de features baseada em estatísticas
    selector = SelectKBest(score_func=f_classif, k='all')
    X_selected = selector.fit_transform(X, y)
    
    # Normalização/Padronização
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    # Criação de features polinomiais (se aplicável)
    if X.shape[1] <= 10:  # Evitar explosão dimensional
        poly_features = self.create_polynomial_features(X_scaled)
        X_scaled = np.hstack([X_scaled, poly_features])
    
    return X_scaled, selector, scaler
```

### 🤖 Comparação de Modelos

```python
def compare_models(self, X, y):
    """
    Compara múltiplos algoritmos de ML
    """
    results = {}
    
    for name, model in self.models.items():
        # Validação cruzada
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        
        # Treinamento e avaliação
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        results[name] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    return results
```

### 🎛️ Otimização de Hiperparâmetros

```python
def hyperparameter_tuning(self, model, param_grid, X, y):
    """
    Otimização automática de hiperparâmetros
    """
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'best_estimator': grid_search.best_estimator_
    }
```

## 📊 Exemplos de Uso

### 1. Pipeline Completo com Dataset Iris

```python
from sklearn.datasets import load_iris
from ml_pipeline import MLPipeline

# Carregue o dataset
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target

# Execute o pipeline
pipeline = MLPipeline()
results = pipeline.run_complete_pipeline(data, 'target')

# Resultados
print(f"Melhor modelo: {results['best_model_name']}")
print(f"Acurácia: {results['best_accuracy']:.4f}")
```

### 2. Análise de Feature Importance

```python
# Obter importância das features
feature_importance = pipeline.get_feature_importance()

# Plotar importância
pipeline.plot_feature_importance(feature_importance)
```

### 3. Predições em Novos Dados

```python
# Carregar modelo salvo
best_model = pipeline.load_model('outputs/best_model.pkl')

# Fazer predições
new_data = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], 
                       columns=['sepal_length', 'sepal_width', 
                               'petal_length', 'petal_width'])
prediction = best_model.predict(new_data)
probability = best_model.predict_proba(new_data)

print(f"Predição: {prediction[0]}")
print(f"Probabilidades: {probability[0]}")
```

## 📈 Visualizações Geradas

### 1. Análise Exploratória
- Distribuições das variáveis
- Matriz de correlação
- Box plots para detecção de outliers
- Gráficos de dispersão para relações entre variáveis

### 2. Avaliação de Modelos
- Comparação de acurácias
- Curvas ROC (para classificação binária)
- Matrizes de confusão
- Gráficos de validação cruzada

### 3. Feature Analysis
- Importância das features
- Seleção de features
- Análise de correlação com target

## ⚡ Performance e Otimização

### Métricas de Performance

```python
def performance_metrics(self):
    """
    Calcula métricas abrangentes de performance
    """
    return {
        'accuracy': self.accuracy_score,
        'precision': self.precision_score,
        'recall': self.recall_score,
        'f1_score': self.f1_score,
        'roc_auc': self.roc_auc_score,
        'training_time': self.training_time,
        'prediction_time': self.prediction_time
    }
```

### Otimizações Implementadas

- **Paralelização**: Uso de `n_jobs=-1` em operações que suportam
- **Validação Eficiente**: Cross-validation otimizada
- **Memory Management**: Limpeza automática de variáveis temporárias
- **Caching**: Cache de resultados intermediários

## 🧪 Testes e Validação

### Executar Testes

```bash
# Testes unitários
python -m pytest tests/

# Teste de integração
python tests/test_integration.py

# Teste de performance
python tests/test_performance.py
```

### Validação de Dados

```python
def validate_data(self, data):
    """
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

## 📊 Casos de Uso

### 1. Classificação de Clientes
- Segmentação de clientes por comportamento
- Predição de churn
- Análise de lifetime value

### 2. Análise Médica
- Diagnóstico assistido por ML
- Análise de exames laboratoriais
- Predição de riscos

### 3. Análise Financeira
- Detecção de fraudes
- Análise de crédito
- Predição de mercado

## 🔧 Configuração Avançada

### Arquivo de Configuração

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

### Parâmetros de Modelos

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

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 👨‍💻 Autor

**Gabriel Demetrios Lafis**

- GitHub: [@galafis](https://github.com/galafis)
- Email: gabrieldemetrios@gmail.com

---

⭐ Se este projeto foi útil, considere deixar uma estrela!

