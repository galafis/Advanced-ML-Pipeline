# Advanced Machine Learning Pipeline

[English](#english) | [Português](#português)

## English

### Overview
A comprehensive machine learning pipeline that automates the entire ML workflow from data preprocessing to model evaluation. This project demonstrates advanced data science techniques including exploratory data analysis, feature engineering, model comparison, and hyperparameter tuning.

### Features
- **Automated EDA**: Comprehensive exploratory data analysis with visualizations
- **Feature Engineering**: Automated feature selection and scaling
- **Model Comparison**: Multiple algorithms comparison (Random Forest, Gradient Boosting, Logistic Regression, SVM)
- **Cross-Validation**: Robust model evaluation with k-fold cross-validation
- **Hyperparameter Tuning**: Automated parameter optimization using GridSearchCV
- **Visualization**: Professional plots for model performance and data insights
- **Model Persistence**: Save and load trained models

### Technologies Used
- **Python 3.8+**
- **Scikit-learn**: Machine learning algorithms and utilities
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib & Seaborn**: Data visualization
- **Jupyter**: Interactive development environment

### Installation

1. Clone the repository:
```bash
git clone https://github.com/galafis/Advanced-ML-Pipeline.git
cd Advanced-ML-Pipeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### Basic Usage
```python
from ml_pipeline import MLPipeline

# Initialize pipeline
pipeline = MLPipeline()

# Load data (uses sample data by default)
data = pipeline.load_data(sample_data=True)

# Run complete pipeline
pipeline.exploratory_analysis()
pipeline.preprocess_data()
pipeline.train_models()
pipeline.hyperparameter_tuning()
pipeline.generate_report()
pipeline.save_model()
```

#### Using Custom Dataset
```python
# Load your own dataset
pipeline.load_data(file_path='your_dataset.csv', sample_data=False)
```

#### Command Line Execution
```bash
python ml_pipeline.py
```

### Project Structure
```
Advanced-ML-Pipeline/
├── ml_pipeline.py          # Main pipeline implementation
├── requirements.txt        # Project dependencies
├── README.md              # Project documentation
├── .gitignore             # Git ignore file
├── eda_analysis.png       # EDA visualizations (generated)
├── model_evaluation.png   # Model comparison plots (generated)
└── best_model.pkl         # Saved best model (generated)
```

### Output Files
- `eda_analysis.png`: Exploratory data analysis visualizations
- `model_evaluation.png`: Model performance comparison charts
- `best_model.pkl`: Serialized best performing model

### Model Performance
The pipeline automatically compares multiple algorithms and selects the best performing model based on cross-validation scores. Typical results include:

- **Random Forest**: High accuracy with feature importance insights
- **Gradient Boosting**: Excellent performance on complex patterns
- **Logistic Regression**: Fast and interpretable baseline
- **SVM**: Effective for high-dimensional data

### Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Português

### Visão Geral
Um pipeline abrangente de machine learning que automatiza todo o fluxo de trabalho de ML, desde o pré-processamento de dados até a avaliação de modelos. Este projeto demonstra técnicas avançadas de ciência de dados, incluindo análise exploratória, engenharia de features, comparação de modelos e ajuste de hiperparâmetros.

### Funcionalidades
- **EDA Automatizada**: Análise exploratória abrangente com visualizações
- **Engenharia de Features**: Seleção e escalonamento automático de características
- **Comparação de Modelos**: Comparação de múltiplos algoritmos (Random Forest, Gradient Boosting, Regressão Logística, SVM)
- **Validação Cruzada**: Avaliação robusta com validação cruzada k-fold
- **Ajuste de Hiperparâmetros**: Otimização automática usando GridSearchCV
- **Visualização**: Gráficos profissionais para performance e insights dos dados
- **Persistência de Modelo**: Salvar e carregar modelos treinados

### Tecnologias Utilizadas
- **Python 3.8+**
- **Scikit-learn**: Algoritmos e utilitários de machine learning
- **Pandas**: Manipulação e análise de dados
- **NumPy**: Computação numérica
- **Matplotlib & Seaborn**: Visualização de dados
- **Jupyter**: Ambiente de desenvolvimento interativo

### Instalação

1. Clone o repositório:
```bash
git clone https://github.com/galafis/Advanced-ML-Pipeline.git
cd Advanced-ML-Pipeline
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

### Uso

#### Uso Básico
```python
from ml_pipeline import MLPipeline

# Inicializar pipeline
pipeline = MLPipeline()

# Carregar dados (usa dados de exemplo por padrão)
data = pipeline.load_data(sample_data=True)

# Executar pipeline completo
pipeline.exploratory_analysis()
pipeline.preprocess_data()
pipeline.train_models()
pipeline.hyperparameter_tuning()
pipeline.generate_report()
pipeline.save_model()
```

#### Usando Dataset Personalizado
```python
# Carregar seu próprio dataset
pipeline.load_data(file_path='seu_dataset.csv', sample_data=False)
```

#### Execução via Linha de Comando
```bash
python ml_pipeline.py
```

### Estrutura do Projeto
```
Advanced-ML-Pipeline/
├── ml_pipeline.py          # Implementação principal do pipeline
├── requirements.txt        # Dependências do projeto
├── README.md              # Documentação do projeto
├── .gitignore             # Arquivo git ignore
├── eda_analysis.png       # Visualizações EDA (gerado)
├── model_evaluation.png   # Gráficos de comparação (gerado)
└── best_model.pkl         # Melhor modelo salvo (gerado)
```

### Arquivos de Saída
- `eda_analysis.png`: Visualizações da análise exploratória
- `model_evaluation.png`: Gráficos de comparação de performance
- `best_model.pkl`: Melhor modelo serializado

### Performance dos Modelos
O pipeline compara automaticamente múltiplos algoritmos e seleciona o modelo com melhor performance baseado em scores de validação cruzada. Resultados típicos incluem:

- **Random Forest**: Alta precisão com insights de importância das features
- **Gradient Boosting**: Excelente performance em padrões complexos
- **Regressão Logística**: Baseline rápido e interpretável
- **SVM**: Efetivo para dados de alta dimensionalidade

### Contribuindo
1. Faça um fork do repositório
2. Crie uma branch de feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -am 'Adicionar nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Crie um Pull Request

### Licença
Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.



## 📋 Descrição

Descreva aqui o conteúdo desta seção.


## 📦 Instalação

Descreva aqui o conteúdo desta seção.


## 💻 Uso

Descreva aqui o conteúdo desta seção.


## 📄 Licença

Descreva aqui o conteúdo desta seção.
