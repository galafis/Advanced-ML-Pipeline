"""
Testes unitários para o Advanced ML Pipeline.
Este arquivo garante que a execução do pipeline gera todos os outputs esperados e atinge acurácia válida.
Requer pytest instalado.
"""

import os
import pytest
from ml_pipeline import MLPipeline, ML_CONFIG

def test_pipeline_execution_and_outputs():
    """
    Testa a execução completa do pipeline, geração de todos os arquivos de saída e limites aceitáveis de acurácia.
    """
    # Executa pipeline em modo sintético
    pipeline = MLPipeline()
    data = pipeline.load_data(sample_data=True)
    pipeline.exploratory_analysis()
    pipeline.preprocess_data()
    pipeline.train_models()
    pipeline.hyperparameter_tuning()
    pipeline.generate_report()
    pipeline.save_model()

    # Verifica a criação dos artefatos em outputs/
    output_dir = ML_CONFIG['output_dir']
    artifacts = [
        "eda_analysis.png",
        "model_evaluation.png",
        "best_model.pkl"
    ]
    for artifact in artifacts:
        path = os.path.join(output_dir, artifact)
        assert os.path.isfile(path), f"Arquivo esperado não encontrado: {path}"

    # Checa se a acurácia é válida
    assert 0.0 <= pipeline.best_score <= 1.0, "Acurácia fora do intervalo esperado"

    pytest.main([__file__, "-v"])
