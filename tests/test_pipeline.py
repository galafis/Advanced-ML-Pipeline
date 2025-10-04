"""
Testes unitários para o Advanced ML Pipeline.
Este arquivo garante que a execução do pipeline gera todos os outputs esperados e atinge acurácia válida.
Requer pytest instalado.
"""

import os
import pytest
from ml_pipeline import MLPipeline
from config import ML_CONFIG

def test_pipeline_execution_and_outputs():
    """
    Testa a execução completa do pipeline, geração de todos os arquivos de saída e limites aceitáveis de acurácia.
    """
    # Executa pipeline em modo sintético
    pipeline = MLPipeline()
    results = pipeline.run_pipeline() # Removido sample_data=True, pois run_pipeline já lida com isso

    # Verifica a criação dos artefatos em outputs/
    output_dir = ML_CONFIG["output_dir"]
    artifacts = [
        "eda_analysis.png",
        "model_evaluation.png",
        "feature_importance.png", # Adicionado para verificar a nova imagem
        "best_model.pkl"
    ]
    for artifact in artifacts:
        path = os.path.join(output_dir, artifact)
        assert os.path.isfile(path), f"Arquivo esperado não encontrado: {path}"

    # Checa se a acurácia é válida
    assert 0.0 <= results["best_accuracy"] <= 1.0, "Acurácia fora do intervalo esperado"

    # Verifica se o melhor modelo foi identificado
    assert results["best_model_name"] is not None, "Melhor modelo não identificado"

    # Verifica se os resultados do pipeline foram retornados
    assert "results" in results, "Resultados do pipeline não retornados"

