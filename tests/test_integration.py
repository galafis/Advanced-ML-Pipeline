"""
Teste de integração para o Advanced ML Pipeline.
Este arquivo testa a execução completa do pipeline de ponta a ponta com dados sintéticos,
garantindo que não há exceções e que o pipeline roda sem falhas.
Requer pytest instalado.
"""
import pytest
from ml_pipeline import MLPipeline


def test_end_to_end_pipeline_integration():
    """
    Teste de integração de ponta a ponta do MLPipeline.
    Executa todo o pipeline com sample_data e verifica que não há exceções.
    """
    try:
        # Inicializa o pipeline
        pipeline = MLPipeline()
        
        # Executa todas as etapas do pipeline com dados sintéticos
        data = pipeline.load_data(sample_data=True)
        pipeline.exploratory_analysis()
        pipeline.preprocess_data()
        pipeline.train_models()
        pipeline.hyperparameter_tuning()
        pipeline.generate_report()
        pipeline.save_model()
        
        # Se chegamos até aqui, o pipeline executou sem exceções
        assert True, "Pipeline executou com sucesso"
        
    except Exception as e:
        # Se qualquer exceção for lançada, o teste falha
        pytest.fail(f"Pipeline falhou com exceção: {str(e)}")


if __name__ == "__main__":
    # Permite executar o teste diretamente
    test_end_to_end_pipeline_integration()
    print("Teste de integração executado com sucesso!")
