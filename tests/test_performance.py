"""
Teste de performance para o Advanced ML Pipeline.
Este arquivo testa a performance de execução do pipeline de ponta a ponta,
garantindo que o tempo total de execução seja inferior a 30 segundos quando executado
com dados sintéticos (sample_data). Caso o tempo limite seja excedido, exibe uma
mensagem clara de falha com informações sobre o tempo de execução.
Requer pytest instalado.
"""
import time
import pytest
from ml_pipeline import MLPipeline


def test_pipeline_performance_under_30_seconds():
    """
    Teste de performance do pipeline de ML.
    
    Executa o pipeline completo de ponta a ponta com dados sintéticos e verifica
    se o tempo total de execução é inferior a 30 segundos. Este teste é importante
    para garantir que o pipeline mantenha uma performance adequada durante o
    desenvolvimento e não sofra regressões significativas de performance.
    
    Raises:
        AssertionError: Caso o pipeline exceda o tempo limite de 30 segundos.
        pytest.fail: Caso ocorra alguma exceção durante a execução do pipeline.
    
    Note:
        O tempo limite de 30 segundos foi definido considerando a execução
        com dados sintéticos pequenos, adequados para testes automatizados.
    """
    # Define o tempo limite em segundos
    TIME_LIMIT = 30.0
    
    try:
        # Marca o tempo de início
        start_time = time.time()
        
        # Inicializa e executa o pipeline completo
        pipeline = MLPipeline()
        
        # Executa todas as etapas do pipeline com dados sintéticos
        data = pipeline.load_data(sample_data=True)
        pipeline.exploratory_analysis()
        pipeline.preprocess_data()
        pipeline.train_models()
        pipeline.hyperparameter_tuning()
        pipeline.generate_report()
        pipeline.save_model()
        
        # Calcula o tempo total de execução
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Verifica se o tempo está dentro do limite
        if execution_time > TIME_LIMIT:
            pytest.fail(
                f"⚠️  PERFORMANCE TEST FAILED: Pipeline execution time exceeded limit!\n"
                f"   Expected: ≤ {TIME_LIMIT} seconds\n"
                f"   Actual: {execution_time:.2f} seconds\n"
                f"   Difference: +{execution_time - TIME_LIMIT:.2f} seconds over limit\n"
                f"   📈 Consider optimizing pipeline performance or adjusting test expectations."
            )
        
        # Se chegamos até aqui, o teste passou
        print(f"✅ Performance test passed! Pipeline executed in {execution_time:.2f} seconds")
        assert True, f"Pipeline executou em {execution_time:.2f}s (limite: {TIME_LIMIT}s)"
        
    except Exception as e:
        # Se qualquer exceção for lançada durante a execução, o teste falha
        pytest.fail(
            f"❌ Pipeline falhou com exceção durante teste de performance: {str(e)}"
        )


def test_pipeline_performance_measurement():
    """
    Teste auxiliar que apenas mede e reporta o tempo de execução do pipeline
    sem aplicar limites rígidos. Útil para monitoramento de performance.
    
    Este teste sempre passa, mas reporta o tempo de execução para ajudar
    no monitoramento de tendências de performance ao longo do tempo.
    """
    try:
        start_time = time.time()
        
        # Executa o pipeline
        pipeline = MLPipeline()
        data = pipeline.load_data(sample_data=True)
        pipeline.exploratory_analysis()
        pipeline.preprocess_data()
        pipeline.train_models()
        pipeline.hyperparameter_tuning()
        pipeline.generate_report()
        pipeline.save_model()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Sempre passa, mas reporta o tempo
        print(f"📊 Pipeline performance measurement: {execution_time:.2f} seconds")
        assert True, f"Tempo de execução medido: {execution_time:.2f}s"
        
    except Exception as e:
        pytest.fail(f"Erro durante medição de performance: {str(e)}")


if __name__ == "__main__":
    # Permite executar os testes diretamente
    print("Executando testes de performance do pipeline...")
    test_pipeline_performance_under_30_seconds()
    test_pipeline_performance_measurement()
    print("Testes de performance executados com sucesso!")
