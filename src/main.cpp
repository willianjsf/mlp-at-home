#include "mlp/mlp.hpp"

int main() {
    // TODO: carregar os dados e processá-los num formato aceitável (vetores de
    // inteiros)
    // TODO: separar em x (entrada) e y (resposta)
    // TODO: separar o x em dados de treino e teste

    // Cria MLP network
    auto mlp = MLPNetwork(120, 10, std::vector<int>(1, 100));

    // TODO: incluir o dados de treinamento
    // mlp.train();
    // TODO: mostrar resultados do treinamento

    // TODO: realizar o teste
    // auto response = mlp.predict();
    // TODO: mostrar as métricas de resultado

    // TODO: exportar dados do trabalho:
    // 1. hiperparametros finais e de inicialização
    // 2. pesos iniciais e finais
    // 3. erro cometido em cada interação
    // 4. saída produzida em cada dado de teste
    return 0;
}
