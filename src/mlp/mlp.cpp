
#include "./mlp.hpp"
#include <assert.h>
#include <math.h>

// multiplicaVetorMatrix recebe um vetor e uma matriz, multiplica ambos e
// retorna um vetor de tamanho igual ao número de linhas da matriz
static std::vector<float>
multiplicaVetorMatrix(std::vector<std::vector<float>> &matriz,
                      std::vector<float> &vetor) {
    auto resultado = std::vector<float>(matriz.size());

    for (size_t i = 0; i < matriz.size(); i++) {
        int sum = 0;
        for (size_t j = 0; j < matriz[0].size(); j++) {
            sum += matriz[i][j] * vetor[j];
        }
        resultado[i] = sum;
    }

    return resultado;
}

// somaVetores soma dois vetores de tamanho igual
static std::vector<float> somaVetores(std::vector<float> &a,
                                      std::vector<float> &b) {
    assert(a.size() == b.size());
    auto resultado = std::vector<float>(a.size());

    for (size_t i = 0; i < a.size(); i++) {
        resultado[i] = a[i] + b[i];
    }

    return resultado;
}

// predict() executa a rede neural, retornando o resultado no formato de
// vetor correspondente
std::vector<float>
MultiLayerPerceptronNetwork::predict(std::vector<float> &data) {
    assert(static_cast<int>(data.size()) == get_input_size());

    auto ativacao =
        std::vector<float>(data.begin(), data.end()); // copia o input

    // para cada camada
    for (int camada = 0; camada < get_hidden_layer_size(); camada++) {

        auto pesos = this->weights[camada];
        auto bias = this->biases[camada];

        // Calcular a combinação linear z = W * a + b
        auto z = multiplicaVetorMatrix(pesos, ativacao);
        z = somaVetores(z, bias);

        // aplicar a função de ativação
        // passar para a próxima camada
        ativacao = activationFunction(z);
    }

    return ativacao;
}

// activationFunction é uma função de ativação, que recebe um vetor e aplica a
// cada componente a função de ativação definida em activation_function, ex.:
// ReLU, Sigmoid, etc.
std::vector<float>
MultiLayerPerceptronNetwork::activationFunction(std::vector<float> &z) {
    auto result = std::vector<float>(z.size());
    for (size_t i = 0; i < z.size(); i++) {
        switch (this->activation_function_type) {
        case ActivationFunctionType::ReLu:
            result[i] = std::max(0.0f, z[i]);
            break;
        case ActivationFunctionType::Sigmoid:
            result[i] = 1.0f / (1.0f + std::exp(-z[i]));
            break;
        default:
            result[i] = z[i];
            break;
        }
    }
    return result;
}
