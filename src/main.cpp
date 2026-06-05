#include "io/dataIO.hpp"
#include "mlp/mlp.hpp"
#include <iostream>
#include <sys/types.h>

// Função auxiliar para achar qual posição do vetor tem o maior valor
char getPredictedLetter(const std::vector<float> &output) {
    int max_index = 0;
    float max_val = output[0];
    for (size_t i = 1; i < output.size(); i++) {
        if (output[i] > max_val) {
            max_val = output[i];
            max_index = i;
        }
    }
    return 'A' + max_index;
}

int main() {
    int input_size = 120; // camada de entrada fotos (12x10)
    int output_size = 26; // camada de saída (26 letras do alfabeto)

    std::cout << "Carregando o dataset de caracteres..." << std::endl;
    auto full_dataset = loadCharacterData("datasets/caracteres/X.txt",
                                          "datasets/caracteres/Y.txt");

    if (full_dataset.empty()) {
        std::cerr << "Dataset vazio. Encerrando." << std::endl;
        return 1;
    }

    // Divide o dataset em 85% treino, 15% teste
    std::vector<TrainingData> train_data, test_data;
    splitTrainTest(full_dataset, 0.85f, train_data, test_data);

    // Inicia Hiperparâmetros
    std::vector<int> hidden_sizes = {60}; // 1 camada escondida de tamanho 60
    int epocas = 200;
    float threshold = 0.005f;
    float learning_rate = 0.001f;

    // cria rede neural
    auto mlp = MLPNetwork(input_size, output_size, hidden_sizes,
                          ActivationFunctionType::SigmoidStandard);

    // Exporta os valores iniciais
    const std::string output_dir = "results";
    ensureDir(output_dir);
    exportWeights(output_dir + "/pesos_iniciais.csv", mlp.getInitialWeights(),
                  "Pesos Iniciais");

    // Fase de Treinamento
    std::cout << "\nFazendo o treinamento..." << std::endl;
    auto epoch_losses = mlp.train(train_data, epocas, threshold, learning_rate);

    std::cout << "\nResultados das previsões (10 primeiras):" << std::endl;
    int acertos = 0;

    // Pega as predições do conjunto de teste
    std::vector<std::vector<float>> predictions;
    predictions.reserve(test_data.size());

    // Realiza testes de acerto do modelo
    for (size_t i = 0; i < test_data.size(); i++) {
        auto response = mlp.predict(test_data[i].input);
        predictions.push_back(response);

        char letra_esperada = getPredictedLetter(test_data[i].output);
        char letra_predita = getPredictedLetter(response);

        if (letra_esperada == letra_predita)
            acertos++;

        // Imprime apenas algumas para não poluir o terminal
        if (i < 10) {
            std::cout << "Esperado: " << letra_esperada
                      << " | Predicao: " << letra_predita << std::endl;
        }
    }
    // Medidas de acurácia e erro final
    float acuracia = (float)acertos / test_data.size() * 100.0f;
    std::cout << "\nRESUMO:" << std::endl;
    std::cout << "Acertos: " << acertos << " de " << test_data.size()
              << std::endl;
    std::cout << "Acuracia: " << acuracia << "%" << std::endl;

    float erro_final = epoch_losses.empty() ? 0.0f : epoch_losses.back();

    // Exporta os resultados
    exportHyperparameters(output_dir + "/hiperparametros.txt", input_size,
                          output_size, hidden_sizes, epocas, threshold,
                          learning_rate, (int)epoch_losses.size(), erro_final);

    exportWeights(output_dir + "/pesos_finais.csv", mlp.getFinalWeights(),
                  "Pesos Finais");
    exportEpochErrors(output_dir + "/erros_por_epoca.csv", epoch_losses);
    exportTestPredictions(output_dir + "/predicoes_teste.csv", test_data,
                          predictions);
    std::cout << "\nDados exportados com sucesso!" << std::endl;

    return 0;
}
