#include "io/dataIO.hpp"
#include "mlp/mlp.hpp"
#include <iostream>
#include <sys/types.h>
#include <chrono>

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

    // Divisão hold-out em três subconjuntos: treino / validação / teste
    // 80% treino, 10% validação, 10% teste
    const float train_ratio = 0.80f;
    const float val_ratio   = 0.10f;
    
    
    std::vector<TrainingData> train_data, val_data, test_data;

    // splitTrainValTest(full_dataset, train_ratio, val_ratio, train_data, val_data, test_data);

    splitTrainValTestStratified(full_dataset, train_ratio, val_ratio, train_data, val_data, test_data);

    // Hiperparâmetros
    std::vector<int> hidden_sizes = {60};  // neurônios na camada escondida
    int epocas = 200;               // número máximo de épocas
    float threshold = 0.0005f;       // limiar de convergência do erro de treino
    float learning_rate = 0.001;    // taxa de aprendizado
    int patience = 15;             // épocas sem melhoria no val_loss antes do early stopping
    
    std::cout << "\n--- Configuracao ---\n";
    std::cout << "learning_rate : " << learning_rate << "\n";
    std::cout << "hidden_sizes  : [";
    for (size_t i = 0; i < hidden_sizes.size(); ++i) {
        std::cout << hidden_sizes[i];
        if (i < hidden_sizes.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
    std::cout << "epocas        : " << epocas    << "\n";
    std::cout << "patience      : " << patience  << "\n";
    std::cout << "threshold     : " << threshold << "\n\n";

    // Construção da rede MLP e exportação dos pesos iniciais
    auto mlp = MLPNetwork(input_size, output_size, hidden_sizes,
                          ActivationFunctionType::SigmoidStandard);

    // Exporta os valores iniciais
    const std::string output_dir = "results";
    ensureDir(output_dir);
    exportWeights(output_dir + "/pesos_iniciais.csv", mlp.getInitialWeights(),
                  "Pesos Iniciais");

    // Fase de Treinamento
    std::cout << "\nFazendo o treinamento..." << std::endl;

    auto t_inicio = std::chrono::high_resolution_clock::now();
    
    // Variações autorais (descomente para testar):
    // addNoise(train_data, 0.1f, 123);
    // addNoise(train_data, 0.2f, 123);
    // mergeClasses(full_dataset, 'O', 'D'); // trata O como D
    // mergeClasses(full_dataset, 'I', 'J'); // trata I como J

    TrainResult result = mlp.train(train_data, val_data,
                                   epocas, threshold, learning_rate, patience);

    auto t_fim = std::chrono::high_resolution_clock::now();
    auto tempo_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_fim - t_inicio).count();
    std::cout << "\nTempo de treinamento: " << tempo_ms / 1000.0 << "s\n";

    // Avaliação no conjunto de teste

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

    float erro_final_treino = result.train_losses.empty() ? 0.0f : result.train_losses.back();
    // Exporta os resultados
    exportHyperparameters(
        output_dir + "/hiperparametros.txt",
        input_size, output_size, hidden_sizes,
        epocas, threshold, learning_rate,
        val_ratio, patience,
        (int)result.train_losses.size(), erro_final_treino);

    exportWeights(output_dir + "/pesos_finais.csv", mlp.getFinalWeights(), "Pesos Finais");
    exportEpochErrors(output_dir + "/erros_por_epoca.csv", result.train_losses, result.val_losses);
    exportTestPredictions(output_dir + "/predicoes_teste.csv", test_data, predictions);
    std::cout << "\nDados exportados com sucesso!" << std::endl;

    return 0;
}
