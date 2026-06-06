#include "dataIO.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <map>

void ensureDir(const std::string &path) {
    // Se o diretório não existe cria-o, senão faz nada
    std::filesystem::create_directories(path);
}

std::vector<TrainingData> loadData(const std::string &filename, int input_size,
                                   int output_size) {
    // Carrega o dataset
    std::vector<TrainingData> dataset;
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Erro ao abrir o arquivo: " << filename << std::endl;
        return dataset;
    }

    bool first_line = true;
    while (std::getline(file, line)) {
        // Tratando um bugzinho com o BOM (Byte Order Mark)
        if (first_line) {
            if (line.size() >= 3 && line.substr(0, 3) == "\xEF\xBB\xBF") {
                line.erase(0, 3);
            }
            first_line = false;
        }
        if (line.empty() ||
            line.find_first_not_of(" \r\n\t") == std::string::npos) {
            continue;
        }

        std::stringstream ss(line);
        std::string value;
        TrainingData data;

        // Lê os atributos de entrada (X)
        for (int i = 0; i < input_size && std::getline(ss, value, ','); i++) {
            float val;
            try {
                val = std::stof(value);
            } catch (std::invalid_argument) {
                std::cout << "Erro na formatação do CSV" << '\n';
            }
            data.input.push_back(val);
        }

        // Lê os rótulos/saída (Y)
        for (int i = 0; i < output_size && std::getline(ss, value, ','); i++) {
            float val;
            try {
                val = std::stof(value);

            } catch (std::invalid_argument) {
                std::cout << "Erro na formatação do CSV" << '\n';
            }
            data.output.push_back(val);
        }

        dataset.push_back(data);
    }
    return dataset;
}

// divisão treino / teste (hold-out dois subconjuntos)
void splitTrainTest(const std::vector<TrainingData> &full_data,
                    float train_ratio, std::vector<TrainingData> &train_data,
                    std::vector<TrainingData> &test_data) {
    // Divide os dados em treino e teste usando o algoritmo randomico do c++
    std::vector<TrainingData> shuffled_data = full_data;

    std::random_device rd;
    std::mt19937 g(rd());
    // embaralha os dados
    std::shuffle(shuffled_data.begin(), shuffled_data.end(), g);

    // define tamanho do treino pelo ratio
    size_t train_size = static_cast<size_t>(shuffled_data.size() * train_ratio);

    // copia uma parte dos dados para treino
    train_data.assign(shuffled_data.begin(),
                      shuffled_data.begin() + train_size);
    // copia o restante para teste
    test_data.assign(shuffled_data.begin() + train_size, shuffled_data.end());

    std::cout << "Treino: " << train_data.size() << " dados\n"
              << "Teste: " << test_data.size() << " dados" << std::endl;
}

// Divisão treino / validação / teste (hold-out três subconjuntos)
// O dataset é embaralhado antes da divisão
// As proporções somam 1.0: train_ratio + val_ratio + (1 - ambos) = teste.
void splitTrainValTest(const std::vector<TrainingData> &full_data,
                       float train_ratio, float val_ratio,
                        std::vector<TrainingData> &train_data,
                        std::vector<TrainingData> &val_data,
                         std::vector<TrainingData> &test_data) {
 
    std::vector<TrainingData> shuffled = full_data;
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(shuffled.begin(), shuffled.end(), g);
 
    size_t n = shuffled.size();
    size_t train_size = static_cast<size_t>(n * train_ratio);
    size_t val_size = static_cast<size_t>(n * val_ratio);
 
    train_data.assign(shuffled.begin(), shuffled.begin() + train_size);
    val_data.assign (shuffled.begin() + train_size, shuffled.begin() + train_size + val_size);
    test_data.assign (shuffled.begin() + train_size + val_size, shuffled.end());
 
    std::cout << "Treino    : " << train_data.size() << " dados ("
              << (int)(train_ratio * 100) << "%)\n"
              << "Validacao : " << val_data.size()   << " dados ("
              << (int)(val_ratio   * 100) << "%)\n"
              << "Teste     : " << test_data.size()  << " dados ("
              << (int)((1.0f - train_ratio - val_ratio) * 100) << "%)"
              << std::endl;
}

void splitTrainValTestStratified(const std::vector<TrainingData> &full_data,
                                  float train_ratio, float val_ratio,
                                  std::vector<TrainingData> &train_data,
                                  std::vector<TrainingData> &val_data,
                                  std::vector<TrainingData> &test_data) {
 
    // Agrupa as amostras pela posição do maior valor no vetor de saída (índice da letra)
    std::map<int, std::vector<TrainingData>> class_map;
    for (const auto &sample : full_data) {
        int class_idx = 0;
        float max_val = sample.output[0];
        for (int i = 1; i < (int)sample.output.size(); i++) {
            if (sample.output[i] > max_val) {
                max_val   = sample.output[i];
                class_idx = i;
            }
        }
        class_map[class_idx].push_back(sample);
    }
 
    std::random_device rd;
    std::mt19937 g(rd());
 
    // Para cada classe: embaralha e divide proporcionalmente
    for (auto &[class_idx, samples] : class_map) {
        std::shuffle(samples.begin(), samples.end(), g);
 
        size_t n          = samples.size();
        size_t train_size = static_cast<size_t>(n * train_ratio);
        size_t val_size   = static_cast<size_t>(n * val_ratio);
        // test_size = n - train_size - val_size
        // (usa o resto exato para não perder nenhuma amostra por arredondamento)
 
        train_data.insert(train_data.end(),
                          samples.begin(),
                          samples.begin() + train_size);
        val_data.insert(val_data.end(),
                        samples.begin() + train_size,
                        samples.begin() + train_size + val_size);
        test_data.insert(test_data.end(),
                         samples.begin() + train_size + val_size,
                         samples.end());
    }
 
    // Re-embaralha cada subconjunto para misturar as classes entre si
    std::shuffle(train_data.begin(), train_data.end(), g);
    std::shuffle(val_data.begin(),   val_data.end(),   g);
    std::shuffle(test_data.begin(),  test_data.end(),  g);
 
    std::cout << "Treino    : " << train_data.size() << " dados ("
              << (int)(train_ratio * 100) << "%)\n"
              << "Validacao : " << val_data.size()   << " dados ("
              << (int)(val_ratio   * 100) << "%)\n"
              << "Teste     : " << test_data.size()  << " dados ("
              << (int)((1.0f - train_ratio - val_ratio) * 100) << "%)"
              << std::endl;
}

void addNoise(std::vector<TrainingData> &data, float noise_ratio,
              unsigned int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> prob(0.0f, 1.0f);

    for (auto &sample : data) {
        // Itera sobre cada pixel da entrada (valores -1 ou 1)
        for (auto &pixel : sample.input) {
            // Com probabilidade noise_ratio, inverte o pixel
            if (prob(gen) < noise_ratio)
                pixel *= -1.0f;
        }
        // sample.output (rótulo) NÃO é alterado — só o input recebe ruído
    }
}

void splitTrainValTestOrdered(const std::vector<TrainingData> &full_data,
                              float train_ratio, float val_ratio,
                              std::vector<TrainingData> &train_data,
                              std::vector<TrainingData> &val_data,
                              std::vector<TrainingData> &test_data) {
    // Sem shuffle: mantém ordem original do arquivo (A, A, ..., B, B, ..., Z)
    size_t n          = full_data.size();
    size_t train_size = static_cast<size_t>(n * train_ratio);
    size_t val_size   = static_cast<size_t>(n * val_ratio);

    train_data.assign(full_data.begin(),
                      full_data.begin() + train_size);
    val_data.assign  (full_data.begin() + train_size,
                      full_data.begin() + train_size + val_size);
    test_data.assign (full_data.begin() + train_size + val_size,
                      full_data.end());

    std::cout << "[SEM EMBARALHAMENTO]\n"
              << "Treino    : " << train_data.size() << " amostras\n"
              << "Validacao : " << val_data.size()   << " amostras\n"
              << "Teste     : " << test_data.size()
              << " amostras (ultimas letras do alfabeto)\n";
}

void mergeClasses(std::vector<TrainingData> &data, char from_class, char to_class) {
    int from_idx = from_class - 'A';
    int to_idx   = to_class   - 'A';
    for (auto &sample : data) {
        // Se esta amostra tem 'from_class' como alvo, remapeia para 'to_class'
        if (sample.output[from_idx] == 1.0f) {
            sample.output[from_idx] = -1.0f; // remove O como alvo
            sample.output[to_idx]   =  1.0f; // define D como alvo
        }
    }
}

// Carregamento do dataset de Caracteres (arquivos X e Y separados)
std::vector<TrainingData> loadCharacterData(const std::string &x_filename,
                                            const std::string &y_filename) {
    std::vector<TrainingData> dataset;
    std::ifstream x_file(x_filename);
    std::ifstream y_file(y_filename);
    std::string x_line, y_line;

    if (!x_file.is_open() || !y_file.is_open()) {
        std::cerr << "Erro ao abrir os arquivos de caracteres!" << std::endl;
        return dataset;
    }

    bool is_first_line = true;

    // Lê as linhas dos dois arquivos simultaneamente
    while (std::getline(x_file, x_line) && std::getline(y_file, y_line)) {

        // Limpa o BOM (Byte Order Mark)
        if (is_first_line) {
            if (x_line.size() >= 3 && x_line.substr(0, 3) == "\xEF\xBB\xBF")
                x_line.erase(0, 3);
            if (y_line.size() >= 3 && y_line.substr(0, 3) == "\xEF\xBB\xBF")
                y_line.erase(0, 3);
            is_first_line = false;
        }

        if (x_line.empty() || y_line.empty() ||
            y_line.find_first_not_of(" \r\n\t") == std::string::npos) {
            continue;
        }

        std::stringstream ss(x_line);
        std::string value;
        TrainingData data;

        try {
            for (int i = 0; i < 120 && std::getline(ss, value, ','); i++) {
                data.input.push_back(std::stof(value));
            }

            // Converte a letra para um vetor One-Hot Encoding (-1 em todas as
            // posições menos a correta)
            data.output = std::vector<float>(26, -1.0f);

            // Remove espaços e quebras de linha que possam ter vindo junto com
            // a letra
            y_line.erase(
                std::remove_if(y_line.begin(), y_line.end(), ::isspace),
                y_line.end());
            char letra = y_line[0];

            if (letra >= 'A' && letra <= 'Z') {
                int index = letra - 'A';
                data.output[index] =
                    1.0f; // Coloca 1 apenas na posição da letra correta
            }

            dataset.push_back(data);
        } catch (const std::invalid_argument &e) {
            std::cerr << "Erro ao converter valor na linha X: " << x_line
                      << std::endl;
            exit(1);
        }
    }
    return dataset;
}

<<<<<<< HEAD
void exportHyperparameters(const std::string &filename,
                           int input_size, int output_size,
                           const std::vector<int> &hidden_sizes,
                           int epocas, float threshold, float learning_rate,
                           float val_ratio, int patience,
=======
void exportHyperparameters(const std::string &filename, int input_size,
                           int output_size,
                           const std::vector<int> &hidden_sizes, int epocas,
                           float threshold, float learning_rate,
>>>>>>> 30ce82841efac84a69a79c1e9abf88365cffb2db
                           int epocas_executadas, float erro_final) {
    // abre o arquivo de exportar
    std::ofstream f(filename);
    if (!f.is_open()) {
        std::cerr << "Erro ao criar arquivo: " << filename << std::endl;
        return;
    }
    // escreve hiperparametros
    f << std::fixed << std::setprecision(8);
    f << "---- Hiperparametros de inicializacao ----" << std::endl;
    f << "Tamanho da entrada     : " << input_size << std::endl;
    f << "Tamanho da saida       : " << output_size << std::endl;
    f << "Camadas escondidas     : [";
    for (size_t i = 0; i < hidden_sizes.size(); i++) {
        f << hidden_sizes[i];
        if (i + 1 < hidden_sizes.size())
            f << ", ";
    }
    f << "]" << std::endl;
    f << "Epocas maximas         : " << epocas << std::endl;
    f << "Threshold              : " << threshold << std::endl;
    f << "Taxa de aprendizado    : " << learning_rate << std::endl;
    f << "Proporcao de validacao : " << val_ratio << std::endl;
    f << "Paciencia (patience)   : " << patience << std::endl;

    f << std::endl;
    f << "---- Resultados finais ----" << std::endl;
    f << "Epocas executadas      : " << epocas_executadas << std::endl;
    f << "Erro final (MSE medio) : " << erro_final << std::endl;

    std::cout << "Arquivo com hiperparametros: " << filename << std::endl;
}

void exportWeights(const std::string &filename, const WeightSnapshot &snap,
                   const std::string &label) {
    std::ofstream f(filename); // abre arquivo
    if (!f.is_open()) {
        std::cerr << "Erro ao criar arquivo: " << filename << std::endl;
        return;
    }

    // Escreve pesos
    f << std::fixed << std::setprecision(8);

    f << "# " << label << std::endl;
    f << "# Formato: layer_idx,tipo ,neuron_idx,valor_0,valor_1,..."
      << std::endl;
    f << "# tipo=weight -> pesos de um neurônio (um valor por entrada)"
      << std::endl;
    f << "# tipo=bias   -> bias de um neurônio (um unico valor)" << std::endl;

    for (size_t layer = 0; layer < snap.weights.size(); layer++) {
        // Pesos
        for (size_t neuron = 0; neuron < snap.weights[layer].size(); neuron++) {
            f << layer << ",weight," << neuron;
            for (float w : snap.weights[layer][neuron]) {
                f << "," << w;
            }
            f << "\n";
        }
        // Biases
        for (size_t neuron = 0; neuron < snap.biases[layer].size(); neuron++) {
            f << layer << ",bias," << neuron << ","
              << snap.biases[layer][neuron] << "\n";
        }
    }

    std::cout << "Arquivo com Pesos (" << label << "): " << filename
              << std::endl;
}

// Versão com apenas o erro de treino (uma coluna)
void exportEpochErrors(const std::string &filename,
<<<<<<< HEAD
                       const std::vector<float> &train_losses) {
    std::ofstream f(filename);
=======
                       const std::vector<float> &epoch_losses) {
    std::ofstream f(filename); // abre arquivo
>>>>>>> 30ce82841efac84a69a79c1e9abf88365cffb2db
    if (!f.is_open()) {
        std::cerr << "Erro ao criar arquivo: " << filename << std::endl;
        return;
    }

    // exporta erros por epoca
    f << std::fixed << std::setprecision(8);
    f << "epoca,erro_treino_mse" << std::endl;
    for (size_t i = 0; i < train_losses.size(); i++) {
        f << (i + 1) << "," << train_losses[i] << "\n";
    }

    std::cout << "Arquivo com Erros por epoca : " << filename << std::endl;
}

// Versão com erro de treino e validação (duas colunas) para plotar as duas curvas
void exportEpochErrors(const std::string &filename,
                       const std::vector<float> &train_losses,
                       const std::vector<float> &val_losses) {

    std::ofstream f(filename);
    if (!f.is_open()) {
        std::cerr << "Erro ao criar arquivo: " << filename << std::endl;
        return;
    }
 
    f << std::fixed << std::setprecision(8);
    f << "epoca,erro_treino_mse,erro_val_mse\n";
    for (size_t i = 0; i < train_losses.size(); i++) {
        float val = (i < val_losses.size()) ? val_losses[i] : 0.0f;
        f << (i + 1) << "," << train_losses[i] << "," << val << "\n";
    }
 
    std::cout << "Arquivo com Erros por epoca (treino + val): " << filename << std::endl;
}

void exportTestPredictions(const std::string &filename,
                           const std::vector<TrainingData> &test_data,
                           const std::vector<std::vector<float>> &predictions) {
    std::ofstream f(filename);
    if (!f.is_open()) {
        std::cerr << "Erro ao criar arquivo: " << filename << std::endl;
        return;
    }

    f << std::fixed << std::setprecision(6);

    // índice, letra esperada, letra predita, acerto,
    // saída para cada classe (A..Z)
    f << "idx,letra_esperada,letra_predita,acerto";
    for (char c = 'A'; c <= 'Z'; c++) {
        f << ",saida_" << c;
    }
    f << std::endl;

    for (size_t i = 0; i < test_data.size() && i < predictions.size(); i++) {

        // letra esperada
        int expected_idx = 0;
        float expected_max = test_data[i].output[0];
        for (int j = 1; j < 26; j++) {
            if (test_data[i].output[j] > expected_max) {
                expected_max = test_data[i].output[j];
                expected_idx = j;
            }
        }

        // letra predita
        int predicted_idx = 0;
        float predicted_max = predictions[i][0];
        for (int j = 1; j < 26; j++) {
            if (predictions[i][j] > predicted_max) {
                predicted_max = predictions[i][j];
                predicted_idx = j;
            }
        }

        char expected_letter = 'A' + expected_idx;
        char predicted_letter = 'A' + predicted_idx;
        int acerto = (expected_letter == predicted_letter) ? 1 : 0;

        f << i << "," << expected_letter << "," << predicted_letter << ","
          << acerto;
        for (int j = 0; j < 26; j++) {
            f << "," << predictions[i][j];
        }
        f << "\n";
    }

    std::cout << "Arquivo com Predicoes de teste: " << filename << std::endl;
}
