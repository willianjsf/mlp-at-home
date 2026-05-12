// Faz com que esse header file seja incluido apenas uma vez (similar ao Header
// Guards)
#pragma once

#include <random>

enum ActivationFunctionType {
    ReLu,
    Sigmoid,
};

// MultiLayerPerceptronNetwork é uma implementação autoral de uma rede neural
// MLP
#include <vector>
class MultiLayerPerceptronNetwork {
  private:
    std::vector<std::vector<std::vector<float>>>
        weights;                            // matrix de pesos por camada
    std::vector<std::vector<float>> biases; // matriz de viéses

    std::vector<int> layers_sizes;

    enum ActivationFunctionType activation_function_type =
        ActivationFunctionType::ReLu;

    std::vector<float> activationFunction(std::vector<float> &z);

  public:
    // construtor
    MultiLayerPerceptronNetwork(
        int input_size, int output_size, std::vector<int> hidden_layer_sizes,
        ActivationFunctionType activation_function_type =
            ActivationFunctionType::ReLu) {
        this->activation_function_type = activation_function_type;

        this->layers_sizes = {input_size};
        this->layers_sizes.insert(this->layers_sizes.end(),
                                  hidden_layer_sizes.begin(),
                                  hidden_layer_sizes.end());
        this->layers_sizes.push_back(output_size);

        // Gera números aleatorios
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_real_distribution<> dist(-1.0, 1.0); // [-1,1)

        // inicializa pesos e viés
        for (size_t i = 0; i < this->layers_sizes.size() - 1; i++) {
            this->weights.push_back(std::vector<std::vector<float>>());
            this->biases.push_back(std::vector<float>());
            for (int j = 0; j < this->layers_sizes[i + 1]; j++) {
                this->weights[i].push_back(std::vector<float>());
                this->biases[i].push_back(dist(rng));
                for (int k = 0; k < this->layers_sizes[i]; k++) {
                    this->weights[i][j].push_back(dist(rng));
                }
            }
        }
    }

    int get_num_layers() { return this->layers_sizes.size(); }
    int get_input_size() { return this->layers_sizes[0]; }
    int get_output_size() { return this->layers_sizes.back(); }
    int get_hidden_layer_size() { return this->layers_sizes.size() - 2; }

    void train(std::vector<std::vector<float>> X,
               std::vector<std::vector<float>> y, int epoches, float tolerance);

    std::vector<float> predict(std::vector<float> &data);
};
