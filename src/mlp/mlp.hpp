// Faz com que esse header file seja incluido apenas uma vez (similar ao Header
// Guards)
#pragma once

#include <vector>

// Matrix representa uma matrix NxM
typedef std::vector<std::vector<float>> Matrix;

// ActivationFunctionType representa o tipo de função de ativação a ser usada em
// uma rede neural
enum ActivationFunctionType {
    ReLu,
    Sigmoid,
};

// Layer representa uma Camada dentro da rede neural, com seus pesos, viéses e
// valores usados para computar o backpropagation
class Layer {
  public:
    Matrix weights;
    std::vector<float> biases;
    std::vector<float> activations;
    Matrix weight_gradients;
    std::vector<float> bias_gradients;
    std::vector<float> deltas;

    // Construtor de Camada
    Layer(int num_neurons, int num_inputs);
};

// TrainingSample representa uma amostra de treinamento, contendo um input e um
// output esperado
typedef struct TrainingSample {
    std::vector<float> input;
    std::vector<float> output;
} TrainingData;

// MultiLayerPerceptronNetwork é uma implementação autoral de uma rede neural
// MLP
class MLPNetwork {
  private:
    // camadas da rede
    std::vector<Layer> layers;
    // tipo de função de ativação. ex.: ReLU, sigmoid
    ActivationFunctionType activation_type;

    // forwardPropagation realiza a propagação para frente da rede, computando
    // as ativações de cada camada
    std::vector<float> forwardPropagation(const std::vector<float> &input);
    // backwardPropagation realiza a propagação para trás da rede, computando os
    // deltas de cada camada
    void backwardPropagation(const std::vector<float> &input,
                             const std::vector<float> &expected_output);
    // updateWeightsAndBiases atualiza os pesos e os biases da rede
    void updateWeightsAndBiases(float learning_rate);

    // trainForEpoch realiza o treinamento de uma época, computando os deltas e
    // atualizando os pesos e os biases
    float trainForEpoch(const std::vector<TrainingData> &data,
                        float learning_rate);

    // activation_function é a função de ativação da rede
    float activation_function(float z);
    // activation_function_derivative é a derivada da função de ativação
    float activation_function_derivative(float z);

  public:
    MLPNetwork(
        int input_size, int output_size, std::vector<int> hidden_layer_sizes,
        ActivationFunctionType activation_type = ActivationFunctionType::ReLu);

    // predict realiza a predição da rede para um dado de entrada
    std::vector<float> predict(const std::vector<float> &input);
    // train realiza o treinamento da rede para um dado de entrada e saída
    void train(const std::vector<TrainingData> &data, int epoches,
               float threshold, float learningRate);
};
