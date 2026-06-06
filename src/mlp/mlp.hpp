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
    SigmoidBipolar,
    SigmoidStandard,
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

// Armazena os pesos e biases de todas as camadas
struct WeightSnapshot {
    std::vector<Matrix> weights;
    std::vector<std::vector<float>> biases;
};

// TrainResult encapsula os resultados de uma execução de treinamento:
// os erros MSE por época tanto no conjunto de treino quanto no de validação
struct TrainResult {
    std::vector<float> train_losses; // MSE médio por época no conjunto de treino
    std::vector<float> val_losses;   // MSE médio por época no conjunto de validação
};

// MultiLayerPerceptronNetwork é uma implementação autoral de uma rede neural
// MLP
class MLPNetwork {
  private:
    // camadas da rede
    std::vector<Layer> layers;
    // tipo de função de ativação. ex.: ReLU, sigmoid
    ActivationFunctionType activation_type;

    // snapshot dos pesos iniciais
    WeightSnapshot initial_weights;

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

    // Pega os pesos atuais
    WeightSnapshot captureWeights() const;

  public:
    MLPNetwork(int input_size, int output_size,
               std::vector<int> hidden_layer_sizes,
               ActivationFunctionType activation_type =
                   ActivationFunctionType::SigmoidStandard);

    // predict realiza a predição da rede para um dado de entrada
    std::vector<float> predict(const std::vector<float> &input);

    // computeLoss calcula o MSE médio de um conjunto sem atualizar os pesos
    // é usado para avaliar o erro no conjunto de validação a cada época
    float computeLoss(const std::vector<TrainingData> &data);

    // train executa o treinamento completo com parada antecipada (early stopping)
    // retorna um TrainResult com os vetores de erro por época (treino e validação)
    TrainResult train(const std::vector<TrainingData> &train_data,
                      const std::vector<TrainingData> &val_data,
                      int epochs, float threshold, float learning_rate,
                      int patience = 10);

    // getInitialWeights retorna uma snapshot dos pesos e biases da rede
    const WeightSnapshot &getInitialWeights() const;

    // getFinalWeights retorna uma snapshot dos pesos e biases da rede após o
    // treinamento
    WeightSnapshot getFinalWeights() const;

    // getNumLayers retorna o número de camadas da rede
    int getNumLayers() const;
};
