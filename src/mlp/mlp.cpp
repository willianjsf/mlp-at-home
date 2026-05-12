
#include "./mlp.hpp"
#include <assert.h>
#include <math.h>
#include <random>

// Construtor de Camada
Layer::Layer(int num_neurons, int num_inputs) {
    // inicializa tudo com 0 (zero)
    biases = std::vector<float>(num_neurons);
    activations = std::vector<float>(num_neurons);
    deltas = std::vector<float>(num_neurons);
    bias_gradients = std::vector<float>(num_neurons);
    weights = Matrix(num_neurons, std::vector<float>(num_inputs));
    weight_gradients = Matrix(num_neurons, std::vector<float>(num_inputs));

    // Preenche os pesos com números aleatórios
    std::random_device rnd;
    std::mt19937 gen(rnd());
    std::uniform_real_distribution<> dist(-1.0, 1.0);

    for (int i = 0; i < num_neurons; ++i) {
        for (int j = 0; j < num_inputs; ++j) {
            weights[i][j] = dist(gen);
        }
    }
}

// Constrututor da rede neural
MLPNetwork::MLPNetwork(int input_size, int output_size,
                       std::vector<int> hidden_layer_sizes,
                       ActivationFunctionType activation_type) {
    this->activation_type = activation_type;

    // Cria camadas escondidas + camada de saída
    // A camada de entrada é representada pelo valor de entrada da primeira
    // camada escondida
    for (size_t i = 0; i < hidden_layer_sizes.size(); i++) {
        layers.push_back(
            Layer(hidden_layer_sizes[i],
                  i == 0 ? input_size : hidden_layer_sizes[i - 1]));
    }
    layers.push_back(Layer(output_size, hidden_layer_sizes.back()));
}

// MSE (Mean Square Error)
static float MSE(const std::vector<float> &a, const std::vector<float> &b) {
    float sum = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        sum += pow(a[i] - b[i], 2);
    }
    return sum / a.size();
}

// Derivada da MSE
static float MSEDerivative(float prediction, float target) {
    return 2 * (prediction - target);
}

float MLPNetwork::trainForEpoch(const std::vector<TrainingData> &data,
                                float learningRate) {
    float total_loss = 0.0;
    for (auto &sample : data) {
        auto input = sample.input;
        auto target = sample.output;

        // Reset gradients
        for (auto &layer : layers) {
            layer.weight_gradients =
                Matrix(layer.weights.size(),
                       std::vector<float>(layer.weights[0].size()));
            layer.bias_gradients = std::vector<float>(layer.biases.size());
        }
        // Forward pass
        auto output = forwardPropagation(sample.input);

        // Backward pass
        backwardPropagation(sample.input, target);

        // Acumulate loss
        auto loss = MSE(output, target);
        total_loss += loss;

        // Update weights
        updateWeightsAndBiases(learningRate);
    }
    return total_loss / data.size();
}

std::vector<float>
MLPNetwork::forwardPropagation(const std::vector<float> &input) {

    std::vector<float> activation(input.begin(), input.end());

    for (auto &layer : layers) {
        auto next_activation = std::vector<float>(layer.biases.size());

        for (int neuron = 0; layer.biases.size(); neuron++) {
            auto z = layer.biases[neuron];

            for (size_t input = 0; input < activation.size(); input++) {
                z += layer.weights[neuron][input] * activation[input];
            }

            next_activation[neuron] = activation_function(z);
        }

        layer.activations =
            std::vector<float>(next_activation.begin(), next_activation.end());
        activation = next_activation;
    }

    return activation;
}

void MLPNetwork::backwardPropagation(
    const std::vector<float> &input,
    const std::vector<float> &expected_output) {

    auto output = forwardPropagation(input);

    // deltas do layer de saída
    auto output_layer = layers.back();
    auto output_layer_id = layers.size() - 1;
    for (size_t neuron = 0; neuron < output_layer.deltas.size() - 1; neuron++) {
        auto prediction = output[neuron];
        auto target = expected_output[neuron];
        auto activation_func_deriv = activation_function_derivative(prediction);

        output_layer.deltas[neuron] =
            MSEDerivative(prediction, target) * activation_func_deriv;
    }

    // Camada escondida
    for (int i = output_layer_id - 1; i >= 0; i--) {
        auto layer = layers[i];
        auto next_layer = layers[i + 1];

        if (i == 0) {
            auto previous_activation = input;
        } else {
            auto previous_activation = layers[i - 1].activations;
        }

        for (float neuron : layer.deltas) {
            auto activation_func_deriv = activation_function_derivative(neuron);
            float sum_delta_weights = .0;
            for (size_t next_neuron = 0; next_neuron < next_layer.deltas.size();
                 next_neuron++) {
                sum_delta_weights += next_layer.deltas[next_neuron] *
                                     next_layer.weights[neuron][next_neuron];
            }
            layer.deltas[neuron] = activation_func_deriv * sum_delta_weights;
        }
    }

    // Calcular gradientes
    for (size_t i = 0; i < layers.size() - 1; i++) {
        auto layer = layers[i];

        std::vector<float> layer_input;
        if (i == 0) {
            layer_input = input;
        } else {
            layer_input = layers[i - 1].activations;
        }

        for (size_t neuron = 0; neuron < layer.biases.size() - 1; neuron++) {
            layer.bias_gradients[neuron] += layer.deltas[neuron];
        }

        for (size_t neuron = 0; neuron < layer.deltas.size() - 1; neuron++) {
            for (size_t j = 0; j < layer_input.size(); j++) {
                layer.weight_gradients[neuron][j] +=
                    layer.deltas[neuron] * layer_input[j];
            }
        }
    }
}

void MLPNetwork::updateWeightsAndBiases(float learning_rate) {
    // Para cada camada da rede
    for (size_t i = 0; i < layers.size() - 1; i++) {
        auto layer = layers[i];

        // atualiza os pesos
        for (size_t neuron = 0; neuron < layer.weights.size() - 1; neuron++) {
            for (size_t input = 0; input < layer.weights[neuron].size();
                 input++) {
                auto gradient = layer.weight_gradients[neuron][input];
                layer.weights[neuron][input] -= learning_rate * gradient;
            }
        }

        // atualiza os biases
        for (size_t neuron = 0; neuron < layer.biases.size() - 1; neuron++) {
            auto gradient = layer.bias_gradients[neuron];
            layer.biases[neuron] -= learning_rate * gradient;
        }
    }
}

void MLPNetwork::train(const std::vector<TrainingData> &data, int epoches,
                       float threshold, float learning_rate) {
    // para cada época
    for (int epoch = 0; epoch < epoches; epoch++) {
        // executa um treinamento
        float average_loss = trainForEpoch(data, learning_rate);

        // se a taxa de perda for menor que o threshold podemos parar a
        // simulação
        if (average_loss < threshold) {
            break;
        }
    }
}

std::vector<float> MLPNetwork::predict(const std::vector<float> &input) {
    return forwardPropagation(input);
}

float MLPNetwork::activation_function(float z) {
    switch (activation_type) {
    case ActivationFunctionType::ReLu:
        return std::max(0.0f, z);
    case ActivationFunctionType::Sigmoid:
        return 1.0f / (1.0f + std::exp(-z));
    default:
        return z;
    }
}

float MLPNetwork::activation_function_derivative(float z) {
    switch (activation_type) {
    case ActivationFunctionType::ReLu:
        return z > 0.0f ? 1.0f : 0.0f;
    case ActivationFunctionType::Sigmoid:
        return z * (1.0f - z);
    default:
        return 1.0f;
    }
}
