#include "./mlp.hpp"
#include <assert.h>
#include <cmath>
#include <iostream>
#include <math.h>
#include <random>
#include <limits>

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
    float limit = sqrt(6.0f / (num_inputs + num_neurons));
    std::uniform_real_distribution<> dist(-limit, limit);

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

    // Pega os pesos iniciais
    initial_weights = captureWeights();
}

// Pega os pesos e biases atuais
WeightSnapshot MLPNetwork::captureWeights() const {
    WeightSnapshot snap;
    for (const auto &layer : layers) {
        snap.weights.push_back(layer.weights);
        snap.biases.push_back(layer.biases);
    }
    return snap;
}

const WeightSnapshot &MLPNetwork::getInitialWeights() const {
    return initial_weights;
}

WeightSnapshot MLPNetwork::getFinalWeights() const { return captureWeights(); }

int MLPNetwork::getNumLayers() const { return (int)layers.size(); }

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

        // Reseta gradientes
        for (auto &layer : layers) {
            layer.weight_gradients =
                Matrix(layer.weights.size(),
                       std::vector<float>(layer.weights[0].size()));
            layer.bias_gradients = std::vector<float>(layer.biases.size());
        }
        // realiza forward propagation
        auto output = forwardPropagation(sample.input);

        // realiza backward propagation
        backwardPropagation(sample.input, target);

        // Erro (loss) acumulada
        total_loss += MSE(output, target);

        updateWeightsAndBiases(learningRate);
    }
    return total_loss / data.size();
}

std::vector<float>
MLPNetwork::forwardPropagation(const std::vector<float> &input) {
    // "última ativação" ou estado de ativação atual da camada
    std::vector<float> activation(input.begin(), input.end());

    // Para cada layer
    for (auto &layer : layers) {
        auto next_activation = std::vector<float>(layer.biases.size(), 0.0f);
        // calculamos o resultado da função de ativação para cada neuronio da
        // próxima camada
        for (size_t neuron = 0; neuron < layer.biases.size(); neuron++) {
            float z = layer.biases[neuron];
            for (size_t input = 0; input < activation.size(); input++) {
                z += layer.weights[neuron][input] * activation[input];
            }

            next_activation[neuron] = activation_function(z);
        }

        // atualizamos o estado das ativações da camada atual
        layer.activations =
            std::vector<float>(next_activation.begin(), next_activation.end());
        // a ativação atual se torna a última ativação
        activation = next_activation;
    }
    // ultima camada possui o resultado final
    return activation;
}

// Etapa de Backward Propagation
void MLPNetwork::backwardPropagation(
    const std::vector<float> &input,
    const std::vector<float> &expected_output) {

    // deltas do layer de saída
    auto &output_layer = layers.back(); // camada de saída
    auto output_layer_id = layers.size() - 1;
    // computa o delta de cada neuronio na camada de saída usando o MSE
    for (size_t neuron = 0; neuron < output_layer.deltas.size(); neuron++) {
        auto prediction = output_layer.activations[neuron];
        auto target = expected_output[neuron];
        auto activation_func_deriv = activation_function_derivative(prediction);

        output_layer.deltas[neuron] =
            MSEDerivative(prediction, target) * activation_func_deriv;
    }

    // Camada escondida
    for (int i = output_layer_id - 1; i >= 0; i--) {
        auto &layer = layers[i];
        auto &next_layer = layers[i + 1];

        // Computa o delta da camada escondida com a derivada da função de
        // ativação
        for (size_t neuron = 0; neuron < layer.deltas.size(); neuron++) {
            auto activation_func_deriv =
                activation_function_derivative(layer.activations[neuron]);
            float sum_delta_weights = .0;
            for (size_t next_neuron = 0; next_neuron < next_layer.deltas.size();
                 next_neuron++) {
                sum_delta_weights += next_layer.deltas[next_neuron] *
                                     next_layer.weights[next_neuron][neuron];
            }
            layer.deltas[neuron] = activation_func_deriv * sum_delta_weights;
        }
    }

    // Calcular gradientes
    for (size_t i = 0; i < layers.size(); i++) {
        auto &layer = layers[i];

        std::vector<float> layer_input;
        if (i == 0) {
            layer_input = input;
        } else {
            layer_input = layers[i - 1].activations;
        }

        for (size_t neuron = 0; neuron < layer.biases.size(); neuron++) {
            layer.bias_gradients[neuron] += layer.deltas[neuron];
        }

        for (size_t neuron = 0; neuron < layer.deltas.size(); neuron++) {
            for (size_t j = 0; j < layer_input.size(); j++) {
                layer.weight_gradients[neuron][j] +=
                    layer.deltas[neuron] * layer_input[j];
            }
        }
    }
}

void MLPNetwork::updateWeightsAndBiases(float learning_rate) {
    // Para cada camada da rede
    for (size_t i = 0; i < layers.size(); i++) {
        auto &layer = layers[i];

        // atualiza os pesos
        for (size_t neuron = 0; neuron < layer.weights.size(); neuron++) {
            for (size_t input = 0; input < layer.weights[neuron].size();
                 input++) {
                auto gradient = layer.weight_gradients[neuron][input];
                layer.weights[neuron][input] -= learning_rate * gradient;
            }
        }

        // atualiza os biases
        for (size_t neuron = 0; neuron < layer.biases.size(); neuron++) {
            auto gradient = layer.bias_gradients[neuron];
            layer.biases[neuron] -= learning_rate * gradient;
        }
    }
}

// Usado a cada época para calcular o erro de validação
// Realiza apenas o forward pass para cada amostra (sem backpropagation)
float MLPNetwork::computeLoss(const std::vector<TrainingData> &data) {
    float total_loss = 0.0f;
    for (const auto &sample : data) {
        auto output = forwardPropagation(sample.input);
        total_loss += MSE(output, sample.output);
    }
    return total_loss / (float)data.size();
}

// treinamento com Early Stopping baseado no erro de validação
// Retorna TrainResult com vetores de train_losses e val_losses por época
TrainResult MLPNetwork::train(const std::vector<TrainingData> &train_data,
                               const std::vector<TrainingData> &val_data,
                               int epochs, float threshold, float learning_rate,
                               int patience) {
    TrainResult result;
    result.train_losses.reserve(epochs);
    result.val_losses.reserve(epochs);
    
    float best_val_loss      = std::numeric_limits<float>::infinity();
    int   patience_counter   = 0;
    WeightSnapshot best_weights = captureWeights(); // começa com os pesos atuais

    // para cada época
    for (int epoch = 0; epoch < epochs; epoch++) {
        // executa um treinamento
        float train_loss = trainForEpoch(train_data, learning_rate);
        result.train_losses.push_back(train_loss);

        // calcula o erro de validação
        float val_loss = computeLoss(val_data);
        result.val_losses.push_back(val_loss);

        std::cout << "Epoca " << epoch + 1 << "/" << epochs
                  << " | Treino (MSE): " << train_loss
                  << " | Validacao (MSE): " << val_loss
                  << std::endl;

        // verifica melhoria no erro de validação (early stopping)
        if (val_loss < best_val_loss) {
            // se houve melhoria: salva os melhores pesos e reseta a paciência
            best_val_loss    = val_loss;
            patience_counter = 0;
            best_weights     = captureWeights();
        } else {
            // sem melhoria: incrementa o contador de paciência
            patience_counter++;
 
            if (patience_counter >= patience) {
                // Parada antecipada ativada, restaura os pesos do melhor momento
                std::cout << "Parada antecipada: val_loss nao melhorou por "
                          << patience << " epocas consecutivas.\n"
                          << "Restaurando pesos do melhor momento "
                          << "(val_loss = " << best_val_loss << ")." << std::endl;
 
                for (size_t i = 0; i < layers.size(); i++) {
                    layers[i].weights = best_weights.weights[i];
                    layers[i].biases  = best_weights.biases[i];
                }
                break;
            }
        }
 
        // parada por threshold no erro de treino (critério secundário)
        if (train_loss < threshold) {
            std::cout << "Treinamento encerrado: erro de treino abaixo do threshold ("
                      << threshold << ")." << std::endl;
            break;
        }
    }
 
    return result;
}

std::vector<float> MLPNetwork::predict(const std::vector<float> &input) {
    // realiza a predição com a função forwardPropagation (o dado 'atravessa a
    // rede')
    return forwardPropagation(input);
}

float MLPNetwork::activation_function(float z) {
    switch (activation_type) {
    case ActivationFunctionType::ReLu:
        return std::max(0.0f, z);
    case ActivationFunctionType::SigmoidStandard:
        // Sigmoid um pouco alterada para deixar o resultado entre -1 e 1, como
        // nos dados do dataset
        return (2.0f / (1.0f + std::exp(-z))) - 1.0f;
    case ActivationFunctionType::SigmoidBipolar:
        return tanh(z);
    default:
        return z;
    }
}

float MLPNetwork::activation_function_derivative(float z) {
    // Retorna a derivada da função de ativação para o valor z
    switch (activation_type) {
    case ActivationFunctionType::ReLu:
        return z > 0.0f ? 1.0f : 0.0f;
    case ActivationFunctionType::SigmoidStandard:
        return 0.5f * (1.0f - z * z);
    case ActivationFunctionType::SigmoidBipolar:
        return 1.0f - z * z;
    default:
        return 1.0f;
    }
}
