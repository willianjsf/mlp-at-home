#pragma once

#include "../mlp/mlp.hpp"
#include <string>
#include <vector>

// Cria o diretório (e subdiretórios) caso não existam
void ensureDir(const std::string &path);

// Função para carregar os dados do CSV
std::vector<TrainingData> loadData(const std::string &filename, int input_size,
                                   int output_size);

// Função para separar os dados em Treino e Teste
void splitTrainTest(const std::vector<TrainingData> &full_data,
                    float train_ratio, std::vector<TrainingData> &train_data,
                    std::vector<TrainingData> &test_data);

// Função para dividir o dataset em três subconjuntos: treino, validação e teste
void splitTrainValTest(const std::vector<TrainingData> &full_data,
                        float train_ratio, float val_ratio,
                       std::vector<TrainingData> &train_data,
                         std::vector<TrainingData> &val_data,
                        std::vector<TrainingData> &test_data);

// Versão estratificada da divisão treino/val/test, que mantém a proporção de classes em cada subconjunto
void splitTrainValTestStratified(const std::vector<TrainingData> &full_data,
                                  float train_ratio, float val_ratio,
                                  std::vector<TrainingData> &train_data,
                                  std::vector<TrainingData> &val_data,
                                  std::vector<TrainingData> &test_data);

// Inverte aleatoriamente 'noise_ratio' fração dos pixels de cada amostra.
// Usado para criar variações ruidosas do conjunto de teste
void addNoise(std::vector<TrainingData> &data, float noise_ratio,
              unsigned int seed = 42);


// Mescla duas classes: reatribui todas as amostras de 'from_class' para 'to_class'.
// Ex.: mergeClasses(data, 'O', 'D') faz a rede aprender D e O como a mesma letra.
void mergeClasses(std::vector<TrainingData> &data, char from_class, char to_class);

// Função para carregar os dados do dataset de Caracteres
std::vector<TrainingData> loadCharacterData(const std::string &x_filename,
                                            const std::string &y_filename);


// Exporta os hiperparâmetros de inicialização e finais
void exportHyperparameters(const std::string &filename,
                           int input_size, int output_size,
                           const std::vector<int> &hidden_sizes,
                           int epocas, float threshold, float learning_rate,
                           float val_ratio, int patience,
                           int epocas_executadas, float erro_final);

// Exporta os pesos e biases
void exportWeights(const std::string &filename, const WeightSnapshot &snap,
                   const std::string &label);

// Exporta o erro médio
void exportEpochErrors(const std::string &filename,
                       const std::vector<float> &epoch_losses);

// Exporta o erro de treino e de validação por época (duas colunas)
// Versão usada quando há early stopping, permite plotar as duas curvas
void exportEpochErrors(const std::string &filename,
                       const std::vector<float> &train_losses,
                       const std::vector<float> &val_losses);

// Exporta as previsões do conjunto de teste
void exportTestPredictions(const std::string &filename,
                           const std::vector<TrainingData> &test_data,
                           const std::vector<std::vector<float>> &predictions);