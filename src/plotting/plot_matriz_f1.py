import os
import string

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def calcular_matriz_confusao(y_real, y_predito, num_classes):
    matriz = np.zeros((num_classes, num_classes), dtype=int)
    for real, pred in zip(y_real, y_predito):
        matriz[real][pred] += 1
    return matriz


def plotar_matriz_confusao(matriz, num_classes, labels):
    # Aumentando o tamanho da figura porque temos 26 classes
    fig, ax = plt.subplots(figsize=(10, 10))

    cax = ax.imshow(matriz, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    ax.set_title("Matriz de Confusão do MLP (Dados de Teste)")
    ax.set_xlabel("Letra Predita (Saída da Rede)")
    ax.set_ylabel("Letra Real (Gabarito)")

    # Adicionando o alfabeto nos eixos X e Y
    ticks = np.arange(num_classes)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Escreve a contagem dentro dos quadrados
    limite = matriz.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            # Condição para escrever o número APENAS se for maior que zero.
            # Isso evita que a grade fique lotada de "0" e poluída visualmente.
            if matriz[i, j] > 0:
                ax.text(
                    j,
                    i,
                    format(matriz[i, j], "d"),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if matriz[i, j] > limite else "black",
                )

    fig.tight_layout()
    os.makedirs("graficos", exist_ok=True)
    caminho = os.path.join("graficos", "grafico_6_matriz_confusao.png")
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    print(f"  → Matriz de Confusão salva em: {caminho}")
    plt.close()


def calcular_macro_f1(matriz_confusao):
    """
    Calcula o Macro F1-Score a partir de uma matriz de confusão do NumPy.
    """
    num_classes = matriz_confusao.shape[0]
    f1_scores = []

    for i in range(num_classes):
        # Verdadeiros Positivos (TP) = diagonal principal
        tp = matriz_confusao[i, i]

        # Falsos Positivos (FP) = soma da coluna 'i', menos o TP
        fp = np.sum(matriz_confusao[:, i]) - tp

        # Falsos Negativos (FN) = soma da linha 'i', menos o TP
        fn = np.sum(matriz_confusao[i, :]) - tp

        # Ignora a classe se ela não apareceu no teste e a rede não a previu
        if tp + fp == 0 and tp + fn == 0:
            continue

        # Calcula Precision e Recall evitando divisão por zero
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Calcula o F1-Score da classe atual
        if precision + recall > 0:
            f1_classe = 2 * (precision * recall) / (precision + recall)
        else:
            f1_classe = 0.0

        f1_scores.append(f1_classe)

    # Macro F1 é a média aritmética dos F1-scores individuais
    if not f1_scores:
        return 0.0
    return np.mean(f1_scores)


if __name__ == "__main__":
    arquivo_teste = "predicoes_teste.csv"

    if os.path.exists(arquivo_teste):
        df = pd.read_csv(arquivo_teste)

        # Definimos o alfabeto como nossas 26 classes (A=0, B=1... Z=25)
        alfabeto = list(string.ascii_uppercase)
        mapa_letras = {letra: idx for idx, letra in enumerate(alfabeto)}

        # Convertendo as strings 'A', 'B' dos dados para índices inteiros 0, 1
        df["y_real_idx"] = df["letra_esperada"].map(mapa_letras)
        df["y_predito_idx"] = df["letra_predita"].map(mapa_letras)

        # Prevenção: Remove possíveis linhas vazias no fim do arquivo
        df = df.dropna(subset=["y_real_idx", "y_predito_idx"])

        # Isolando os arrays numéricos que a nossa função precisa
        y_real = df["y_real_idx"].astype(int).tolist()
        y_predito = df["y_predito_idx"].astype(int).tolist()

        print("Calculando matriz de confusão com as previsões do modelo...")

        # Cálculo e plotagem
        num_classes_total = 26
        matriz = calcular_matriz_confusao(y_real, y_predito, num_classes_total)

        # Calcula e imprime o Macro F1-Score
        f1_final = calcular_macro_f1(matriz)
        print(f"Desempenho Geral (Macro F1-Score): {f1_final * 100:.2f}%")

        plotar_matriz_confusao(matriz, num_classes_total, labels=alfabeto)

    else:
        print(f"Erro: O arquivo '{arquivo_teste}' não foi encontrado nesta pasta.")
