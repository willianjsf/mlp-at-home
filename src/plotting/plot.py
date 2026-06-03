"""
plot.py
--------------------
Gera todos os gráficos de análise de hiperparâmetros da rede MLP.

Como usar:
1. Coloque este arquivo na raiz do projeto (mesma pasta dos CSVs).
2. Preencha o dicionário ACURACIAS abaixo com os valores do terminal.
3. Execute: python3 plot.py

Dependências: pip3 install pandas matplotlib --break-system-packages
"""

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# PREENCHA AQUI com os valores anotados do terminal após cada experimento
# Deixe None nos experimentos que ainda não foram executados
# ─────────────────────────────────────────────────────────────────────────────
ACURACIAS = {
    1: 60.90,  # lr=0.0001, hidden={60}, epocas=200
    2: 90.60,  # lr=0.001,  hidden={60}, epocas=200  ← baseline (já feito)
    3: 92.48,  # lr=0.01,   hidden={60}, epocas=200
    4: 87.96,  # lr=0.1,    hidden={60}, epocas=200
    5: 87.96,  # lr=0.001,  hidden={30}, epocas=200
    6: 90.22,  # lr=0.001,  hidden={90}, epocas=200
    7: 89.09,  # lr=0.001,  hidden={120},epocas=200
    8: 92.10,  # lr=0.001,  hidden={60}, epocas=100
    9: 88.72,  # lr=0.001,  hidden={60}, epocas=500
}

# Descrição legível de cada experimento (para legendas e rótulos)
DESCRICOES = {
    1: "lr=0.0001",
    2: "lr=0.001 (baseline)",
    3: "lr=0.01",
    4: "lr=0.1",
    5: "hidden=30",
    6: "hidden=90",
    7: "hidden=120",
    8: "épocas=100",
    9: "épocas=500",
}


# Configurações visuais globais

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "grid.alpha": 0.35,
    }
)

OUTPUT_DIR = "graficos"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Funções auxiliares


def carregar_csv(exp_id):
    """Carrega o CSV de erros de um experimento. Retorna None se não existir."""
    nome = f"erros_por_epoca_exp{exp_id}.csv"
    if not os.path.exists(nome):
        print(
            f"  [aviso] arquivo não encontrado: {nome} — experimento {exp_id} ignorado."
        )
        return None
    df = pd.read_csv(nome)
    return df


def salvar(nome_arquivo):
    caminho = os.path.join(OUTPUT_DIR, nome_arquivo)
    plt.savefig(caminho, bbox_inches="tight")
    print(f"  → salvo em: {caminho}")


# ─────────────────────────────────────────────────────────────────────────────
# Gráfico 1 — Curva de erro do baseline (experimento 2)
# ─────────────────────────────────────────────────────────────────────────────
def grafico_baseline():
    print("\n[Gráfico 1] Curva de erro — baseline (exp 2)")
    df = carregar_csv(2)
    if df is None:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["epoca"], df["erro_medio_mse"], color="#1f77b4", linewidth=1.5)
    ax.set_title("Curva de erro durante o treinamento (baseline)")
    ax.set_xlabel("Época")
    ax.set_ylabel("MSE médio")
    ax.grid(True)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    fig.tight_layout()
    salvar("grafico_1_baseline.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Gráfico 2 — Comparação: taxa de aprendizado (exps 1, 2, 3, 4)
# ─────────────────────────────────────────────────────────────────────────────
def grafico_learning_rate():
    print("\n[Gráfico 2] Comparação — taxa de aprendizado (exps 1, 2, 3, 4)")
    exps = [1, 2, 3, 4]
    cores = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]

    fig, ax = plt.subplots(figsize=(8, 4))
    algum_plotado = False

    for exp_id, cor in zip(exps, cores):
        df = carregar_csv(exp_id)
        if df is None:
            continue
        ax.plot(
            df["epoca"],
            df["erro_medio_mse"],
            label=DESCRICOES[exp_id],
            color=cor,
            linewidth=1.5,
        )
        algum_plotado = True

    if not algum_plotado:
        print("  Nenhum arquivo disponível para este gráfico.")
        plt.close()
        return

    ax.set_title("Influência da taxa de aprendizado")
    ax.set_xlabel("Época")
    ax.set_ylabel("MSE médio")
    ax.legend()
    ax.grid(True)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    fig.tight_layout()
    salvar("grafico_2_learning_rate.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Gráfico 3 — Comparação: tamanho da camada escondida (exps 5, 2, 6, 7)
# ─────────────────────────────────────────────────────────────────────────────
def grafico_hidden_size():
    print("\n[Gráfico 3] Comparação — tamanho da camada escondida (exps 5, 2, 6, 7)")
    exps = [5, 2, 6, 7]
    cores = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]

    fig, ax = plt.subplots(figsize=(8, 4))
    algum_plotado = False

    for exp_id, cor in zip(exps, cores):
        df = carregar_csv(exp_id)
        if df is None:
            continue
        ax.plot(
            df["epoca"],
            df["erro_medio_mse"],
            label=DESCRICOES[exp_id],
            color=cor,
            linewidth=1.5,
        )
        algum_plotado = True

    if not algum_plotado:
        print("  Nenhum arquivo disponível para este gráfico.")
        plt.close()
        return

    ax.set_title("Influência do tamanho da camada escondida")
    ax.set_xlabel("Época")
    ax.set_ylabel("MSE médio")
    ax.legend()
    ax.grid(True)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    fig.tight_layout()
    salvar("grafico_3_hidden_size.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Gráfico 4 — Comparação: número de épocas (exps 8, 2, 9)
# ─────────────────────────────────────────────────────────────────────────────
def grafico_epocas():
    print("\n[Gráfico 4] Comparação — número de épocas (exps 8, 2, 9)")
    exps = [8, 2, 9]
    cores = ["#d62728", "#1f77b4", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(8, 4))
    algum_plotado = False

    for exp_id, cor in zip(exps, cores):
        df = carregar_csv(exp_id)
        if df is None:
            continue
        ax.plot(
            df["epoca"],
            df["erro_medio_mse"],
            label=DESCRICOES[exp_id],
            color=cor,
            linewidth=1.5,
        )
        algum_plotado = True

    if not algum_plotado:
        print("  Nenhum arquivo disponível para este gráfico.")
        plt.close()
        return

    ax.set_title("Influência do número de épocas")
    ax.set_xlabel("Época")
    ax.set_ylabel("MSE médio")
    ax.legend()
    ax.grid(True)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    fig.tight_layout()
    salvar("grafico_4_epocas.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Gráfico 5 — Barras: acurácia de todos os experimentos
# ─────────────────────────────────────────────────────────────────────────────
def grafico_acuracias():
    print("\n[Gráfico 5] Acurácia por experimento (barra)")

    exps_disponiveis = {k: v for k, v in ACURACIAS.items() if v is not None}

    if not exps_disponiveis:
        print("  Nenhuma acurácia preenchida no dicionário ACURACIAS. Pulando.")
        return

    labels = [f"Exp {k}\n{DESCRICOES[k]}" for k in exps_disponiveis]
    valores = list(exps_disponiveis.values())
    cores = ["#1f77b4" if k != 2 else "#ff7f0e" for k in exps_disponiveis]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    bars = ax.bar(labels, valores, color=cores, edgecolor="white", width=0.6)

    # Rótulo com o valor em cima de cada barra
    for bar, val in zip(bars, valores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title("Acurácia por experimento")
    ax.set_ylabel("Acurácia (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.4)

    # Linha de referência no baseline
    if 2 in exps_disponiveis:
        ax.axhline(
            y=ACURACIAS[2],
            color="#ff7f0e",
            linestyle="--",
            linewidth=1,
            label=f"Baseline ({ACURACIAS[2]:.2f}%)",
        )
        ax.legend()

    fig.tight_layout()
    salvar("grafico_5_acuracias.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Execução
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Gerando gráficos — MLP hiperparâmetros")
    print("=" * 55)

    grafico_baseline()
    grafico_learning_rate()
    grafico_hidden_size()
    grafico_epocas()
    grafico_acuracias()

    print(f"\nPronto! Todos os gráficos foram salvos em: ./{OUTPUT_DIR}/")
    print("Arquivos gerados:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  {f}")
