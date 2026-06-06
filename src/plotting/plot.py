# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "matplotlib>=3.10.9",
#     "pandas>=3.0.3",
# ]
# ///

"""
plot.py — Gera todos os gráficos de análise de hiperparâmetros da rede MLP.

Localização : src/plotting/plot.py
Lê CSVs de  : <raiz>/results/erros_por_epoca_exp{N}.csv
Salva em    : <raiz>/graficos/

Dependências: pip install pandas matplotlib --break-system-packages

Execute: python3 plot.py
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


# Caminhos — relativos à raiz do projeto (src/plotting/ → ../../)

SCRIPT_DIR   = Path(__file__).parent        # src/plotting/
ROOT         = SCRIPT_DIR.parent.parent     # raiz do projeto
RESULTS_DIR  = ROOT                         # CSVs ficam na raiz do projeto
GRAFICOS_DIR = ROOT / "graficos"

os.makedirs(GRAFICOS_DIR, exist_ok=True)


# Dados dos experimentos

ACURACIAS = {
    1: 73.07,  # lr=0.0001, hidden=60, epocas=200
    2: 91.02,  # lr=0.001,  hidden=60, epocas=200  ← baseline
    3: 87.17,  # lr=0.01,   hidden=60, epocas=200
    4: 90.38,  # lr=0.1,    hidden=60, epocas=200
    5: 90.38,  # lr=0.001,  hidden=30, epocas=200
    6: 89.74,  # lr=0.001,  hidden=90, epocas=200
    7: 91.02,  # lr=0.001,  hidden=120, epocas=200
    8: 92.30,  # lr=0.001,  hidden=60, epocas=100
    9: 90.38,  # lr=0.001,  hidden=60, epocas=400
}

DESCRICOES = {
    1: "lr=0.0001",
    2: "lr=0.001, hidden=60, epocas=200 (baseline)",
    3: "lr=0.01",
    4: "lr=0.1",
    5: "hidden=30",
    6: "hidden=90",
    7: "hidden=120",
    8: "épocas=100",
    9: "épocas=400",
}

# Dados das variações autorais (Tabela 3)
# Arquivos: erros_por_epoca_exp{ID}_overfitting.csv
# V1 → exp4  | V2 → exp10 | V3 → exp11 | V4 → exp12 | V5 → exp13

OVERFITTING_IDS = [4, 10, 11, 12, 13]

DESCRICOES_OVERFITTING = {
    4:  "V1 — Dados limpos",
    10: "V2 — Ruído 10%",
    11: "V3 — Ruído 20%",
    12: "V4 — Classes mescladas (D+O)",
    13: "V5 — Sem estratificação + classes mescladas (I+J)",
}

ACURACIAS_OVERFITTING = {
    4:  91.0256,
    10: 80.1282,
    11: 73.7179,
    12: 90.3846,
    13: 95.5224,
}

EPOCAS_EFETIVAS_OVERFITTING = {
    4:   72,
    10: 233,
    11: 222,
    12: 332,
    13: 320,
}
# O fallback "erro_medio_mse" garante compatibilidade com CSVs do formato antigo.

COL_TREINO_NOVO  = "erro_treino_mse"
COL_VAL_NOVO     = "erro_val_mse"
COL_TREINO_ANTIGO = "erro_medio_mse"   # formato gerado antes do early stopping


# Configurações visuais globais

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "grid.alpha": 0.35,
})


# Funções auxiliares

def carregar_csv(exp_id):
    """Carrega o CSV de erros de um experimento. Retorna None se não existir."""
    caminho = RESULTS_DIR / f"erros_por_epoca_exp{exp_id}.csv"
    if not caminho.exists():
        print(f"  [aviso] {caminho.name} não encontrado — exp {exp_id} ignorado.")
        return None
    return pd.read_csv(caminho)


def col_treino(df):
    """Retorna o nome da coluna de erro de treino, independente do formato do CSV."""
    if COL_TREINO_NOVO in df.columns:
        return COL_TREINO_NOVO
    return COL_TREINO_ANTIGO   # fallback para CSVs no formato antigo


def tem_validacao(df):
    """Retorna True se o CSV contém a coluna de erro de validação (novo formato)."""
    return COL_VAL_NOVO in df.columns


def salvar(nome_arquivo):
    caminho = GRAFICOS_DIR / nome_arquivo
    plt.savefig(caminho, bbox_inches="tight")
    print(f"  → salvo em: {caminho}")



# Gráfico 1 — Curva de erro do baseline com treino e validação

def grafico_baseline():
    """
    Plota as curvas de erro de treino e validação do experimento baseline (exp 2).
    Se o CSV for do formato antigo (sem coluna de validação), plota apenas o treino.
    """
    print("\n[Gráfico 1] Curva de erro — baseline (exp 2)")
    df = carregar_csv(2)
    if df is None:
        return

    col_t = col_treino(df)
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(df["epoca"], df[col_t],
            color="#1f77b4", linewidth=1.5, label="Treino")

    if tem_validacao(df):
        ax.plot(df["epoca"], df[COL_VAL_NOVO],
                color="#ff7f0e", linewidth=1.5, linestyle="--", label="Validação")
        ax.legend()
        ax.set_title("Curva de erro — treino e validação (baseline)")
    else:
        ax.set_title("Curva de erro durante o treinamento (baseline)")

    ax.set_xlabel("Época")
    ax.set_ylabel("MSE médio")
    ax.grid(True)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    fig.tight_layout()
    salvar("grafico_1_baseline.png")
    plt.close()



# Gráfico 2 — Comparação: taxa de aprendizado (exps 1, 2, 3, 4)

def grafico_learning_rate():
    """
    Compara o erro de treino para diferentes valores de taxa de aprendizado.
    Plota apenas a curva de treino para manter o gráfico legível com 4 linhas.
    """
    print("\n[Gráfico 2] Comparação — taxa de aprendizado (exps 1, 2, 3, 4)")
    exps  = [1, 2, 3, 4]
    cores = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]

    fig, ax = plt.subplots(figsize=(8, 4))
    algum_plotado = False

    for exp_id, cor in zip(exps, cores):
        df = carregar_csv(exp_id)
        if df is None:
            continue
        ax.plot(df["epoca"], df[col_treino(df)],
                label=DESCRICOES[exp_id], color=cor, linewidth=1.5)
        algum_plotado = True

    if not algum_plotado:
        print("  Nenhum arquivo disponível.")
        plt.close()
        return

    ax.set_title("Influência da taxa de aprendizado")
    ax.set_xlabel("Época")
    ax.set_ylabel("MSE médio (treino)")
    ax.legend()
    ax.grid(True)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    fig.tight_layout()
    salvar("grafico_2_learning_rate.png")
    plt.close()



# Gráfico 3 — Comparação: tamanho da camada escondida (exps 5, 2, 6, 7)

def grafico_hidden_size():
    """
    Compara o erro de treino para diferentes quantidades de neurônios
    na camada escondida: 30, 60, 90 e 120.
    """
    print("\n[Gráfico 3] Comparação — neurônios na camada escondida (exps 5, 2, 6, 7)")
    exps  = [5, 2, 6, 7]
    cores = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]

    fig, ax = plt.subplots(figsize=(8, 4))
    algum_plotado = False

    for exp_id, cor in zip(exps, cores):
        df = carregar_csv(exp_id)
        if df is None:
            continue
        ax.plot(df["epoca"], df[col_treino(df)],
                label=DESCRICOES[exp_id], color=cor, linewidth=1.5)
        algum_plotado = True

    if not algum_plotado:
        print("  Nenhum arquivo disponível.")
        plt.close()
        return

    ax.set_title("Influência do número de neurônios na camada escondida")
    ax.set_xlabel("Época")
    ax.set_ylabel("MSE médio (treino)")
    ax.legend()
    ax.grid(True)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    fig.tight_layout()
    salvar("grafico_3_hidden_size.png")
    plt.close()



# Gráfico 4 — Comparação: número de épocas (exps 8, 2, 9)

def grafico_epocas():
    """
    Compara o erro de treino para diferentes limites máximos de épocas:
    100, 200 e 400 (o early stopping pode encerrar antes do limite).
    """
    print("\n[Gráfico 4] Comparação — número de épocas (exps 8, 2, 9)")
    exps  = [8, 2, 9]
    cores = ["#d62728", "#1f77b4", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(8, 4))
    algum_plotado = False

    for exp_id, cor in zip(exps, cores):
        df = carregar_csv(exp_id)
        if df is None:
            continue
        ax.plot(df["epoca"], df[col_treino(df)],
                label=DESCRICOES[exp_id], color=cor, linewidth=1.5)
        algum_plotado = True

    if not algum_plotado:
        print("  Nenhum arquivo disponível.")
        plt.close()
        return

    ax.set_title("Influência do número máximo de épocas")
    ax.set_xlabel("Época")
    ax.set_ylabel("MSE médio (treino)")
    ax.legend()
    ax.grid(True)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    fig.tight_layout()
    salvar("grafico_4_epocas.png")
    plt.close()



# Gráfico 5 — Barras: acurácia de todos os experimentos

def grafico_acuracias():
    """
    Gráfico de barras com a acurácia (%) de cada experimento no conjunto de teste.
    O baseline (exp 2) é destacado em laranja e uma linha horizontal marca seu valor.
    """
    print("\n[Gráfico 5] Acurácia por experimento (barra)")

    exps_disponiveis = {k: v for k, v in ACURACIAS.items() if v is not None}
    if not exps_disponiveis:
        print("  Nenhuma acurácia preenchida. Pulando.")
        return

    labels = [f"Exp {k}\n{DESCRICOES[k]}" for k in exps_disponiveis]
    valores = list(exps_disponiveis.values())
    cores   = ["#ff7f0e" if k == 2 else "#1f77b4" for k in exps_disponiveis]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    bars = ax.bar(labels, valores, color=cores, edgecolor="white", width=0.6)

    for bar, val in zip(bars, valores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.2f}%",
                ha="center", va="bottom", fontsize=9)

    if 2 in exps_disponiveis:
        ax.axhline(y=ACURACIAS[2], color="#ff7f0e", linestyle="--",
                   linewidth=1, label=f"Baseline ({ACURACIAS[2]:.2f}%)")
        ax.legend()

    ax.set_title("Acurácia por experimento (conjunto de teste)")
    ax.set_ylabel("Acurácia (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    salvar("grafico_5_acuracias.png")
    plt.close()

def carregar_csv_overfitting(exp_id):
    """Carrega o CSV de erros de uma variação autoral (_overfitting). Retorna None se não existir."""
    caminho = RESULTS_DIR / f"erros_por_epoca_exp{exp_id}_overfitting.csv"
    if not caminho.exists():
        print(f"  [aviso] {caminho.name} não encontrado — variação exp {exp_id} ignorada.")
        return None
    return pd.read_csv(caminho)


# Gráfico 6 — Variações autorais: treino vs validação (V1–V5)
# Lê apenas arquivos nomeados erros_por_epoca_exp{N}_overfitting.csv
# IDs esperados: 4 (V1), 10 (V2), 11 (V3), 12 (V4), 13 (V5)

def grafico_early_stopping():
    """
    Plota as curvas de treino e validação das variações autorais (V1–V5).
    Cada subplot mostra uma variação do conjunto de teste, identificada pelo
    nome do arquivo erros_por_epoca_exp{N}_overfitting.csv.
    Ilustra como o early stopping se comporta em cada cenário de variação.
    """
    print("\n[Gráfico 6] Variações autorais — treino vs validação (V1–V5)")

    dados = {}
    for exp_id in OVERFITTING_IDS:
        df = carregar_csv_overfitting(exp_id)
        if df is not None:
            dados[exp_id] = df

    if not dados:
        print("  Nenhum CSV _overfitting encontrado. "
              "Renomeie os arquivos conforme: erros_por_epoca_expX_overfitting.csv")
        return

    n    = len(dados)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3.5),
                             squeeze=False)

    for ax_idx, (exp_id, df) in enumerate(dados.items()):
        r, c = divmod(ax_idx, cols)
        ax   = axes[r][c]

        col_t = col_treino(df)
        ax.plot(df["epoca"], df[col_t],
                color="#1f77b4", linewidth=1.3, label="Treino")

        if tem_validacao(df):
            ax.plot(df["epoca"], df[COL_VAL_NOVO],
                    color="#ff7f0e", linewidth=1.3, linestyle="--", label="Validação")

        # Marca a época em que o early stopping disparou (última época registrada)
        epocas_efetivas = EPOCAS_EFETIVAS_OVERFITTING.get(exp_id)
        if epocas_efetivas and epocas_efetivas == len(df):
            ax.axvline(x=epocas_efetivas, color="#d62728", linewidth=1,
                       linestyle=":", label=f"Parada: época {epocas_efetivas}")

        acuracia = ACURACIAS_OVERFITTING.get(exp_id)
        titulo   = DESCRICOES_OVERFITTING.get(exp_id, f"Exp {exp_id}")
        if acuracia is not None:
            titulo += f"\nAcurácia: {acuracia:.4f}%"
        ax.set_title(titulo, fontsize=10)
        ax.set_xlabel("Época", fontsize=9)
        ax.set_ylabel("MSE", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

    for ax_idx in range(len(dados), rows * cols):
        r, c = divmod(ax_idx, cols)
        axes[r][c].set_visible(False)

    fig.suptitle("Variações Autorais — Treino vs Validação (Early Stopping)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    salvar("grafico_6_variacoes_autorais.png")
    plt.close()

def grafico_convergencia_todos():
    """
    Grade 3×3 com um subplot por experimento (exps 1–9).
    Cada subplot exibe as curvas de treino (azul sólido) e validação
    (laranja tracejado), mostrando que ambas convergem juntas nos
    experimentos normais.
    """
    print("\n[Gráfico 7] Convergência treino/validação — todos os experimentos (1–9)")

    todos_ids = list(ACURACIAS.keys())  # [1, 2, ..., 9]
    cols = 3
    rows = (len(todos_ids) + cols - 1) // cols  # 3 linhas

    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 5, rows * 3.8),
                             squeeze=False)

    for ax_idx, exp_id in enumerate(todos_ids):
        r, c = divmod(ax_idx, cols)
        ax   = axes[r][c]

        df = carregar_csv(exp_id)
        if df is None:
            ax.set_visible(False)
            continue

        col_t = col_treino(df)
        ax.plot(df["epoca"], df[col_t],
                color="#1f77b4", linewidth=1.3, label="Treino")

        if tem_validacao(df):
            ax.plot(df["epoca"], df[COL_VAL_NOVO],
                    color="#ff7f0e", linewidth=1.3, linestyle="--", label="Validação")

        acuracia = ACURACIAS.get(exp_id)
        titulo   = f"Exp {exp_id} — {DESCRICOES.get(exp_id, '')}"
        if acuracia is not None:
            titulo += f"\nAcurácia: {acuracia:.2f}%"
        ax.set_title(titulo, fontsize=9)
        ax.set_xlabel("Época", fontsize=8)
        ax.set_ylabel("MSE", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

    # Oculta subplots vazios na última linha (se houver)
    for ax_idx in range(len(todos_ids), rows * cols):
        r, c = divmod(ax_idx, cols)
        axes[r][c].set_visible(False)

    fig.suptitle("Convergência Treino e Validação — Experimentos 1 a 9",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    salvar("grafico_7_convergencia_todos.png")
    plt.close()

# Execução
if __name__ == "__main__":
    print("=" * 55)
    print("  Gerando gráficos — MLP hiperparâmetros")
    print(f"  Lendo de : {RESULTS_DIR}  (raiz do projeto)")
    print(f"  Salvando : {GRAFICOS_DIR}")
    print("=" * 55)

    grafico_baseline()
    grafico_learning_rate()
    grafico_hidden_size()
    grafico_epocas()
    grafico_acuracias()
    grafico_early_stopping()
    grafico_convergencia_todos()

    arquivos = sorted(os.listdir(GRAFICOS_DIR))
    print(f"\nPronto! {len(arquivos)} gráficos salvos em: {GRAFICOS_DIR}")
    for f in arquivos:
        print(f"  {f}")