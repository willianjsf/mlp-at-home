<!-- This file conforms to the Standard Readme Style -->

# MultiLayer Perceptron At Home

<!-- INSERT BANNER HERE -->

<!-- INSERT BADGES HERE -->

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

Implementação de uma Rede Neural **MultiLayer Perceptron** utilizando o algoritmo de Backpropagation com gradiente descendente, na linguagem C++. Esse projeto é um exemplo de uso do modelo MLP para a classificação de imagens para a disciplina de _Inteligência Artificial_. Os dados de teste podem ser encontrados neste repositório bem como o código fonte.

### Pré-requisitos

- Zig +0.14
- C++ 17+
- Python 3.14+

## Instalação

1. Baixe o compilador do Zig:

- **Windows**: `winget install -e --id zig.zig`
- **MacOS**: `brew install zig`
- **Linux**: `sudo apt-get install -y zig`
- **Mise-en-place**: `mise install -g zig:0.14`

Veja outras formas de instalação aqui: https://ziglang.org/learn/getting-started/#managers.

2. Clone este repositório e entre no respositório.

```sh
git clone https://github.com/caio-bernardo/mlp-at-home.git
cd mlp-at-home
```

## Uso

Para compilar e executar este projeto é necessário ter o compilador Zig (veja formas de instalação nos pré-requisitos). Na raíz deste projeto encontra-se um arquivo `build.zig` que permite compilar e executar este projeto, os comandos disponíveis são:

1. `zig build`. Para compilar o projeto em `zig-out/bin`; e
2. `zig build run`. Para executar o projeto.

## Visualização de Dados

Você pode visualizar os dados produzidos pela rede na pasta `graficos`, essas imagens foram geradas com o uso dos scripts na pasta `src/plotting`. É preciso instalar as dependências (`matplotlib`, `numpy` e `pandas`) antes de rodar os scripts. Caso você tenha o gerenciador [uv](https://docs.astral.sh/uv) instalado. Você pode rodar os scripts apenas com `uv run <local/nome.py>`.

## Organização do Projeto

```sh
src
├── io
│   ├── dataIO.cpp
│   └── dataIO.hpp
├── mlp
│   ├── mlp.cpp
│   └── mlp.hpp
├── plotting
│   ├── plot_matriz_f1.py
│   └── plot.py
└── main.cpp
```

Pelo código acima, o projeto está organizado da seguinte forma:

- **io**: módulo contendo funções de manipulação de arquivo (carregar dataset e exportar parâmetros).
- **mlp**: módulo da rede neural, contém funções de treinamento e predição
- **plotting**: scripts Python para gerar gráficos e visualizações.
- **main.cpp**: ponto de entrada do programa, carrega dataset, realiza treinamento e testes de predição, exportando os resultados encontrados.

<!-- ### Contributors -->

## License

Este projeto está sob a licença MIT. Saiba mais em [LICENSE](LICENSE).

This file was made with [Make Your Reads](https://github.com/caio-bernardo/make-your-reads).

```

```
