# Reproduzindo EMPT: Pipeline para Processamento de Sinais de EEG

Este repositório contém a implementação do pipeline **EMPT** (EEG Mixture-of-Experts ProbSparse Transformer), desenvolvido como parte do **Trabalho de Conclusão de Curso em Engenharia da Computação na Universidade Federal de Pelotas (UFPel)**.

O projeto foca na classificação de sinais de eletroencefalograma (EEG) para **Imaginação Motora (MI)**, utilizando uma arquitetura baseada em Transformers com mecanismos de eficiência e especialização de redes.



## Resumo do Projeto

A pesquisa reproduz e organiza um pipeline de processamento ponta a ponta, desde a aquisição de dados brutos até a classificação final. O diferencial do modelo EMPT reside na combinação de:
- **ProbSparse Attention**: Uma variante da atenção clássica que reduz a complexidade computacional ao focar nos pares query-key com maior "esparsidade" de informação.
- **Mixture-of-Experts (MoE)**: Uma camada que ativa apenas um subconjunto de "especialistas" (redes neurais densas) para cada amostra, permitindo maior capacidade de aprendizado sem um custo computacional proibitivo.


## Requisitos de Sistema

O pipeline foi desenvolvido e testado utilizando **Python 3.8+**. Para garantir a reprodutibilidade dos resultados, as versões principais das bibliotecas utilizadas são:

* **TensorFlow**: `2.15.1`
* **Braindecode**: `0.8.1`
* **MNE**: `1.8.0`
* **MOABB**: `1.2.0`
* **Scikit-learn**: `1.5.2`
* **Numpy**: `1.26.4`
* **Matplotlib**: `3.9.4`

Mais detalhes em `requirements.txt`.

## Estrutura de Arquivos

O código foi modularizado para seguir boas práticas de engenharia de software:

* `architecture.py`: Contém todas as camadas customizadas (`TopKMoE`, `ProbSparseAttention`) e a estrutura do modelo `EMPTModel`.
* `preprocessing.py`: Funções para carregamento automático de datasets via MOABB e pré-processamento via Braindecode (filtros, normalização exponencial e janelamento).
* `main.py`: Ponto de entrada do experimento. Coordena o K-Fold, o treinamento e a geração de gráficos.

## Tecnologias Utilizadas

* **TensorFlow/Keras**: Framework principal para Deep Learning.
* **MNE-Python**: Processamento de dados neurofisiológicos.
* **Braindecode**: Biblioteca especializada em Deep Learning para EEG.
* **MOABB**: Mother of All BCI Benchmarks, utilizado para garantir a reprodutibilidade dos datasets.

## Como Utilizar

### 1. Instalação das Dependências

Bash:
* `pip install -r requirements.txt`

ou:
* `pip install tensorflow==2.15.1 braindecode==0.8.1 moabb==1.2.0 mne==1.8.0 scikit-learn==1.5.2 matplotlib==3.9.4`


### 2. Execução

Para iniciar o treinamento e a validação cruzada (K-Fold):

Bash:
* `python main.py`

## Resultados e Logs
O pipeline gera automaticamente:

* `resultados_moe.txt`: Um log detalhado contendo a acurácia de cada fold por sujeito.

* `Gráficos de Treino`: Ficheiros .png com as curvas de Loss e Accuracy para cada etapa da validação.

## Autoria

* `Autor`: Santiago Del Valle Alvarez Martinez

* `Orientador`: Prof. Dr. Marilton Sanchotene de Aguiar

* `Instituição`: Centro de Desenvolvimento Tecnológico (CDeTec) - UFPel

* `Ano`: 2026

## Este trabalho é uma contribuição acadêmica para a área de Interfaces Cérebro-Computador (BCI).