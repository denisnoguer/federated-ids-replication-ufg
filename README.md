Summary in English
This project replicates a federated learning pipeline for intrusion detection (IDS), comparing the performance of traditional centralized training with a manual federated approach, using the classic KDD Cup 99 dataset.
The goal is to practically demonstrate the challenges, limitations, and opportunities of federated learning (FL) in real-world security scenarios, providing a foundation for future research and improvements.

** ALL FILES USED IN THE PROJECT CAN BE DOWNLOADED FROM THE ADDRESS BELOW ON GOOGLE DRIVE. THEY CAN ALSO BE REUSED AND PUBLISHED, BUT FIRST SEND AN EMAIL TO denisnogueira@discente.ufg.br

https://drive.google.com/drive/folders/1EjGZGoUtxG6pSGt9K_gUqYqvQkGtO9PW



The pipeline features:

Detailed data preprocessing (category conversion, normalization, handling missing data).

Centralized training (full data access) and manual federated learning (data split among simulated clients, FedAvg aggregation).

Evaluation with multiple metrics: accuracy, precision, recall, F1-score, confusion matrix, and efficiency comparison (training time, memory usage, communication).

Critical discussion of the results, highlighting limitations and proposing improvements such as increasing epochs, federated rounds, data balancing, and advanced strategies (FedProx, normalization, etc.).

All code is fully commented, ready for replication and adaptation, serving as a reference for teachers, students, and researchers interested in security, distributed AI, and applied data science.




Português Brasil
Este projeto realiza a replicação do pipeline de aprendizado federado aplicado à detecção de intrusos (IDS), comparando o desempenho entre o treinamento centralizado tradicional e a abordagem federada manual, utilizando o dataset clássico KDD Cup 99.
O objetivo é demonstrar na prática os desafios, limitações e oportunidades do aprendizado federado (FL) em cenários reais de segurança, além de servir como base para futuras pesquisas e aprimoramentos.

** TODOS OS ARQUIVOS USADOS NESTE PROJETO ESTAO DISPONIVES NO GOOGLE DRIVE NO LINK ABAIXO, PODEM SER USADOS LIVREMENTE PARA APRENDIZADO, mas antes mande um e-mail para email: denisnogueira@discente.ufg.br

LINK GOOGLE DRIVE ARQUIVOS FL
https://drive.google.com/drive/folders/1EjGZGoUtxG6pSGt9K_gUqYqvQkGtO9PW

O pipeline inclui:

Pré-processamento detalhado dos dados (conversão de categorias, padronização, tratamento de dados ausentes).

Treinamento centralizado (acesso total aos dados) e federado manual (divisão entre clientes simulados, agregação FedAvg).

Avaliação com múltiplas métricas: acurácia, precisão, recall, F1-score, matriz de confusão, além de comparação de eficiência (tempo, memória, comunicação).

Discussão crítica sobre os resultados, justificando limitações e sugerindo melhorias, como aumento de épocas, rodadas federadas, balanceamento de dados e uso de estratégias avançadas (FedProx, normalização, etc.).

O código está totalmente comentado, pronto para replicação e adaptações, servindo como referência para professores, alunos e pesquisadores interessados em segurança, IA distribuída e ciência de dados aplicada.


# Replication of Federated Learning for Intrusion Detection

This repository contains the code and materials for replicating the federated learning paper applied to intrusion detection, using the KDD Cup 99 dataset. The goal is to allow other researchers and students to reproduce the experiments, analyze the results and propose improvements.

## How to run
0. LINUX UBUNTU (UPDATED)
1. Clone the repository from GOOGLE DRIVE
https://drive.google.com/drive/folders/1EjGZGoUtxG6pSGt9K_gUqYqvQkGtO9PW
2. Install the dependencies (see requirements.txt)
3. Run the main script (MODIFY THE CODE:
4. The results and graphs will be saved in the same folder.

## Requirements
- Python 3.9+
- TensorFlow
- Pandas
- Scikit-learn
- Matplotlib
- (and others listed in requirements.txt)

## Citation
If you use this repository, please cite as follows:
> Denis. Federated Learning Replication for IDS, 2024.


========================
PORTUGUES BRASIL

# Replicação de Aprendizado Federado para Detecção de Intrusos

Este repositório contém o código e materiais para replicação do artigo de aprendizado federado aplicado à detecção de intrusos, usando o dataset KDD Cup 99. O objetivo é permitir que outros pesquisadores e estudantes possam reproduzir os experimentos, analisar os resultados e propor melhorias.

## Como rodar
0. LINUX UBUNTU (ATUALIZADO)
1. Clone o repositório do GOOGLE DRIVE
https://drive.google.com/drive/folders/1EjGZGoUtxG6pSGt9K_gUqYqvQkGtO9PW
2. Instale as dependências (veja requirements.txt)
3. Execute o script principal (MODIFIQUE O CÓDIGO:
4. Os resultados e gráficos serão salvos na mesma pasta.

## Requisitos
- Python 3.9+
- TensorFlow
- Pandas
- Scikit-learn
- Matplotlib
- (e outras listadas em requirements.txt)

## Citação
Se usar este repositório, cite conforme:
> Denis. Replicação de aprendizado federado para IDS, 2024.



