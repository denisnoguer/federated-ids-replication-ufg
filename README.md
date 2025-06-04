Summary in English
This project replicates a federated learning pipeline for intrusion detection (IDS), comparing the performance of traditional centralized training with a manual federated approach, using the classic KDD Cup 99 dataset.
The goal is to practically demonstrate the challenges, limitations, and opportunities of federated learning (FL) in real-world security scenarios, providing a foundation for future research and improvements.

The pipeline features:

Detailed data preprocessing (category conversion, normalization, handling missing data).

Centralized training (full data access) and manual federated learning (data split among simulated clients, FedAvg aggregation).

Evaluation with multiple metrics: accuracy, precision, recall, F1-score, confusion matrix, and efficiency comparison (training time, memory usage, communication).

Critical discussion of the results, highlighting limitations and proposing improvements such as increasing epochs, federated rounds, data balancing, and advanced strategies (FedProx, normalization, etc.).

All code is fully commented, ready for replication and adaptation, serving as a reference for teachers, students, and researchers interested in security, distributed AI, and applied data science.




Português Brasil
Este projeto realiza a replicação do pipeline de aprendizado federado aplicado à detecção de intrusos (IDS), comparando o desempenho entre o treinamento centralizado tradicional e a abordagem federada manual, utilizando o dataset clássico KDD Cup 99.
O objetivo é demonstrar na prática os desafios, limitações e oportunidades do aprendizado federado (FL) em cenários reais de segurança, além de servir como base para futuras pesquisas e aprimoramentos.

O pipeline inclui:

Pré-processamento detalhado dos dados (conversão de categorias, padronização, tratamento de dados ausentes).

Treinamento centralizado (acesso total aos dados) e federado manual (divisão entre clientes simulados, agregação FedAvg).

Avaliação com múltiplas métricas: acurácia, precisão, recall, F1-score, matriz de confusão, além de comparação de eficiência (tempo, memória, comunicação).

Discussão crítica sobre os resultados, justificando limitações e sugerindo melhorias, como aumento de épocas, rodadas federadas, balanceamento de dados e uso de estratégias avançadas (FedProx, normalização, etc.).

O código está totalmente comentado, pronto para replicação e adaptações, servindo como referência para professores, alunos e pesquisadores interessados em segurança, IA distribuída e ciência de dados aplicada.

