![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white&style=flat)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white&style=flat)
![Flower](https://img.shields.io/badge/Flower-Federated%20Learning-orange?style=flat)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white&style=flat)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white&style=flat)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikit-learn&logoColor=white&style=flat)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?logo=matplotlib&logoColor=white&style=flat)

![Status](https://img.shields.io/badge/Status-Em%20Desenvolvimento-yellow)
![Projeto Acadêmico](https://img.shields.io/badge/UFG-Mestrado%20em%20Ciência%20da%20Computação-blueviolet?style=flat&logo=academia)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Discussões](https://img.shields.io/badge/Discussões-GitHub-blue?logo=github)
![Segurança](https://img.shields.io/badge/Segurança-Foco%20em%20Ataques%20DDoS-red)
![NSL-KDD](https://img.shields.io/badge/Dataset-NSL--KDD-blue?style=flat)
![image](https://github.com/user-attachments/assets/ed439a6e-a652-409c-8753-f5a98c52af7a)


## ABOUT THIS DOCUMENTATION ### 
FL Federated Learning Applied to Intrusion Detection (IDS) on the KDD Cup 99 dataset, 
Including Python code and practical results, including time of learning and memory, Docker Ambientation.   
Research developed at **[UFG - Universidade Federal de Goiás](https://www.ufg.br/)**  
UFG - Universidade Federal de Goias.  
DENIS NOGUEIRA DO NASCIMENTO   
email:  <denisnoguer@gmail.com>;  



---

## 🧰 Tools Overview

The following are the tools and technologies used throughout this project. They represent essential components for most machine learning and data science workflows.

- 🐳 **[Docker](https://hub.docker.com/)**  
  Containerization platform used to build and run reproducible environments. This project runs in a Dockerized setup.

- 🐍 **[Python](https://github.com/python/cpython)**  
  High-level programming language ideal for data science, AI, and automation tasks.

- 📦 **[NumPy](https://github.com/numpy/numpy)**  
  A core library for scientific computing in Python, mainly used for array and matrix operations.

- 🐼 **[Pandas](https://github.com/pandas-dev/pandas)**  
  A powerful tool for data manipulation and analysis, especially when working with structured datasets.

- 🧠 **[TensorFlow](https://github.com/tensorflow/tensorflow)**  
  An open-source deep learning framework used for building neural networks in applications such as Computer Vision and NLP.

- 📊 **[Scikit-learn](https://github.com/scikit-learn/scikit-learn)**  
  A simple and efficient tool for data mining and machine learning.

- 🐧 **[Linux – Ubuntu](https://ubuntu.com/)**  
  The base operating system used to run and test the environment — recommended: latest LTS version.

## Focus Objective ### 

This project replicates a federated learning pipeline for Intrusion Detection Systems (IDS), comparing the performance of traditional centralized training with a manual federated approach using the classic KDD Cup 99 dataset.  
The goal is to demonstrate, in practice, the challenges, limitations, and opportunities of Federated Learning (FL) in real-world security scenarios, while serving as a foundation for future research and improvements.


## Organization ### 
This repository is organized into three main sections:   
1- Program and requirements  
2- Download, instalation and initialization of dependences   
3- Comparations and graphics about the results  


## Repository and files ###   


Google Drive  
  https://drive.google.com/drive/folders/1EjGZGoUtxG6pSGt9K_gUqYqvQkGtO9PW


## THE Requeriments  

1- Program and requirements 

- Python 3.9+
- FLOWER
- TensorFlow
- Pandas
- Scikit-learn
- Matplotlib
- (all complete list is on requirements.txt)

## Essentinal Library :  

REQUIRED COMMANDS IN LINUX

  -- pip install tensorflow flwr pandas numpy scikit-learn matplotlib

## How to run
0. LINUX UBUNTU (UPDATED)
1. Clone the repository from GOOGLE DRIVE
https://drive.google.com/drive/folders/1EjGZGoUtxG6pSGt9K_gUqYqvQkGtO9PW
2. Install the dependencies (see requirements.txt)
3. Run the main script (MODIFY THE CODE:
4. The results and graphs will be saved in the same folder.



## Total Dependeces.


●	TensorFlow (treino do MLP)

●	Flower (flwr) (ORQUESTRATIONS)

●	pandas, numpy, scikit-learn (DATA READ)

●	matplotlib (GRAPHICS)

## DOCKER CONFIGURATIONS

THE DIRETORIES USED IS> 

/0/IDS


ACTIVE VIRTUAL MODE IN FL
source venv/bin/activate

python fl_manual_avancado.py
(NAME OF FILE IS  "fl_manual_avancado.py")

## 5. 🛡️ Attack Categories in the Dataset

The KDD Cup 99 dataset includes various types of attacks grouped into four main categories, each with specific behaviors and threats:

- 🔥 **DoS** – *Denial of Service*  
  The attacker tries to make a machine or network resource unavailable.  
  **Examples**: `smurf`, `neptune`, `teardrop`, `pod`, `land`, `back`

- 🟣 **U2R** – *User to Root*  
  The attacker starts as a normal user and attempts to gain root-level access.  
  **Examples**: `buffer_overflow`, `loadmodule`, `perl`, `rootkit`

- 🔵 **R2L** – *Remote to Local*  
  An external attacker tries to gain local user access.  
  **Examples**: `ftp_write`, `guess_passwd`, `imap`, `multihop`, `phf`, `spy`, `warezclient`, `warezmaster`

- 🟢 **Probe** – *Surveillance / Scanning Activities*  
  The attacker probes and scans networks to gather information.  
  **Examples**: `ipsweep`, `nmap`, `portsweep`, `satan`


## BILD THE REPLICAL CONTAINERS  
## 🖥️ Local Execution Path

This step must be executed **locally** on your machine, within the selected project directory.

In this project, the working directory used was:  
`/0/IDS`

---
9.1 -  Graphic Generation CODE

![image](https://github.com/user-attachments/assets/39c5d161-5f21-4976-9b41-a440dde268df)


 



## 📊 Step 9.2 – Accuracy Results from the Experiment

| Scenario                                | Accuracy (%)     |
|----------------------------------------|------------------|
| Centralized                             | 99.16            |
| Federated (FedAvg, best configuration) | 99.40 (+0.24 pp) |
| Federated (FedAvg, worst configuration)| 96.62 (−2.54 pp) |

---

## 🧩 9.1 – Code for Graphic Generation




_______________________________________
Adição opcional do Gráfico comparativo

 Se quiser salvar como imagem PNG para anexar ao relatório, adicione estas linhas no final do script, antes de plt.show():
python
CopiarEditar
plt.savefig("comparacao_centralizado_federado.png", dpi=120, bbox_inches='tight')
plt.show()

O arquivo de comparativo que possui as fotos e dados demonstrativo neste tutorial pode ser acessadas no endereço do google drive abaixo:

https://drive.google.com/file/d/175wU3g_1heesrj4pg6VRAvoO-BOopjAM/view?usp=drive_link

=-=-=-=--=-==-=-=-=-==-=--==-=-=
TUTORIAL IN ENGLISH

# Replication of Federated Learning for Intrusion Detection

This repository contains the code and materials for replicating the federated learning paper applied to intrusion detection, using the KDD Cup 99 dataset. The goal is to allow other researchers and students to reproduce the experiments, analyze the results and propose improvements.



## Requirements
- Python 3.9+
- TensorFlow
- Pandas
- Scikit-learn
- Matplotlib
- (and others listed in requirements.txt)
Título:
FEDERATED LEARNING-BASED APPROACH FOR INTRUSION DETECTION IN COMPUTER NETWORKS

ABORDAGEM BASEADA EM APRENDIZADO FEDERADO PARA A DETECÇÃO DE INTRUSÃO EM REDES DE COMPUTADORES.

LIST OF TECHNICAL METHODOLOGY APPLIED FOR FL.
1.	Environment preparation:
•	Install Docker / Docker Compose
•	Create a Python environment and install libraries (Flower, TensorFlow/PyTorch, scikit-learn)
2.	Data preprocessing:
•	Download and partition KDD-99, MotionSense, etc.
•	Encoding and normalization
•	Generate partitions for FL
3.	Centralized model:
•	Implement and train a local MLP
•	Validate reference accuracy
4.	Basic FL orchestration:
•	Create server.py and client.py using Flower
•	Adjust docker-compose.yml for 20 clients
5.	FL Experiments (IDS):
•	Run FL varying N, E, and B
•	Collect logs and accuracy metrics
6.	DDoS extension:
•	Unzip DDos.zip
•	Configure environment and DDoS data
•	Execute and collect results

7.	ERI Project:
•	Test modifications and generate new results
Analysis and visualization:
•	Consolidate logs
•	Generate comparative graphs (centralized vs FL, IDS vs DDoS vs ERI)
Detailed data analysis and report (methodology, results, discrepancies)
•	Suggestions for future improvements


1- INSTALLATION TUTORIAL FOR DEPENDENCIES
(Note: Latest versions of Linux Ubuntu were used.)
Installing dependencies
Python (VIRTUAL)
Step 2 – Create and activate the Python virtual environment
Inside /0/IDS, run:


Step B – Install dependencies manually
2- List of Required Dependencies
•	TensorFlow (MLP training)
•	Flower (flwr) (federated orchestration)
•	pandas, numpy, scikit-learn (preprocessing)
•	matplotlib (graphs)
bash
# Ainda com o (venv) ativo em /0/IDS:
pip install --upgrade pip

# Instalar as bibliotecas essenciais:
pip install tensorflow flwr pandas numpy scikit-learn matplotlib


2- List of Required Dependencies
•	TensorFlow (MLP training)
•	Flower (flwr) (federated orchestration)
•	pandas, numpy, scikit-learn (preprocessing)
•	matplotlib (graphs)
●	matplotlib (gráficos)


3- List of Available Flags (such as --epochs, --batch-size, etc.)
•	Centralized training (LOW)
•	Configure Federated Learning
•	Collect results
 
4- DOCKER CONFIGURATION STEP
•	Server service: Ensure only 25 rounds are used.
•	Client service: Configure 3 replicas, 2 local epochs, and 1 round.
Important Docker commands to activate the environment:
Navigate to /0/IDS
Activate the virtual environment:


python fl_manual_avancado.py
(NOME DO ARQUIVO ESCOLHIDO)


5. List of Attack Types Used
•	The names (e.g., back, buffer_overflow, ipsweep) are attack types or network activities detected/classified in the dataset.
•	Next to them are acronyms such as dos, u2r, r2l, probe, which are attack categories.
Attack types:
•	dos: Denial of Service. E.g., smurf, neptune, teardrop, pod, land, back
•	u2r: User to Root. E.g., buffer_overflow, loadmodule, perl, rootkit
•	r2l: Remote to Local. E.g., ftp_write, guess_passwd, imap, multihop, phf, spy, warezclient, warezmaster
•	probe: Scanning activities. E.g., ipsweep, nmap, portsweep, satan
Probable origin of the dataset:
These are classic from the famous KDD Cup 99 Dataset (and variants such as NSL-KDD).
How this pipeline works:
The preproc.py script preprocesses data so it can be used in ML models that classify connections as attack or normal, and if attack, the type.
Traditional & Visionary View:
Traditional: Use of these datasets is classic in cybersecurity — professionals worldwide have used them for decades as IDS references.
GRAPH MODEL
 
6. Build and Run Containers with 3 Client Replicas

8. DDoS Application
Test accuracy: 0.958790123462677
(model output, e.g., predicted probabilities)
 
Test accuracy: 0.958790123462677
0
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step
[[0.9807248 0.0192752]]
1
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step
[[0.00809862 0.99190134]]

-=--=-=-=-=-=-=-=-=-=-==--==--=-=-==-=-=-=-=-=-=-=-

9. DATA ANALISYS
CENARY	Acurácia (%)
CENTRALIZED	99,16
Federado (FedAvg, melhor configuração)	99,40 (+0,24 pp)
Federado (FedAvg, pior configuração)	96,62 (−2,54 pp)



10- Replicating Federated Learning (Methodology Summary)
Preprocessing:
KDD Cup dataset loaded, categorical variables mapped to numerical values. Only numerical columns used; missing values handled to avoid type errors.
Data Partitioning:
Training set divided equally into 3 subsets, simulating 3 clients, as per standard FL paradigm.
Local Training:
Each “client” trained an identical neural network for 2 epochs, fully isolated, no data sharing.
Federated Aggregation:
Final weights of the 3 models aggregated via simple mean (FedAvg), forming a global federated model, as per the literature.
Evaluation:

Final Accuracy Analysis
The initial federated accuracy was lower than the centralized result due to limited epochs per client (2) and no multiple federated rounds (successive aggregations). This is expected and serves as a starting point for further experimentation and improvements.
________________________________________
4. CONCLUSION
The federated pipeline was successfully replicated, highlighting the performance gap between centralized and federated training.
Results demonstrate the importance of parameters such as number of epochs, rounds, and aggregation techniques, serving as a basis for future research.
The code remains open for adjustments and extensions, demonstrating mastery of Federated Learning fundamentals in practice.
Critical Discussion
•	The number of epochs per client greatly impacts the federated model's generalization capability.
•	Data balancing and the number of federated rounds are critical for FL success.
•	More advanced aggregation strategies (such as FedProx) and preprocessing can mitigate some difficulties.
Example Script Used in the Project
(Full Python code — leave as is, or convert comments to English.)
=-=-=-=-=-=-=-=-


## 🧠 Script: Manual Federated Learning (Advanced)

<details>
<summary><strong>Click to expand the code (Python Script)</strong></summary>

```python
# Denis - Aprendizado Federado Manual SEM ERROS de tipo/shape/dict
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time, os, psutil
from sklearn.metrics import classification_report, confusion_matrix

# =============== PRE-PROCESSAMENTO ===============
df = pd.read_csv("dataset/__processed_kdd.csv", low_memory=False)

# ... (demais linhas seguem aqui normalmente)
# Últimas linhas do script aqui


==-=-=-=-=====-=-=-==-==-=-=-=-=-=-=-=-=-=-=-=-=-===-=-==-=-=-=-=====-=-=-==-=
Reference:
●	DAMACENO, Alexsander; C. RIBEIRO, Maria do Rosário; OLIVEIRA-JR, Antonio; DE OLIVEIRA, Renan R.. Abordagem Baseada em Aprendizado Federado para a Detecção de Intrusão em Redes de Computadores. In: ESCOLA REGIONAL DE INFORMÁTICA DE GOIÁS (ERI-GO), 11. , 2023, Goiânia/GO. 



Goiânia, 2 JULY 2025
Autor – DENIS NOGUEIRA DO NASCIMENTO – denisnoguer@gmail.com



## Federated ids Replication UFG ##
## "@Professor {Antonio Oliveira Jr,
    title = "Complete Federated ids Replication",
    author = "Nogueira, Denis Nascimento",
    journal = "GitHub",
    year = "2025",
    url = "https://github.com/denisnoguer/federated-ids-replication-ufg",
}
"##
