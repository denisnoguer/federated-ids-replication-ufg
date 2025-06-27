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


## 1- THE Requeriments and Instalation

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


## ⚙️ Total Dependencies

- 🧠 **TensorFlow** – Used to train the MLP model  
- 🤝 **Flower (flwr)** – Orchestration of federated learning rounds  
- 📊 **Pandas, NumPy, Scikit-learn** – Data loading, processing and handling  
- 📈 **Matplotlib** – Graph and result visualization

---

## 🐳 Docker Configuration

- 📁 **Project directory used**:  
  `/0/IDS`

---

## 🧪 Activate Virtual Environment for FL Execution

- ⚡ Activate the virtual environment:
  ```bash
  source venv/bin/activate



## 🛡️ Attack TYPES and Categories in the Dataset

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


##📊 Step 3- Comparations and graphics about the results


## GRAPHIC CODE 1 EXAMPLE
![image](https://github.com/user-attachments/assets/ad317280-899d-48eb-a608-c28905bb9ed1)



## 📊 The Results Step  – Accuracy Results from the Experiment

| Scenario                                | Accuracy (%)     |
|----------------------------------------|------------------|
| Centralized                             | 99.16            |
| Federated (FedAvg, best configuration) | 99.40 (+0.24 pp) |
| Federated (FedAvg, worst configuration)| 96.62 (−2.54 pp) |

---


---
## 📚 Thanks and Citing This Hard Work!

If you use this repository or base your work on this implementation, please consider citing it as:

```bibtex
@article{Nascimento2025_FL_IDS,
  title     = "Replication of Federated Learning Applied to Intrusion Detection",
  professor = "Antonio Oliveira Jr.",
  author    = "Denis Nogueira do Nascimento",
  journal   = "GitHub",
  date      = "July 2025",
  url       = "https://github.com/denisnoguer/federated-ids-replication-ufg"
}
````
---


## 📊 Models and Scripts & Python Codes




## 🧠 Model of Script: Manual Federated Learning (Advanced)

<details>
<summary><strong>▶️ Click to expand the code (Python Script)</strong></summary>

```python
# Denis - Manual Federated Learning without type/shape/dict errors
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time, os, psutil
from sklearn.metrics import classification_report, confusion_matrix

# =============== PREPROCESSING ===============
df = pd.read_csv("dataset/__processed_kdd.csv", low_memory=False)

# ... script continues normally
# Final lines of the script go here

</details>
```


