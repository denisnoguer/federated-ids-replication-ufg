## TUTORIAL COMPLETO DO ESTUDO UTILIZADO ### 

Português Brasil
Este projeto realiza a replicação do pipeline de aprendizado federado aplicado à detecção de intrusos (IDS), comparando o desempenho entre o treinamento centralizado tradicional e a abordagem federada manual, utilizando o dataset clássico KDD Cup 99.
O objetivo é demonstrar na prática os desafios, limitações e oportunidades do aprendizado federado (FL) em cenários reais de segurança, além de servir como base para futuras pesquisas e aprimoramentos.

** TODOS OS ARQUIVOS USADOS NESTE PROJETO ESTAO DISPONIVES NO GOOGLE DRIVE NO LINK ABAIXO, PODEM SER USADOS LIVREMENTE PARA APRENDIZADO, mas antes mande um e-mail para email: denisnogueira@discente.ufg.br

LINK GOOGLE DRIVE ARQUIVOS FL
https://drive.google.com/drive/folders/1EjGZGoUtxG6pSGt9K_gUqYqvQkGtO9PW

## Requisitos Necessários ##
0. LINUX UBUNTU (ATUALIZADO)
1. Clone o repositório do GOOGLE DRIVE
https://drive.google.com/drive/folders/1EjGZGoUtxG6pSGt9K_gUqYqvQkGtO9PW
2. Instale as dependências (veja requirements.txt)
3. Execute o script principal (MODIFIQUE O CÓDIGO:
4. Os resultados e gráficos serão salvos na mesma pasta.

## Requisitos
- Python 3.9+
- FLOWER
- TensorFlow
- Pandas
- Scikit-learn
- Matplotlib
- (e outras listadas em requirements.txt)

LISTAGEM DA METODOLOGIA TÉCNICA UTILIZADA PARA APLICAÇÃO DE FL

1.	Preparação do ambiente: • Instalação Docker / Docker Compose
• Criação de ambiente Python e instalação de libs (Flower, TensorFlow/PyTorch, scikit-learn)
2.	Pré-processamento de dados - • Download e particionamento KDD-99, MotionSense etc.
• Encoding e normalização
• Gerar partições para FL
3.	 Modelo centralizado- • Implementar e treinar MLP local
• Validar acurácia de referência
4.	 Orquestração FL básica- • Criar server.py e client.py no Flower
• Ajustar docker-compose.yml para 20 clientes
5.	 Experimentos FL (IDS) - • Rodar FL variando N, E e B
• Coleta de logs e métricas de acurácia
6.	 Extensão DDoS -• Descompactar DDos.zip
• Configurar ambiente e dados DDoS
• Executar e coletar resultados


1- TUTORIAL DE INSTALAÇÃO DAS DEPENDÊNCIAS
	(Obs: Foram utilizadas as versões mais atuais do Linux Ubuntu)

Instalação das dependências
Python (VIRTUAL)
Passo 2 – Criar e ativar o ambiente virtual Python
Dentro de /0/IDS, execute:
bash
CopiarEditar
python3 -m venv venv
source venv/bin/activate


Passo B – Instalar as dependências manualmente
bash
# Ainda com o (venv) ativo em /0/IDS:
pip install --upgrade pip

# Instalar as bibliotecas essenciais:
pip install tensorflow flwr pandas numpy scikit-learn matplotlib


##Instalação da Dependências e atualização das bibliotecas.

 
2- Dependências que devem ser Instaladas
●	TensorFlow (treino do MLP)

●	Flower (flwr) (orquestração federada)

●	pandas, numpy, scikit-learn (pré-processamento)

●	matplotlib (gráficos)

3- LISTAGEM DE FLAGS DISPONIVEIS disponíveis (como --epochs, --batch-size etc.). 
Treino centralizado (LOW)
Configurar o Federeted Learning
Coletar resultados
 
4-  ETAPA - CONFIGURAÇÃO DO DOCKER
Serviço server – garanta que use só 25 rodadas:
Serviço client – configure 3 réplicas, 2 épocas locais e 1 rodada:

Listagem de Comandos importantes para ativação do ambiente Docker
NAVEGUE ATE /0/IDS
ATIVE O MODO VIRTUAL
source venv/bin/activate

Comando para Executar o  APRENDIZADO FEDERADO em ambiente virtualizado em Linux com Python.
Obs: Comando deve ser executado no diretório raiz do projeto.

python fl_manual_avancado.py
(NOME DO ARQUIVO ESCOLHIDO)

5. Lista de nomes dos tipos de ataques utilizados:
Os nomes que aparecem (ex: back, buffer_overflow, ipsweep, etc.) são tipos de ataques ou atividades de rede detectadas/classificadas no dataset.
 Ao lado deles, aparecem siglas como dos, u2r, r2l, probe, que são categorias de ataques.  Tipos de ataques usados:
●	dos: Denial of Service (negação de serviço). Ex: smurf, neptune, teardrop, pod, land, back

●	u2r: User to Root. Ataques onde um usuário comum tenta obter acesso de root/admin. Ex: buffer_overflow, loadmodule, perl, rootkit

●	r2l: Remote to Local. Alguém de fora tenta obter acesso ao sistema como usuário local. Ex: ftp_write, guess_passwd, imap, multihop, phf, spy, warezclient, warezmaster

●	probe: Atividades de sondagem, escaneamento de portas, etc. Ex: ipsweep, nmap, portsweep, satan

Visão tradicional & visionária
●	Tradicional: O uso desses datasets é clássico em segurança cibernética — profissionais do mundo todo usam há décadas como referência, ajudando a construir a base de IDS modernos.
MODELO GRÁFICO DO CENÁRIO DE APRENDIZADO FEDERADO APLICADO EM DOIS CENÁRIOS, 1 TRADICIONAL CENTRALIZADO E O SEGUNDO O APRENDIZADO FEDERADO.
 


6. Construir e subir os containers réplica de 3 clientes
 Este próximo passo deve ser feito diretamente em local da máquina do diretório escolhido

Neste projeto foi usado /0/IDS


8- APLICAÇÃO DDOS 
 
Test accuracy: 0.958790123462677
0
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step
[[0.9807248 0.0192752]]
1
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step
[[0.00809862  VS  0.99190134]]

-=--=-=-=-=-=-=-=-=-=-==--==--=-=-==-=-=-=-=-=-=-=-

9. Valores dos dados e Acurácia Executados no experimento
Cenário	Acurácia (%)
Centralizado	99,16
Federado (FedAvg, melhor configuração)	99,40 (+0,24 pp)
Federado (FedAvg, pior configuração)	96,62 (−2,54 pp)

9.1 -  Gráfico comparativo

 Mesmo que a acurácia federada manual (0.0988) tenha ficado baixa, você agora tem:
●	Pipeline federado funcional 

●	Gráfico comparativo 

AO CONCLUIR O CODIGO DEVE GERAR O GRÁFICO OCOMPARATIVO

 
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
SCRIPT ML MANUAL  AVANÇADO - 

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

# Mapear labels e categorias, SEM ERRO
df["_Attack Type"] = df["Attack Type"].map({
    "normal": 0, "dos": 1, "probe": 2, "r2l": 3, "u2r": 4
}).astype(np.int32)
df["protocol_type"] = df["protocol_type"].map({"tcp": 0, "udp": 1, "icmp": 2}).astype(np.float32)
df["flag"] = df["flag"].map({
    "SF": 0, "S1": 1, "REJ": 2, "S2": 3, "S0": 4,
    "S3": 5, "RSTO": 6, "RSTR": 7, "RSTOS0": 8, "OTH": 9, "SH": 10
}).astype(np.float32)

# CONVERTE TODAS as colunas que restarem para float32
for col in df.columns:
    if col not in ["Attack Type", "_Attack Type"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)

# REMOVE STRINGS e garante que só tem número
df = df.select_dtypes(include=[np.number])

# Remove linhas com NaN (se existirem) - opcional, pode substituir por 0 se quiser
df = df.fillna(0)

# Separa X e y
X = df.drop(columns=["_Attack Type"]).to_numpy(dtype=np.float32)
y = df["_Attack Type"].to_numpy(dtype=np.int32)

# =============== TREINO / TESTE SPLIT ===============
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# =============== FUNÇÃO DE MODELO ===============
def build_model(learning_rate=0.001):
    m = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(5, activation="softmax"),
    ])
    m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return m

# =============== TREINAMENTO CENTRALIZADO ===============
print("\n=== CENTRALIZADO ===")
model_central = build_model()
start_time = time.time()
model_central.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
tempo_central = time.time() - start_time
print(f"Tempo treino centralizado: {tempo_central:.2f}s")
loss_c, acc_central = model_central.evaluate(X_test, y_test, verbose=0)
print(f"Acurácia Centralizada: {acc_central:.4f}")

y_pred_central = np.argmax(model_central.predict(X_test), axis=1)
print("Classification Report Centralizado:")
print(classification_report(y_test, y_pred_central, digits=4))
print("Confusion Matrix Centralizado:")
print(confusion_matrix(y_test, y_pred_central))

# =============== FEDERADO MANUAL (3 CLIENTES) ===============
print("\n=== FEDERADO MANUAL ===")
learning_rates = [0.001, 0.01, 0.005]
num_clients = 3
splits_X = np.array_split(X_train, num_clients)
splits_y = np.array_split(y_train, num_clients)
client_weights = []
train_times = []

for cid in range(num_clients):
    print(f"Cliente {cid+1}, taxa aprendizado={learning_rates[cid]}")
    model = build_model(learning_rate=learning_rates[cid])
    t0 = time.time()
    model.fit(splits_X[cid], splits_y[cid], epochs=2, batch_size=32, verbose=0)
    t1 = time.time()
    train_times.append(t1 - t0)
    client_weights.append(model.get_weights())
    print(f"Tempo treino cliente {cid+1}: {train_times[-1]:.2f}s")

print(f"Tempo total federado: {sum(train_times):.2f}s")
# Tamanho do modelo em MB
weights_size = sum(w.nbytes for w in client_weights[0])
print(f"Tamanho dos pesos (por rodada): {weights_size/1024**2:.2f}MB")
dataset_size = os.path.getsize('dataset/__processed_kdd.csv')
print(f"Tamanho do dataset centralizado: {dataset_size/1024**2:.2f}MB")

# Agregação FedAvg manual
avg_weights = [np.mean(layer_weights, axis=0) for layer_weights in zip(*client_weights)]
global_model = build_model()
global_model.set_weights(avg_weights)
loss_f, acc_fed = global_model.evaluate(X_test, y_test, verbose=0)
print(f"Acurácia Federada Manual: {acc_fed:.4f}")

y_pred_fed = np.argmax(global_model.predict(X_test), axis=1)
print("Classification Report Federado:")
print(classification_report(y_test, y_pred_fed, digits=4))
print("Confusion Matrix Federado:")
print(confusion_matrix(y_test, y_pred_fed))

# GRÁFICO
plt.figure(figsize=(6,4))
plt.bar(["Centralizado", "Federado"], [acc_central, acc_fed], color=["#3A5BA0", "#7CB342"])
plt.ylabel("Acurácia")
plt.title("Centralizado vs Federado Manual")
plt.ylim(0, 1.05)
for i, v in enumerate([acc_central, acc_fed]):
    plt.text(i, v + 0.03, f"{v:.4f}", ha="center", fontsize=12)
plt.tight_layout()
plt.savefig("comparacao_centralizado_federado_final.png", dpi=150)
plt.show()


==-=-=-=-=====-=-=-==-==-=-=-=-=-=-=-=-=-=-=-=-=-===-=-==-=-=-=-=====-=-=-==-=
Reference:
●	DAMACENO, Alexsander; C. RIBEIRO, Maria do Rosário; OLIVEIRA-JR, Antonio; DE OLIVEIRA, Renan R.. Abordagem Baseada em Aprendizado Federado para a Detecção de Intrusão em Redes de Computadores. In: ESCOLA REGIONAL DE INFORMÁTICA DE GOIÁS (ERI-GO), 11. , 2023, Goiânia/GO. 



Goiânia, 04 de junho de 2024
Autor – DENIS NOGUEIRA DO NASCIMENTO – denisnoguer@gmail.com
