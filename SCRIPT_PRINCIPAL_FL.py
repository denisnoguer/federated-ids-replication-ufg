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
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Bloco para plotar e mostrar gráfico em uma janela popup
def mostrar_grafico_tkinter(acc_central, acc_fed):
    fig, ax = plt.subplots(figsize=(6,4))
    bars = ax.bar(["Centralizado", "Federado"], [acc_central, acc_fed], color=["#3A5BA0", "#7CB342"])
    ax.set_ylabel("Acurácia")
    ax.set_title("Centralizado vs Federado Manual")
    ax.set_ylim(0, 1.05)
    for i, v in enumerate([acc_central, acc_fed]):
        ax.text(i, v + 0.03, f"{v:.4f}", ha="center", fontsize=12)
    plt.tight_layout()

    # Integra Matplotlib ao Tkinter
    root = tk.Tk()
    root.title("Comparativo de Acurácia")
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    tk.Button(root, text="Fechar", command=root.destroy, font=("Arial", 12)).pack(pady=10)
    root.mainloop()

# No final do script, CHAME:
mostrar_grafico_tkinter(acc_central, acc_fed)

plt.show()

