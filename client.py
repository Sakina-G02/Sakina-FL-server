"""
SAKINA FEDERATED LEARNING CLIENT: Local Training & WESAD Processing
------------------------------------------------------------------
DESCRIPTION:
This client script performs local model training on decentralized physiological data. 
It connects to the Sakina FL Server to receive global weights, fine-tunes them 
on subject-specific data, and sends the updated parameters back.

THE WESAD DATASET:
The 'WESAD' (Wearable Stress and Affect Detection) dataset is a publicly available 
dataset containing physiological data from 15 subjects. For the Sakina project, 
we focus on the following 'Wrist' sensor modalities:[cite: 1]
- BVP (Blood Volume Pulse): Used to derive heart rate variations.
- TEMP (Skin Temperature): Monitored for changes associated with stress responses.

DATA UTILIZATION:
1. Windowing: The raw data is divided into 60-second windows with a 30-second overlap.
2. Labeling: We use a majority-label strategy. WESAD label '2' (Stress) is mapped 
   to 1, while all other states (baseline, amusement, etc.) are mapped to 0.[cite: 2]
3. Feature Extraction: For every window, we calculate 5 key features: BVP (Mean, Std) 
   and Temperature (Mean, Std, Slope).
4. Balancing: Since stress events are rare, training uses a class weight (10.0 for 
   stress) to ensure the model prioritizes identifying stress accurately.[cite: 2]
"""

import sys
import pickle
import numpy as np
import flwr as fl
import tensorflow as tf
from collections import Counter
from sklearn.preprocessing import StandardScaler

# ----------------------------
# ========== CONFIG ==========
# ----------------------------

# Reads Subject ID from the automation .bat script
if len(sys.argv) > 1:
    SUBJECT_ID = sys.argv[1] 
else:
    SUBJECT_ID = "S3" 

# Path to WESAD subjects on your local drive
PKL_PATH = rf"D:\Sakina\model training\Datasets\WESAD\{SUBJECT_ID}\{SUBJECT_ID}.pkl"

# Sakina Project Windowing Logic
WINDOW_SIZE_SEC = 60 
STEP_SEC = 30

epch = 3 ##################################

# Frequencies for sensor data 
FS_BVP = 64
FS_TEMP = 4
FS_LABEL = 700

# ----------------------------
# ===== LOGIC =====
# ----------------------------
def aggregate_label(win_labels: np.ndarray) -> int:
    """Exactly your majority label strategy for stress detection."""
    cnt = Counter(win_labels)
    if len(cnt) == 0:
        return 0
    # choose label with highest count, tie -> choose 1
    best = sorted(cnt.items(), key=lambda x: (x[1], x[0]))[-1][0]
    # WESAD specific: 2 is stress, map to 1. All others map to 0.
    return 1 if int(best) == 2 else 0

def process_wesad_client_data(pkl_path):
    print(f"[{SUBJECT_ID}] Opening pickle file... please wait.")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    bvp = data['signal']['wrist']['BVP'].flatten()
    temp = data['signal']['wrist']['TEMP'].flatten()
    labels = data['label'].flatten()

    X, y = [], []
    total_sec = len(temp) // FS_TEMP
    
    print(f"[{SUBJECT_ID}] Extracting 5 features and creating windows...")
    for start_sec in range(0, total_sec - WINDOW_SIZE_SEC + 1, STEP_SEC):
        b_start, b_end = start_sec * FS_BVP, (start_sec + WINDOW_SIZE_SEC) * FS_BVP
        t_start, t_end = start_sec * FS_TEMP, (start_sec + WINDOW_SIZE_SEC) * FS_TEMP
        l_start, l_end = start_sec * FS_LABEL, (start_sec + WINDOW_SIZE_SEC) * FS_LABEL

        win_bvp = bvp[b_start:b_end]
        win_temp = temp[t_start:t_end]
        win_labels = labels[l_start:l_end]

        if len(win_bvp) == 0 or len(win_temp) == 0:
            continue

        # Feature Extraction: BVP Mean/Std and TEMP Mean/Std/Slope
        bvp_mean = float(np.mean(win_bvp))
        bvp_std  = float(np.std(win_bvp))
        temp_mean = float(np.mean(win_temp))
        temp_std  = float(np.std(win_temp))
        temp_slope = float((win_temp[-1] - win_temp[0]) / max(1.0, float(len(win_temp))))

        X.append([bvp_mean, bvp_std, temp_mean, temp_std, temp_slope])
        y.append(aggregate_label(win_labels))

    print(f"[{SUBJECT_ID}] Data Ready: Created {len(X)} windows.")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1)

# ----------------------------
# ===== FL CLIENT CLASS ======
# ----------------------------
class SakinaClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train

    def get_parameters(self, config):
        return self.model.get_weights()

    # Inside SakinaClient class in client.py
    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        print(f"\n--- [{SUBJECT_ID}] Training with Class Weights ---")
        
        # Force the model to prioritize the Stress class (Label 1)
        class_weight = {0: 1.0, 1: 10.0} 
        
        self.model.fit(
            self.x_train, 
            self.y_train, 
            epochs=epch, 
            batch_size=32, 
            class_weight=class_weight, 
            verbose=1
        )
        print(f"--- [{SUBJECT_ID}] Training finished. ---")
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        """Returns accuracy AND recall to prevent server KeyError."""
        self.model.set_weights(parameters)
        results = self.model.evaluate(self.x_train, self.y_train, verbose=0)
        
        # results order: [loss, accuracy, recall]
        loss = results[0]
        accuracy = results[1]
        recall = results[2]
        
        return loss, len(self.x_train), {"accuracy": accuracy, "recall": recall}

if __name__ == "__main__":
    X, y = process_wesad_client_data(PKL_PATH)
    
    # Local Feature Scaling
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    # Rebuilding MLP Architecture 
    inputs = tf.keras.Input(shape=(5,))
    x = tf.keras.layers.Dense(64, activation="relu")(inputs)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Metrics must match server expectation
    model.compile(
        optimizer="adam", 
        loss="binary_crossentropy", 
        metrics=["accuracy", tf.keras.metrics.Recall(name="recall")]
    )

    print(f"[{SUBJECT_ID}] Connecting to Sakina FL Server...")
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=SakinaClient(model, X_s, y))
