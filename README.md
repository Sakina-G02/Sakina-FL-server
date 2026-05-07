# Sakina FL Server

This repository contains the central **Federated Learning (FL)** server for the Sakina stress monitoring system. [cite_start]It serves as the decentralized "brain" of the project, managing global model updates while ensuring user privacy[cite: 599, 616].

## Project Overview
[cite_start]Sakina is a real-time stress monitoring system that utilizes Federated Learning to train AI models without ever seeing raw user data[cite: 442, 599]. [cite_start]The server manages the global model lifecycle, benchmarks various aggregation strategies, and provides a communication bridge for mobile clients[cite: 623, 675].



---

## System Architecture & Workflow

[cite_start]The server interacts with two main layers to facilitate decentralized learning[cite: 615, 616]:

1.  [cite_start]**The Edge Layer (ESP32):** Performs local inference using a quantized TFLite model[cite: 450, 612].
2.  [cite_start]**The Mobile Layer (Flutter):** Acts as the intermediary, performing local fine-tuning on-device and communicating with this server via REST API[cite: 452, 622, 623].

![Sakina System Architecture](image_dfa095.png)

### [cite_start]High-Level FL Pipeline [cite: 454, 462]
* [cite_start]**Model Pull:** Clients download the current global weights from the server[cite: 623, 639].
* [cite_start]**Local Training:** Clients fine-tune the model on their own physiological data (BVP/Temperature)[cite: 453, 622].
* [cite_start]**Weight Push:** Clients send updated weights (not raw data) back to this server[cite: 603, 639].
* [cite_start]**Aggregation:** The server merges updates into a new, smarter global model[cite: 637, 674].

![Federated Learning Network](image_dfa074.png)

---

## Repository Components

### 1. `server.py` (Research & Benchmarking)
[cite_start]Used for evaluating FL aggregation strategies using simulated Python clients[cite: 675].
* [cite_start]**Interactive selection:** Supports **FedAvg, FedProx, FedAdagrad, FedYogi,** and **FedAdam**[cite: 682].
* **Loss Function:** Implements a custom **Focal Loss** to handle class imbalances in stress data.
* [cite_start]**Performance Tracking:** Logs Accuracy and Recall results to help identify the most reliable strategy[cite: 676].

### 2. `server_app_connection.py` (Live Mobile Bridge)
[cite_start]The production-ready bridge allowing physical mobile devices to join the FL network[cite: 623, 628].
* [cite_start]**Dual-Protocol:** Runs **Flower gRPC** (port 8080) for simulations and **Flask REST API** (port 5050) for Flutter app connections[cite: 467, 638].
* **Weight Translation:** Automatically handles transposing model weights between **Dart (Flutter)** and **Keras (Python)** formats.
* **SakinaStrategy:** A custom class that merges real-time mobile updates into the global model rounds.

### 3. `client.py` (Simulation Client)
[cite_start]Helper script to simulate independent users during the research phase[cite: 669].
* [cite_start]**Data Processing:** Uses the **WESAD dataset** to treat 17 subjects as independent clients (S2–S17)[cite: 447, 496, 668].
* [cite_start]**Feature Extraction:** Generates 5 key features (BVP Mean/Std, TEMP Mean/Std/Slope) from 60-second windows[cite: 446, 606, 633].

---

## Benchmarking & Strategy Selection

[cite_start]We evaluated multiple strategies to find a balance between **Accuracy** and **Recall** to avoid "fake accuracy"[cite: 503, 676, 678].

![FL Strategy Results Table](image_dfa0ce.png)

| Strategy | Epoch 3 (100 Rnds) | Result |
| :--- | :--- | :--- |
| **FedAvg** | Acc: 0.883, Rec: 0.625 | [cite_start]High accuracy, but low recall[cite: 677]. |
| **FedYogi** | Acc: 0.728, Rec: 0.990 | [cite_start]High recall, but unacceptable accuracy[cite: 679]. |
| **FedAdam** | **Acc: 0.819, Rec: 0.826** | [cite_start]**Optimal Balance (Selected Strategy)[cite: 507, 680].** |

---

## Getting Started

### 1. Prerequisites
* Place your pre-trained base model (`MLP_big_epoch.h5`) in the root directory.
* Install dependencies: `pip install flwr tensorflow flask numpy scikit-learn`

### 2. Run Simulation Benchmarking
```bash
python server.py
# Select FEDADAM for the best-balanced performance.