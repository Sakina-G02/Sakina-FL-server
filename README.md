# Sakina FL Server & Simulation

This repository contains the central Federated Learning (FL) infrastructure for the Sakina project. It includes tools for both high-level strategy benchmarking and live mobile application integration.

---

## System Components

### 1. FL Simulation Server (`server.py`)
The central coordinator used to benchmark global model performance across a decentralized network.
* [cite_start]**Strategy Selection:** Benchmarks multiple algorithms—**FEDAVG, FEDPROX, FEDADAGRAD, FEDYOGI,** and **FEDADAM**—to find the optimal balance of Accuracy vs. Recall[cite: 624, 675].
* [cite_start]**Global Aggregation:** Merges updated weights from various clients into a refined global model after each communication round[cite: 501, 674].
* **Custom Loss:** Employs **Focal Loss** to ensure the model remains sensitive to rare stress events during the aggregation phase.

### 2. FL Simulation Client (`client.py`)
Simulates a single user's device by processing specific subjects from the **WESAD dataset**.
* [cite_start]**Subject-Specific Data:** Loads physiological data for subjects **S2 through S17** (excluding S12), extracting **BVP** and **Temperature** signals[cite: 668, 669].
* **WESAD Processing:** Divides raw data into **60-second windows** and applies majority labeling, mapping WESAD label '2' to **Stress (1)** and all others to **Not Stressed (0)**.
* [cite_start]**Privacy-Preserving Training:** Performs local fine-tuning using subject-specific data and transmits only model weights to the server, keeping raw data on the simulated device[cite: 599, 672].

### 3. FL App-Bridge Server (`server_app_connection.py`)
This script functions identically to the simulation server but includes a **REST API bridge** for real-world deployment.
* [cite_start]**Mobile Connectivity:** Provides dedicated endpoints for the **Sakina Flutter App** to pull the global model and push local weights via HTTP[cite: 623, 639].
* **Weight Conversion:** Automatically handles the technical translation of weights between **Dart (Flutter)** and **Keras (Python)** formats.

---

## System Architecture

![Sakina System Architecture](ssa.png)

[cite_start]The Sakina system follows a tiered architecture[cite: 616]:
1. **Edge Device (ESP32):** Real-time inference on physiological data.
2. **Mobile App (Flutter):** Local on-device training and server communication.
3. **FL Server (Python):** Global weight aggregation and strategy management.

---

## How to Run the Simulation

### Step 1: Download the Dataset
You must have the WESAD dataset downloaded to run the simulation. 
[ [https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html] ]

### Step 2: Automatic Execution
Run the provided `run_simulation.bat` file in this repository. The script automates the benchmarking process as follows:
1.  **Select Strategy:** It lists available strategies and prompts for your choice (e.g., `FEDADAM`).
2.  **Launch Server:** Starts the central FL server using your chosen strategy.
3.  [cite_start]**Launch Clients:** Automatically initializes 15 independent clients representing subjects **S2–S11** and **S13–S17** to begin collaborative training.

### Step 3: Deployment (Optional)
If you wish to connect a physical mobile device instead of running a simulation, execute:
`python server_app_connection.py`

---
**Sakina Project** | *Privacy-Preserving Stress Monitoring using Federated Learning*