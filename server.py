"""
SAKINA FEDERATED LEARNING SERVER: Strategy Comparison & Evaluation
-----------------------------------------------------------------
DESCRIPTION:
This server manages the global training process for the Sakina stress detection system. 
Its primary purpose is to benchmark different Federated Learning (FL) strategies 
(FedAvg, FedProx, FedAdam, etc.) to determine which optimizes the global model 
most effectively while maintaining privacy.

KEY FUNCTIONS:
- Strategy Testing: Allows the user to select specific FL aggregation algorithms.
- Global Model Management: Loads a pre-trained MLP model and updates it using 
  aggregated weights from decentralized clients.
- Evaluation: Aggregates metrics (Accuracy and Recall) from all participating clients
  to track global performance over multiple rounds.
"""

import flwr as fl
import tensorflow as tf
import os
import sys
import logging
from flwr.common import Metrics

# ----------------------------
# ========== CONFIG ==========
# ----------------------------
#Modle path
PRETRAINED_MODEL_PATH = r"D:\Sakina\model training\10000_subj_training_ARC\Data\processed_out\MLP_big_epoch.h5"
#FL rounds
rnd = 100 

logging.basicConfig(
    filename="fl_results.txt",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# ----------------------------
# ===== YOUR FOCAL LOSS ======
# ----------------------------
def focal_loss(alpha=0.25, gamma=2.0):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        bce = - (y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        return tf.reduce_mean(alpha_t * tf.pow(1.0 - p_t, gamma) * bce)
    return loss_fn

# ----------------------------
# ===== GLOBAL ARCHITECTURE ==
# ----------------------------
def get_model():
    if not os.path.exists(PRETRAINED_MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {PRETRAINED_MODEL_PATH}")
    
    print(f"--- Loading Pre-trained Sakina Model ---")
    model = tf.keras.models.load_model(PRETRAINED_MODEL_PATH, compile=False)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=focal_loss(),
        metrics=["accuracy", tf.keras.metrics.Recall(name="recall")]
    )
    return model

# ----------------------------
# ===== METRICS AGGREGATION ==
# ----------------------------
def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    recalls = [num_examples * m["recall"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    avg_acc = sum(accuracies) / sum(examples)
    avg_rec = sum(recalls) / sum(examples)

    result_msg = f"Round Results - Accuracy: {avg_acc:.4f}, Recall: {avg_rec:.4f}"
    print(f"\n[SERVER] {result_msg}")
    logging.info(result_msg)

    return {"accuracy": avg_acc, "recall": avg_rec}

# ----------------------------
# ===== STRATEGY SELECTION ===
# ----------------------------
def get_strategy(strategy_name, model):
    common_params = {
        "fraction_fit": 1.0,          
        "min_fit_clients": 5,         
        "min_available_clients": 5,
        "evaluate_metrics_aggregation_fn": weighted_average,
    }

    initial_params = fl.common.ndarrays_to_parameters(model.get_weights())

    if strategy_name == "FEDAVG":
        return fl.server.strategy.FedAvg(initial_parameters=initial_params, **common_params)
    
    elif strategy_name == "FEDPROX":
        return fl.server.strategy.FedProx(initial_parameters=initial_params, proximal_mu=0.1, **common_params)
    
    elif strategy_name == "FEDADAGRAD":
        return fl.server.strategy.FedAdagrad(
            initial_parameters=initial_params, 
            eta=1e-1, 
            **common_params
        )
        
    elif strategy_name == "FEDYOGI":
        return fl.server.strategy.FedYogi(
            initial_parameters=initial_params,
            eta=1e-1, 
            beta_1=0.9, 
            beta_2=0.999,
            tau=1e-3, 
            **common_params
        )
        
    elif strategy_name == "FEDADAM":
        return fl.server.strategy.FedAdam(
            initial_parameters=initial_params,
            eta=1e-1, 
            beta_1=0.9, 
            beta_2=0.999,
            tau=1e-3, 
            **common_params
        )
        
    else:
        print(f"Warning: Strategy '{strategy_name}' not recognized. Defaulting to FEDPROX.")
        return fl.server.strategy.FedProx(initial_parameters=initial_params, proximal_mu=0.1, **common_params)

if __name__ == "__main__":
    # --- NEW INTERACTIVE PROMPT ---
    if len(sys.argv) > 1:
        cmd_strategy = sys.argv[1].upper()
    else:
        print("\n" + "="*40)
        print("     SAKINA FL SERVER INITIALIZATION")
        print("="*40)
        print("Available Strategies:")
        print("  - FEDAVG")
        print("  - FEDPROX")
        print("  - FEDADAGRAD")
        print("  - FEDYOGI")
        print("  - FEDADAM")
        print("-" * 40)
        
        cmd_strategy = input("Enter strategy name (or press Enter for FEDPROX): ").strip().upper()
        
        # Default to FedProx if the user just presses Enter
        if cmd_strategy == "":
            cmd_strategy = "FEDPROX"

    global_model = get_model()
    fl_strategy = get_strategy(cmd_strategy, global_model)

    print(f"\n--- Sakina FL Server Started ---")
    print(f"Active Strategy: {cmd_strategy}")

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=rnd),
        strategy=fl_strategy,
    )
    
    # SAVING 
    final_save = "global_model_after_FL.keras"
    global_model.save(final_save)
    print(f"\nSuccess: Smarter model saved as {final_save}")