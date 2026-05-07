"""
SAKINA HYBRID FL SERVER: Flower gRPC & Flutter REST Bridge
------------------------------------------------------------------
DESCRIPTION:
This script implements a dual-interface Federated Learning server. It allows 
traditional Python-based simulation clients to connect via gRPC while 
simultaneously providing a REST API for production Flutter mobile clients.

HYBRID ARCHITECTURE:
1. Flower gRPC (Port 8080): Handles high-speed communication with Python clients 
   using the standard Flower protocol.
2. Flask REST API (Port 5050): Acts as a bridge for Flutter clients that 
   cannot natively communicate via gRPC. It handles model downloads and 
   weight uploads via standard HTTP POST/GET requests.
3. SakinaStrategy: A custom aggregation class that extends FedAdam. It 
   silently merges queued updates from Flutter apps into the global model 
   at the end of each Federated round.

CROSS-PLATFORM COMPATIBILITY:
- Weight Transposition: Dart (Flutter) and Keras (Python) store neural 
  network weights in different shapes. This script automatically transposes 
  kernels between [fanOut, fanIn] and [fanIn, fanOut] formats during 
  serialization to ensure model consistency across devices.
- Thread Safety: Uses mutex locks to safely manage a global weights cache 
  accessible by both the training thread and the REST API thread.

API ENDPOINTS:
- GET  /api/global_model: Downloads the latest global weights in Dart format.
- POST /api/local_update: Receives locally trained weights from the Flutter app.
- GET  /api/status: Health check for the server and aggregation queue.
"""

import flwr as fl
import tensorflow as tf
import numpy as np
import os
import sys
import json
import logging
import threading
from flask import Flask, request, jsonify
from flwr.common import Metrics, ndarrays_to_parameters, parameters_to_ndarrays

# ----------------------------
# ========== CONFIG ==========
# ----------------------------
PRETRAINED_MODEL_PATH = r"MLP_big_epoch.h5"
NUM_ROUNDS   = 10
REST_PORT    = 5050   # Flutter app connects here
FLOWER_PORT  = 8080   # Python simulation clients connect here

logging.basicConfig(
    filename="fl_results.txt",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# ----------------------------
# ===== SHARED STATE =========
# ----------------------------
# Holds the latest global model weights (as Keras ndarrays).
# Updated after every Flower aggregation round.
# Read by the REST /api/global_model endpoint.
global_weights_lock  = threading.Lock()
global_weights_cache = None   # list of np.ndarray, set after first round

# Queue of Flutter client updates waiting to be merged.
# Each entry: {"weights": [np.ndarray, ...], "num_samples": int}
flutter_updates_lock = threading.Lock()
flutter_updates_queue = []

# ----------------------------
# ===== FOCAL LOSS ===========
# ----------------------------
def focal_loss(alpha=0.25, gamma=2.0):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        epsilon = tf.keras.backend.epsilon()
        y_pred  = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        bce     = -(y_true * tf.math.log(y_pred) +
                    (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        p_t     = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        alpha_t = y_true * alpha  + (1.0 - y_true) * (1.0 - alpha)
        return tf.reduce_mean(alpha_t * tf.pow(1.0 - p_t, gamma) * bce)
    return loss_fn

# ----------------------------
# ===== MODEL ================
# ----------------------------
def get_model():
    if not os.path.exists(PRETRAINED_MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {PRETRAINED_MODEL_PATH}")
    print("--- Loading Pre-trained Sakina Model ---")
    model = tf.keras.models.load_model(PRETRAINED_MODEL_PATH, compile=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=focal_loss(),
        metrics=["accuracy", tf.keras.metrics.Recall(name="recall")]
    )
    return model

# ----------------------------
# ===== WEIGHT CONVERSION ====
# ----------------------------
# Dart MLP stores weights as [fanOut][fanIn]  (W[j][i] = weight to j from i)
# Keras Dense stores weights as [fanIn][fanOut] (kernel shape = (in, out))
# So every kernel needs to be transposed when crossing the boundary.
# Biases have the same shape in both ([fanOut]) — no conversion needed.

def dart_weights_to_keras(dart_json: dict) -> list:
    """
    Convert Flutter/Dart exported weight JSON → list of Keras ndarrays.
    Keras model.get_weights() order:
      [kernel_0, bias_0, kernel_1, bias_1, kernel_2, bias_2]
    """
    w0k = np.array(dart_json["w0k"], dtype=np.float32)  # Dart: (64, 5)
    w0b = np.array(dart_json["w0b"], dtype=np.float32)  # (64,)
    w1k = np.array(dart_json["w1k"], dtype=np.float32)  # Dart: (32, 64)
    w1b = np.array(dart_json["w1b"], dtype=np.float32)  # (32,)
    w2k = np.array(dart_json["w2k"], dtype=np.float32)  # Dart: (1, 32)
    w2b = np.array(dart_json["w2b"], dtype=np.float32)  # (1,)

    # Transpose kernels: Dart [fanOut, fanIn] → Keras [fanIn, fanOut]
    return [w0k.T, w0b, w1k.T, w1b, w2k.T, w2b]


def keras_weights_to_dart(keras_weights: list) -> dict:
    """
    Convert Keras model.get_weights() → Flutter/Dart JSON weight format.
    keras_weights order: [kernel_0, bias_0, kernel_1, bias_1, kernel_2, bias_2]
    """
    k0, b0, k1, b1, k2, b2 = keras_weights

    # Transpose kernels: Keras [fanIn, fanOut] → Dart [fanOut, fanIn]
    return {
        "architecture": [5, 64, 32, 1],
        "w0k": k0.T.tolist(),   # (64, 5)
        "w0b": b0.tolist(),     # (64,)
        "w1k": k1.T.tolist(),   # (32, 64)
        "w1b": b1.tolist(),     # (32,)
        "w2k": k2.T.tolist(),   # (1, 32)
        "w2b": b2.tolist(),     # (1,)
    }

# ----------------------------
# ===== METRICS AGGREGATION ==
# ----------------------------
def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    accuracies = [n * m["accuracy"] for n, m in metrics]
    recalls    = [n * m["recall"]   for n, m in metrics]
    examples   = [n for n, _ in metrics]

    avg_acc = sum(accuracies) / sum(examples)
    avg_rec = sum(recalls)    / sum(examples)

    msg = f"Round Results - Accuracy: {avg_acc:.4f}, Recall: {avg_rec:.4f}"
    print(f"\n[SERVER] {msg}")
    logging.info(msg)
    return {"accuracy": avg_acc, "recall": avg_rec}

# ----------------------------
# ===== CUSTOM STRATEGY ======
# ----------------------------
class SakinaStrategy(fl.server.strategy.FedAdam):
    """
    Extends FedAdam to also merge any pending Flutter app weight updates
    into each aggregation round using weighted FedAvg before FedAdam applies
    its adaptive server-side update.
    """

    def aggregate_fit(self, server_round, results, failures):
        # 1. Run the normal Flower aggregation on Python simulation clients
        aggregated_params, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_params is None:
            return aggregated_params, aggregated_metrics

        # 2. Collect pending Flutter updates (if any)
        with flutter_updates_lock:
            pending = list(flutter_updates_queue)
            flutter_updates_queue.clear()

        if pending:
            print(f"\n[SERVER] Round {server_round}: merging "
                  f"{len(pending)} Flutter client update(s) into aggregation.")

            # Convert aggregated Flower result back to ndarrays
            flower_weights  = parameters_to_ndarrays(aggregated_params)
            # Count total samples from Flower clients
            flower_samples  = sum(fit_res.num_examples
                                  for _, fit_res in results if fit_res is not None)

            # Weighted average: include Flutter updates
            all_weights = [(flower_samples, flower_weights)] + \
                          [(u["num_samples"], u["weights"]) for u in pending]

            total_samples = sum(n for n, _ in all_weights)
            merged = [
                sum(n * w for (n, ws) in all_weights for w in [ws[i]]) / total_samples
                for i in range(len(flower_weights))
            ]

            # Correct weighted average per layer
            n_layers = len(flower_weights)
            merged = []
            for i in range(n_layers):
                layer_avg = sum(
                    n * ws[i] for n, ws in all_weights
                ) / total_samples
                merged.append(layer_avg)

            aggregated_params = ndarrays_to_parameters(merged)
            print(f"[SERVER] Merged aggregation complete "
                  f"(Flower: {flower_samples} samples, "
                  f"Flutter: {sum(u['num_samples'] for u in pending)} samples).")
        else:
            # No Flutter updates this round — use Flower result as-is
            merged = parameters_to_ndarrays(aggregated_params)

        # 3. Update the shared global weights cache (for /api/global_model)
        with global_weights_lock:
            global global_weights_cache
            global_weights_cache = parameters_to_ndarrays(aggregated_params)

        print(f"[SERVER] Round {server_round} complete. "
              f"Global model updated in REST cache.")

        return aggregated_params, aggregated_metrics


def get_strategy(model) -> fl.server.strategy.Strategy:
    initial_params = ndarrays_to_parameters(model.get_weights())
    return SakinaStrategy(
        initial_parameters=initial_params,
        fraction_fit=1.0,
        min_fit_clients=5,
        min_available_clients=5,
        evaluate_metrics_aggregation_fn=weighted_average,
        eta=1e-1,
        beta_1=0.9,
        beta_2=0.999,
        tau=1e-3,
    )

# ----------------------------
# ===== FLASK REST API =======
# ----------------------------
app = Flask(__name__)

@app.route("/api/global_model", methods=["GET"])
def get_global_model():
    """
    Flutter app calls this to download the latest global model weights.
    Returns JSON in the Dart weight format (w0k, w0b, ...).
    """
    with global_weights_lock:
        weights = global_weights_cache

    if weights is None:
        return jsonify({"error": "Global model not yet available. "
                                  "Wait for the first FL round to complete."}), 503

    response = keras_weights_to_dart(weights)
    response["round"] = "latest"
    return jsonify(response)


@app.route("/api/local_update", methods=["POST"])
def post_local_update():
    """
    Flutter app calls this to upload its locally fine-tuned weights.
    Body (JSON): same format as exportWeights() in fl_local_trainer_v3.dart
      {
        "round": 3,
        "architecture": [5, 64, 32, 1],
        "num_samples": 87,
        "w0k": [[...]], "w0b": [...],
        "w1k": [[...]], "w1b": [...],
        "w2k": [[...]], "w2b": [...]
      }
    """
    try:
        data = request.get_json(force=True)

        if not data:
            return jsonify({"error": "Empty body"}), 400

        required = ["w0k", "w0b", "w1k", "w1b", "w2k", "w2b"]
        if not all(k in data for k in required):
            return jsonify({"error": f"Missing fields. Required: {required}"}), 400

        keras_w     = dart_weights_to_keras(data)
        num_samples = int(data.get("num_samples", 1))

        with flutter_updates_lock:
            flutter_updates_queue.append({
                "weights":     keras_w,
                "num_samples": num_samples,
            })

        fl_round = data.get("round", "?")
        label_dist = data.get("label_distribution", {})
        print(f"\n[REST] Flutter update received — "
              f"samples={num_samples}, round={fl_round}, "
              f"labels={label_dist}")
        logging.info(f"Flutter update: samples={num_samples} round={fl_round} "
                     f"labels={label_dist}")

        return jsonify({"status": "ok",
                        "message": "Update queued for next aggregation round."}), 200

    except Exception as e:
        print(f"[REST ERROR] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/status", methods=["GET"])
def status():
    """Health check — Flutter app can poll this to confirm server is reachable."""
    with global_weights_lock:
        has_model = global_weights_cache is not None
    with flutter_updates_lock:
        queued = len(flutter_updates_queue)
    return jsonify({
        "status":          "running",
        "global_model":    "ready" if has_model else "waiting_for_first_round",
        "queued_updates":  queued,
    })


def run_rest_api():
    """Runs Flask in a background thread. Does not block the Flower server."""
    print(f"[REST] Starting Flask REST API on port {REST_PORT}...")
    # use_reloader=False is required when running in a thread
    app.run(host="0.0.0.0", port=REST_PORT, use_reloader=False)


# ----------------------------
# ===== MAIN =================
# ----------------------------
if __name__ == "__main__":
    # Load model and build strategy
    global_model = get_model()

    # Seed the global weights cache with pre-trained weights so Flutter
    # can pull them immediately without waiting for the first round.
    with global_weights_lock:
        global_weights_cache = global_model.get_weights()
    print("[SERVER] Pre-trained weights loaded into REST cache.")

    fl_strategy = get_strategy(global_model)

    # Start Flask REST API in a background daemon thread
    rest_thread = threading.Thread(target=run_rest_api, daemon=True)
    rest_thread.start()

    print(f"\n--- Sakina FL Server Started ---")
    print(f"Active Strategy : FedAdam")
    print(f"Flower gRPC     : port {FLOWER_PORT}  ← Python simulation clients")
    print(f"Flask REST API  : port {REST_PORT}   ← Flutter app clients")

    # Start Flower gRPC server (blocks until all rounds complete)
    fl.server.start_server(
        server_address=f"0.0.0.0:{FLOWER_PORT}",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=fl_strategy,
    )

    # Save final global model
    final_save = "global_model_after_FL.keras"
    global_model.set_weights(global_weights_cache)
    global_model.save(final_save)
    print(f"\nSuccess: Final global model saved as {final_save}")
