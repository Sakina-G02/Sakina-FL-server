// fl_local_trainer.dart  (v3 — pure Dart, no tflite_flutter)
//
// Architecture: 5 → 64 → 32 → 1  (confirmed from mlp_cgan_noHR_focal_classweight.h5)
//
// Training backend: pure Dart backprop running in a background isolate.
// Initial weights loaded from assets/sakina_initial_weights.json (pre-trained Keras
// weights exported by export_for_fl.py) so fine-tuning starts from a strong baseline.
//
// pubspec.yaml dependencies needed:
//   shared_preferences: ^2.2.0
//   http: ^1.2.1
//
// pubspec.yaml assets needed:
//   - assets/sakina_initial_weights.json
//
// NO tflite_flutter required.

import 'dart:convert';
import 'dart:math';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:shared_preferences/shared_preferences.dart';

// ─────────────────────────────────────────────────────────────
// Data model
// ─────────────────────────────────────────────────────────────

class TrainingSample {
  final List<double> rawFeatures; // 5 values BEFORE StandardScaler
  final double label;             // 0.0 = Normal, 1.0 = Stressed
  final DateTime timestamp;

  TrainingSample({
    required this.rawFeatures,
    required this.label,
    required this.timestamp,
  });

  Map<String, dynamic> toJson() => {
    'f': rawFeatures,
    'l': label,
    't': timestamp.millisecondsSinceEpoch,
  };

  factory TrainingSample.fromJson(Map<String, dynamic> j) => TrainingSample(
    rawFeatures: (j['f'] as List).map((e) => (e as num).toDouble()).toList(),
    label:       (j['l'] as num).toDouble(),
    timestamp:   DateTime.fromMillisecondsSinceEpoch(j['t'] as int),
  );
}

class TrainingResult {
  final int    epochs;
  final int    samplesUsed;
  final double finalLoss;
  final double accuracy;
  final int    flRound;

  const TrainingResult({
    required this.epochs,
    required this.samplesUsed,
    required this.finalLoss,
    required this.accuracy,
    required this.flRound,
  });
}

// ─────────────────────────────────────────────────────────────
// Isolate entry point — pure Dart backprop
// Architecture: 5 → 64 → 32 → 1  (ReLU hidden, Sigmoid output)
// Features passed in are STANDARDIZED (scaler applied before isolate)
// ─────────────────────────────────────────────────────────────

Map<String, dynamic> _trainIsolate(Map<String, dynamic> p) {
  final rawW = p['weights'] as List;
  final rawB = p['biases']  as List;

  final W = rawW
      .map((l) => (l as List)
          .map((row) => (row as List).map((e) => (e as num).toDouble()).toList())
          .toList())
      .toList();
  final B = rawB
      .map((l) => (l as List).map((e) => (e as num).toDouble()).toList())
      .toList();

  final features = (p['features'] as List)
      .map((s) => (s as List).map((e) => (e as num).toDouble()).toList())
      .toList();
  final labels    = (p['labels'] as List).map((e) => (e as num).toDouble()).toList();
  final int epochs    = p['epochs']    as int;
  final double lr     = (p['lr'] as num).toDouble();
  final int batchSize = p['batchSize'] as int;
  final int n = features.length;
  final int L = W.length;

  const double eps = 1e-7;
  double sigmoid(double x) => 1.0 / (1.0 + exp(-x.clamp(-30.0, 30.0)));
  double relu(double x)    => x > 0.0 ? x : 0.0;

  final rng = Random();
  double finalLoss = 0.0;

  for (int epoch = 0; epoch < epochs; epoch++) {
    final idx = List.generate(n, (i) => i)..shuffle(rng);
    double epochLoss = 0.0;

    for (int b = 0; b < n; b += batchSize) {
      final end   = (b + batchSize).clamp(0, n);
      final batch = idx.sublist(b, end);

      final dW = W.map((l) => l.map((row) => List.filled(row.length, 0.0)).toList()).toList();
      final dB = B.map((l) => List.filled(l.length, 0.0)).toList();
      double bLoss = 0.0;

      for (final si in batch) {
        // Forward pass
        final acts = <List<double>>[features[si].toList()];
        for (int l = 0; l < L; l++) {
          final isOut = (l == L - 1);
          final next  = <double>[];
          for (int j = 0; j < W[l].length; j++) {
            double z = B[l][j];
            for (int i = 0; i < acts.last.length; i++) z += W[l][j][i] * acts.last[i];
            next.add(isOut ? sigmoid(z) : relu(z));
          }
          acts.add(next);
        }

        final out = acts.last[0];
        final y   = labels[si];
        bLoss += -(y * log(out.clamp(eps, 1 - eps)) +
                   (1 - y) * log((1 - out).clamp(eps, 1 - eps)));

        // Backward pass
        var delta = [out - y];
        for (int l = L - 1; l >= 0; l--) {
          for (int j = 0; j < W[l].length; j++) {
            dB[l][j] += delta[j];
            for (int i = 0; i < acts[l].length; i++) dW[l][j][i] += delta[j] * acts[l][i];
          }
          if (l > 0) {
            final nd = List.filled(acts[l].length, 0.0);
            for (int i = 0; i < acts[l].length; i++) {
              double d = 0.0;
              for (int j = 0; j < W[l].length; j++) d += delta[j] * W[l][j][i];
              nd[i] = d * (acts[l][i] > 0.0 ? 1.0 : 0.0);
            }
            delta = nd;
          }
        }
      }

      epochLoss += bLoss;
      final scale = lr / batch.length;
      for (int l = 0; l < L; l++) {
        for (int j = 0; j < W[l].length; j++) {
          B[l][j] -= scale * dB[l][j];
          for (int i = 0; i < W[l][j].length; i++) W[l][j][i] -= scale * dW[l][j][i];
        }
      }
    }
    finalLoss = epochLoss / n;
  }

  // Compute accuracy
  int correct = 0;
  for (int i = 0; i < n; i++) {
    var cur = features[i].toList();
    for (int l = 0; l < L; l++) {
      final isOut = l == L - 1;
      final next  = <double>[];
      for (int j = 0; j < W[l].length; j++) {
        double z = B[l][j];
        for (int i2 = 0; i2 < cur.length; i2++) z += W[l][j][i2] * cur[i2];
        next.add(isOut ? sigmoid(z) : relu(z));
      }
      cur = next;
    }
    if ((cur[0] >= 0.5 ? 1.0 : 0.0) == labels[i]) correct++;
  }

  return {'weights': W, 'biases': B, 'finalLoss': finalLoss, 'accuracy': correct / n};
}

// ─────────────────────────────────────────────────────────────
// FLLocalTrainer
// ─────────────────────────────────────────────────────────────

class FLLocalTrainer {
  // Architecture — confirmed from h5 file: 5 → 64 → 32 → 1
  static const List<int> layerSizes = [5, 64, 32, 1];

  // Feature pipeline — matches ESP32 main.cpp exactly
  static const int windowSize = 60;

  // StandardScaler stats from scaler_stats.npz
  // Order: BVP_mean, BVP_std, TEMP_mean_norm, TEMP_std_norm, TEMP_slope
  static const List<double> scalerMean  = [ 0.5117,  0.0625,  0.6177,  0.0890, -0.0000203];
  static const List<double> scalerScale = [ 0.0317,  0.0338,  0.2587,  0.1071,  0.00471  ];

  static const int minTrainSamples  = 10;
  static const int maxStoredSamples = 500;

  static const String _kDataset = 'fl_dataset_v3';
  static const String _kWeights = 'fl_weights_v3';
  static const String _kRound   = 'fl_round_v3';

  // Rolling BLE buffers
  final List<double> _bvpBuf      = [];
  final List<double> _tempNormBuf = [];
  DateTime? _lastSampleAt;

  // Training dataset
  List<TrainingSample> _dataset = [];

  // MLP weights — shape: W[l] is [layerSizes[l+1]][layerSizes[l]]
  List<List<List<double>>> _weights = [];
  List<List<double>>       _biases  = [];

  int  _flRound     = 0;
  bool _initialized = false;

  void Function(int totalSamples)? onSampleAdded;

  // ── Init ─────────────────────────────────────────────────────────────────

  Future<void> init() async {
    if (_initialized) return;

    // Try to load pre-trained weights from asset; fall back to Xavier
    final loaded = await _tryLoadPretrainedWeights();
    if (!loaded) _initXavier();

    await _loadPersisted(); // may override with user's fine-tuned weights
    _initialized = true;
  }

  /// Load sakina_initial_weights.json from assets.
  /// Format: {"w0k": [[...]], "w0b": [...], "w1k": ..., "w1b": ..., "w2k": ..., "w2b": ...}
  Future<bool> _tryLoadPretrainedWeights() async {
    try {
      final raw  = await rootBundle.loadString('assets/sakina_initial_weights.json');
      final json = jsonDecode(raw) as Map<String, dynamic>;
      _weightsFromFlatJson(json);
      debugPrint('[FLLocalTrainer] Loaded pre-trained weights from asset');
      return true;
    } catch (e) {
      debugPrint('[FLLocalTrainer] Could not load pre-trained weights ($e) — using Xavier init');
      return false;
    }
  }

  void _initXavier() {
    final rng = Random(12345);
    _weights = [];
    _biases  = [];
    for (int l = 0; l < layerSizes.length - 1; l++) {
      final fanIn  = layerSizes[l];
      final fanOut = layerSizes[l + 1];
      final limit  = sqrt(6.0 / (fanIn + fanOut));
      _weights.add(List.generate(fanOut, (_) =>
          List.generate(fanIn, (_) => (rng.nextDouble() * 2 - 1) * limit)));
      _biases.add(List.filled(fanOut, 0.0));
    }
  }

  // ── Data Collection ───────────────────────────────────────────────────────

  /// Feed one BLE packet into the rolling window.
  /// Returns true when a full 60-sample window completes (new sample saved).
  bool onBleData({
    required double bvp,
    required double tempCelsius,
    required String stressLabel,
    String? userLabel,
  }) {
    if (!_initialized) return false;

    final now = DateTime.now();
    if (_lastSampleAt != null &&
        now.difference(_lastSampleAt!).inMilliseconds < 800) return false;
    _lastSampleAt = now;

    // (T - 30) / 10 — matches normalizeTempC() in ESP32 main.cpp
    _bvpBuf.add(bvp);
    _tempNormBuf.add((tempCelsius - 30.0) / 10.0);

    while (_bvpBuf.length      > windowSize) _bvpBuf.removeAt(0);
    while (_tempNormBuf.length > windowSize) _tempNormBuf.removeAt(0);
    if (_bvpBuf.length < windowSize) return false;

    final raw   = _extractRawFeatures(_bvpBuf, _tempNormBuf);
    final label = (userLabel ?? stressLabel).toLowerCase() == 'stressed' ? 1.0 : 0.0;

    _dataset.add(TrainingSample(rawFeatures: raw, label: label, timestamp: now));
    if (_dataset.length > maxStoredSamples) _dataset.removeAt(0);

    _saveDataset();
    onSampleAdded?.call(_dataset.length);
    return true;
  }

  List<double> _extractRawFeatures(List<double> bvp, List<double> tempNorm) {
    final bvpMean   = _mean(bvp);
    final bvpStd    = _std(bvp, bvpMean);
    final tempMean  = _mean(tempNorm);
    final tempStd   = _std(tempNorm, tempMean);
    final tempSlope = (tempNorm.last - tempNorm.first) / windowSize.toDouble();
    return [bvpMean, bvpStd, tempMean, tempStd, tempSlope];
  }

  // ── Inference (on-device, personalized) ──────────────────────────────────

  /// Returns stress probability [0,1] for a raw 5-feature window.
  /// Call after enough training has happened for personalized results.
  double predictLocal(List<double> rawFeatures) {
    final x = _standardize(rawFeatures);
    var cur = x.toList();
    for (int l = 0; l < _weights.length; l++) {
      final isOut = l == _weights.length - 1;
      final next  = <double>[];
      for (int j = 0; j < _weights[l].length; j++) {
        double z = _biases[l][j];
        for (int i = 0; i < cur.length; i++) z += _weights[l][j][i] * cur[i];
        next.add(isOut
            ? 1.0 / (1.0 + exp(-z.clamp(-30.0, 30.0)))
            : (z > 0.0 ? z : 0.0));
      }
      cur = next;
    }
    return cur[0];
  }

  // ── Training ──────────────────────────────────────────────────────────────

  Future<TrainingResult> trainLocal({
    int epochs = 3,
    double learningRate = 0.001,
    int batchSize = 16,
  }) async {
    if (_dataset.length < minTrainSamples) {
      throw Exception('Need $minTrainSamples samples — have ${_dataset.length}');
    }

    // Standardize features before passing to isolate
    final stdFeatures = _dataset.map((s) => _standardize(s.rawFeatures)).toList();

    final result = await compute(_trainIsolate, {
      'weights':   _weights,
      'biases':    _biases,
      'features':  stdFeatures,
      'labels':    _dataset.map((s) => s.label).toList(),
      'epochs':    epochs,
      'lr':        learningRate,
      'batchSize': batchSize,
    });

    final rawW = result['weights'] as List;
    final rawB = result['biases']  as List;
    _weights = rawW.map((l) => (l as List)
        .map((row) => (row as List).map((e) => (e as num).toDouble()).toList())
        .toList()).toList();
    _biases  = rawB.map((l) => (l as List)
        .map((e) => (e as num).toDouble()).toList()).toList();

    _flRound++;
    await _saveWeights();

    return TrainingResult(
      epochs:      epochs,
      samplesUsed: _dataset.length,
      finalLoss:   (result['finalLoss'] as num).toDouble(),
      accuracy:    (result['accuracy']  as num).toDouble(),
      flRound:     _flRound,
    );
  }

  // ── Weight Export / Import  (FL Server ↔ App) ────────────────────────────
  //
  // JSON format the Flower server receives/sends:
  // {
  //   "round": 3,
  //   "architecture": [5, 64, 32, 1],
  //   "num_samples": 87,
  //   "label_distribution": {"stressed": 40, "normal": 47},
  //   "w0k": [[...5 x 64...]],   "w0b": [...64...],
  //   "w1k": [[...64 x 32...]],  "w1b": [...32...],
  //   "w2k": [[...32 x 1...]],   "w2b": [...1...]
  // }

  Map<String, dynamic> exportWeights() => {
    'round':        _flRound,
    'architecture': layerSizes,
    'num_samples':  _dataset.length,
    'label_distribution': {'stressed': stressedCount, 'normal': normalCount},
    'w0k': _weights[0],   // (64, 5)  — Dense(64) kernel transposed from TF convention
    'w0b': _biases[0],
    'w1k': _weights[1],   // (32, 64)
    'w1b': _biases[1],
    'w2k': _weights[2],   // (1, 32)
    'w2b': _biases[2],
  };

  /// Apply weights from the FL server (aggregated global model).
  void importWeights(Map<String, dynamic> json) {
    if (json.containsKey('round')) _flRound = (json['round'] as num).toInt();
    if (json.containsKey('w0k')) {
      _weightsFromFlatJson(json);
    } else if (json.containsKey('layers')) {
      // Legacy format from v1/v2 Dart trainer
      final layers = json['layers'] as List;
      _weights = layers.map((l) => (l['weights'] as List)
          .map((row) => (row as List).map((e) => (e as num).toDouble()).toList())
          .toList()).toList();
      _biases  = layers.map((l) => (l['biases'] as List)
          .map((e) => (e as num).toDouble()).toList()).toList();
    }
    _saveWeights();
  }

  void _weightsFromFlatJson(Map<String, dynamic> json) {
    _weights = [
      _parseMatrix(json['w0k']),
      _parseMatrix(json['w1k']),
      _parseMatrix(json['w2k']),
    ];
    _biases = [
      _parseVector(json['w0b']),
      _parseVector(json['w1b']),
      _parseVector(json['w2b']),
    ];
  }

  static List<List<double>> _parseMatrix(dynamic raw) =>
      (raw as List).map((row) =>
          (row as List).map((e) => (e as num).toDouble()).toList()).toList();

  static List<double> _parseVector(dynamic raw) =>
      (raw as List).map((e) => (e as num).toDouble()).toList();

  // ── Persistence ───────────────────────────────────────────────────────────

  Future<void> _loadPersisted() async {
    final prefs = await SharedPreferences.getInstance();
    _flRound = prefs.getInt(_kRound) ?? 0;

    final dJson = prefs.getString(_kDataset);
    if (dJson != null) {
      try {
        _dataset = (jsonDecode(dJson) as List)
            .map((e) => TrainingSample.fromJson(e as Map<String, dynamic>))
            .toList();
      } catch (_) {}
    }

    // Persisted fine-tuned weights override the asset pre-trained weights
    final wJson = prefs.getString(_kWeights);
    if (wJson != null) {
      try { importWeights(jsonDecode(wJson) as Map<String, dynamic>); } catch (_) {}
    }
  }

  Future<void> _saveWeights() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_kWeights, jsonEncode(exportWeights()));
    await prefs.setInt(_kRound, _flRound);
  }

  Future<void> _saveDataset() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_kDataset,
        jsonEncode(_dataset.map((s) => s.toJson()).toList()));
  }

  Future<void> clearDataset() async {
    _dataset.clear();
    _bvpBuf.clear();
    _tempNormBuf.clear();
    _lastSampleAt = null;
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_kDataset);
  }

  // ── Math helpers ─────────────────────────────────────────────────────────

  List<double> _standardize(List<double> raw) =>
      List.generate(raw.length, (i) => (raw[i] - scalerMean[i]) / scalerScale[i]);

  double _mean(List<double> v) {
    double s = 0; for (final x in v) s += x; return s / v.length;
  }

  double _std(List<double> v, double m) {
    double a = 0;
    for (final x in v) { final d = x - m; a += d * d; }
    return sqrt(a / v.length);
  }

  // ── Getters ───────────────────────────────────────────────────────────────

  int  get sampleCount    => _dataset.length;
  int  get flRound        => _flRound;
  bool get hasEnoughData  => _dataset.length >= minTrainSamples;
  int  get stressedCount  => _dataset.where((s) => s.label == 1.0).length;
  int  get normalCount    => _dataset.where((s) => s.label == 0.0).length;
  int  get windowProgress => _bvpBuf.length;
}
