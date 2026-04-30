"""Logic tests for train_probes on dummy data (no real model loading)."""

import json
import pickle

import numpy as np

from src.train_probes import train_all_probes


def _make_dummy_data(separable):
    """30 phrases, 10 triplets, 2 layers, 50 dims, 3 balanced classes.

    If `separable=True`, inject a class-aligned signal so the probe should
    achieve high accuracy. Otherwise return pure random features for chance.
    """
    n_phrases = 30
    n_triplets = 10
    hidden_size = 50
    n_layers = 2

    rng = np.random.RandomState(0)
    labels = np.array(["joy", "anger", "neutral"] * n_triplets)
    group_ids = np.array([i // 3 for i in range(n_phrases)], dtype=np.int64)
    texts = [f"sentence {i}" for i in range(n_phrases)]

    activations = {
        layer: rng.randn(n_phrases, hidden_size).astype(np.float32) * 0.5
        for layer in range(n_layers)
    }

    if separable:
        # Distribute signal over 10 dims per class (more realistic than one
        # hot-dim, and gives the probe enough redundancy to overcome C=0.1).
        label_to_dims = {
            "anger": range(0, 10),
            "joy": range(10, 20),
            "neutral": range(20, 30),
        }
        for layer in range(n_layers):
            for i, lab in enumerate(labels):
                for d in label_to_dims[lab]:
                    activations[layer][i, d] += 3.0

    return {
        "activations": activations,
        "labels": labels,
        "group_ids": group_ids,
        "texts": texts,
        "metadata": {"model_name": "dummy-model"},
    }, n_layers


def _run(tmp_path, payload):
    pkl = tmp_path / "act.pkl"
    js = tmp_path / "probe.json"
    with open(pkl, "wb") as f:
        pickle.dump(payload, f)
    train_all_probes(pkl, js)
    with open(js) as f:
        return json.load(f)


def test_json_format(tmp_path):
    payload, n_layers = _make_dummy_data(separable=False)
    results = _run(tmp_path, payload)

    assert set(results.keys()) == {"results_per_layer", "metadata"}
    assert len(results["results_per_layer"]) == n_layers

    for layer in range(n_layers):
        r = results["results_per_layer"][str(layer)]
        assert set(r.keys()) == {"mean_acc", "std_acc", "scores_per_fold"}
        assert isinstance(r["mean_acc"], float)
        assert isinstance(r["std_acc"], float)
        assert isinstance(r["scores_per_fold"], list)
        assert len(r["scores_per_fold"]) == 5
        assert 0.0 <= r["mean_acc"] <= 1.0
        assert all(0.0 <= s <= 1.0 for s in r["scores_per_fold"])

    meta = results["metadata"]
    assert meta["n_phrases"] == 30
    assert meta["n_classes"] == 3
    assert set(meta["label_mapping"].keys()) == {"anger", "joy", "neutral"}
    assert abs(meta["baseline_chance_level"] - 1 / 3) < 1e-9
    assert meta["seed"] == 42


def test_separable_features_high_accuracy(tmp_path):
    payload, n_layers = _make_dummy_data(separable=True)
    results = _run(tmp_path, payload)
    for layer in range(n_layers):
        mean_acc = results["results_per_layer"][str(layer)]["mean_acc"]
        assert mean_acc > 0.9, (
            f"Separable features should give acc > 0.9 on layer {layer}, "
            f"got {mean_acc:.3f}"
        )


def test_random_features_chance_accuracy(tmp_path):
    payload, n_layers = _make_dummy_data(separable=False)
    results = _run(tmp_path, payload)
    for layer in range(n_layers):
        mean_acc = results["results_per_layer"][str(layer)]["mean_acc"]
        # 3-class chance = 0.333; allow generous slack for small-sample variance
        assert mean_acc < 0.55, (
            f"Random features should be near chance on layer {layer}, "
            f"got {mean_acc:.3f}"
        )
