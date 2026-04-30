"""Smoke test for visualize_embeddings: function runs, PNG is non-trivial."""

import pickle

import numpy as np

from src.visualize_embeddings import visualize_layers


def test_visualize_layers_smoke(tmp_path):
    n_phrases = 12  # 4 triplets
    hidden_size = 50
    n_layers = 4

    rng = np.random.RandomState(0)
    activations = {
        layer: rng.randn(n_phrases, hidden_size).astype(np.float32)
        for layer in range(n_layers)
    }
    labels = np.array(["joy", "anger", "neutral"] * 4)
    group_ids = np.array([i // 3 for i in range(n_phrases)], dtype=np.int64)

    payload = {
        "activations": activations,
        "labels": labels,
        "group_ids": group_ids,
        "texts": [f"s{i}" for i in range(n_phrases)],
        "metadata": {"model_name": "dummy", "n_phrases": n_phrases},
    }

    pkl_path = tmp_path / "act.pkl"
    png_path = tmp_path / "tsne.png"
    with open(pkl_path, "wb") as f:
        pickle.dump(payload, f)

    # perplexity=3 because perplexity must be < n_samples (= 12)
    visualize_layers(
        pkl_path, png_path, layers_to_plot=(0, 1, 2, 3), perplexity=3
    )

    assert png_path.exists()
    assert png_path.stat().st_size > 5_000
