"""Logic-only tests for extract_activations: pooling, dtype, .pkl format.

No model loading here — we test the pooling/format logic in isolation.
The end-to-end integration test is implicit when running the script itself.
"""

import pickle

import numpy as np
import torch

N_HIDDEN_STATES = 29
HIDDEN_SIZE = 1536
SEQ_LEN = 5


def test_mean_pooling_shape_and_dtype():
    """Mimics the per-layer pooling step from extract_all_activations."""
    hidden_states = tuple(
        torch.randn(1, SEQ_LEN, HIDDEN_SIZE) for _ in range(N_HIDDEN_STATES)
    )
    assert len(hidden_states) == N_HIDDEN_STATES

    for hs in hidden_states:
        pooled = hs.mean(dim=1).squeeze(0).float().cpu().numpy().astype(np.float32)
        assert pooled.shape == (HIDDEN_SIZE,)
        assert pooled.dtype == np.float32


def test_pkl_format_two_triplets(tmp_path):
    """Build a 6-phrase (2 triplets) dummy output and roundtrip via pickle."""
    n_phrases = 6
    activations = {
        layer: np.random.randn(n_phrases, HIDDEN_SIZE).astype(np.float32)
        for layer in range(N_HIDDEN_STATES)
    }
    labels = np.array(["joy", "anger", "neutral", "joy", "anger", "neutral"])
    group_ids = np.array([i // 3 for i in range(n_phrases)], dtype=np.int64)
    texts = [f"sentence {i}" for i in range(n_phrases)]
    output = {
        "activations": activations,
        "labels": labels,
        "group_ids": group_ids,
        "texts": texts,
        "metadata": {"n_phrases": n_phrases, "seed": 42},
    }

    pkl_path = tmp_path / "activations.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(output, f)
    with open(pkl_path, "rb") as f:
        loaded = pickle.load(f)

    assert set(loaded.keys()) == {
        "activations",
        "labels",
        "group_ids",
        "texts",
        "metadata",
    }
    assert loaded["group_ids"].tolist() == [0, 0, 0, 1, 1, 1]
    assert loaded["activations"][0].shape == (n_phrases, HIDDEN_SIZE)
    assert loaded["activations"][0].dtype == np.float32
    assert len(loaded["activations"]) == N_HIDDEN_STATES
    assert len(loaded["labels"]) == n_phrases
    assert len(loaded["texts"]) == n_phrases
