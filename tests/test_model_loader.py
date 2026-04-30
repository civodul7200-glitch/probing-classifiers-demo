"""Smoke test for model_loader: shapes, layer count, forward pass."""

import torch

from src.model_loader import load_model

TEST_SENTENCE = "She opened the letter and set it aside."


def test_load_model_and_forward():
    model, tokenizer, config = load_model()

    assert config["n_layers"] == 28, f"expected 28 layers, got {config['n_layers']}"
    assert config["hidden_size"] == 1536, (
        f"expected hidden_size 1536, got {config['hidden_size']}"
    )
    assert config["n_hidden_states"] == 29, (
        f"expected 29 hidden states (28 layers + embedding), "
        f"got {config['n_hidden_states']}"
    )

    inputs = tokenizer(TEST_SENTENCE, return_tensors="pt").to(config["device"])
    seq_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    assert outputs.hidden_states is not None, "hidden_states is None"
    assert len(outputs.hidden_states) == 29, (
        f"expected 29 hidden states (embedding + 28 layers), "
        f"got {len(outputs.hidden_states)}"
    )

    expected_shape = (1, seq_len, 1536)
    for i, hs in enumerate(outputs.hidden_states):
        assert tuple(hs.shape) == expected_shape, (
            f"hidden_states[{i}]: expected {expected_shape}, got {tuple(hs.shape)}"
        )


if __name__ == "__main__":
    test_load_model_and_forward()
    print("OK")
