"""
Extraction des activations pour le probing.
- Tokenisation brute, sans tokens spéciaux : on mesure la
  représentation du contenu textuel pur, sans dilution par les
  tokens conversationnels du modèle Instruct.
- Mean pooling sur tous les tokens du contenu (pas de padding
  car traitement phrase par phrase).
- Float32 explicite pour la rigueur scientifique du probing
  (cf. README).
- Phrase par phrase plutôt que batching : 132 phrases courtes,
  pas besoin d'optimiser sur M1 16GB, et ça simplifie le
  handling des shapes.
- Seed fixée à 42 pour reproductibilité.
"""

import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.model_loader import MODEL_NAME, load_model

SEED = 42
EXPECTED_LABELS = {"joy", "anger", "neutral"}
EXPECTED_N_PHRASES = 132

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)


def _validate_corpus(corpus):
    if len(corpus) != EXPECTED_N_PHRASES:
        raise ValueError(
            f"Expected {EXPECTED_N_PHRASES} sentences, got {len(corpus)}"
        )
    if len(corpus) % 3 != 0:
        raise ValueError(
            f"Corpus length {len(corpus)} not divisible by 3 — triplet structure broken"
        )
    for i in range(len(corpus) // 3):
        triplet = {corpus[3 * i + j]["label"] for j in range(3)}
        if triplet != EXPECTED_LABELS:
            raise ValueError(
                f"Triplet {i} (phrases {3*i}-{3*i+2}) does not contain "
                f"{EXPECTED_LABELS}; got {triplet}"
            )


def extract_all_activations(corpus_path, output_path):
    with open(corpus_path) as f:
        corpus = json.load(f)
    _validate_corpus(corpus)

    model, tokenizer, config = load_model()
    n_phrases = len(corpus)
    n_hidden_states = config["n_hidden_states"]
    hidden_size = config["hidden_size"]
    device = config["device"]

    activations = {
        layer: np.zeros((n_phrases, hidden_size), dtype=np.float32)
        for layer in range(n_hidden_states)
    }
    labels = np.array([entry["label"] for entry in corpus])
    texts = [entry["text"] for entry in corpus]
    group_ids = np.array([i // 3 for i in range(n_phrases)], dtype=np.int64)

    start = time.time()
    for idx, entry in enumerate(tqdm(corpus, desc="extract")):
        inputs = tokenizer(
            entry["text"], return_tensors="pt", add_special_tokens=False
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        for layer_idx, hs in enumerate(outputs.hidden_states):
            pooled = (
                hs.mean(dim=1).squeeze(0).float().cpu().numpy().astype(np.float32)
            )
            activations[layer_idx][idx] = pooled
    elapsed = time.time() - start

    assert activations[0].shape == (132, 1536)
    assert activations[0].dtype == np.float32
    assert len(labels) == 132
    assert len(set(group_ids.tolist())) == 44

    output = {
        "activations": activations,
        "labels": labels,
        "group_ids": group_ids,
        "texts": texts,
        "metadata": {
            "model_name": MODEL_NAME,
            "n_layers": config["n_layers"],
            "hidden_size": config["hidden_size"],
            "n_hidden_states": config["n_hidden_states"],
            "pooling": "mean",
            "tokenization": "raw_no_special_tokens",
            "dtype": "float32",
            "seed": SEED,
            "n_phrases": n_phrases,
        },
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(output, f)
    size_mb = output_path.stat().st_size / (1024 * 1024)

    print(f"[extract] {n_phrases} phrases traitées en {elapsed:.1f}s")
    print(f"[extract] Sauvegardé : {output_path} ({size_mb:.0f} MB)")
    print(
        f"[extract] Shape par couche : {activations[0].shape} × "
        f"{n_hidden_states} couches"
    )


if __name__ == "__main__":
    extract_all_activations("data/corpus.json", "results/activations.pkl")
