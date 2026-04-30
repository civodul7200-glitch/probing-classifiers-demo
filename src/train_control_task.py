"""
Control task : probing avec labels permutés au sein des triplets.

Méthodologie (cf. Hewitt & Liang, "Designing and Interpreting Probes
with Control Tasks", 2019) :
- Permutation des labels EN PRÉSERVANT la structure des triplets :
  chaque triplet contient toujours {joy, anger, neutral}, seul le
  mapping phrase→label est mélangé au sein du triplet.
- Le probe utilise EXACTEMENT le même pipeline que le probe réel
  (importé depuis src.train_probes) : seule la cible y change.
- Si le probe à 0.91-0.98 reflète un vrai signal, le control task
  doit donner ~0.333 (chance) sur toutes les couches.
- Si le control task remontait au-dessus de la chance, ça
  signifierait que le probe a la capacité d'apprendre des labels
  arbitraires — donc qu'il overfite, et que les hautes accuracies
  observées ne reflètent pas un vrai signal.
"""

import json
import pickle
import random
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from src.train_probes import _build_pipeline

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def _shuffle_labels_within_triplets(labels, group_ids, rng):
    """Permute the 3 labels within each triplet, preserving the {joy, anger,
    neutral} composition of every triplet but breaking phrase→label alignment.
    """
    new_labels = labels.copy()
    for g in np.unique(group_ids):
        idx = np.where(group_ids == g)[0]
        new_labels[idx] = rng.permutation(labels[idx])
    return new_labels


def train_control_task(activations_path, output_path):
    with open(activations_path, "rb") as f:
        data = pickle.load(f)

    activations = data["activations"]
    labels = np.asarray(data["labels"])
    group_ids = np.asarray(data["group_ids"])
    source_meta = data.get("metadata", {})

    rng = np.random.RandomState(SEED)
    shuffled = _shuffle_labels_within_triplets(labels, group_ids, rng)

    le = LabelEncoder()
    y = le.fit_transform(shuffled)
    label_mapping = {str(c): int(i) for i, c in enumerate(le.classes_)}

    n_phrases = len(y)
    n_classes = len(le.classes_)
    n_layers_probed = len(activations)
    chance = 1.0 / n_classes
    cv = GroupKFold(n_splits=5)

    layer_keys = sorted(activations.keys(), key=int)

    results_per_layer = {}
    for layer in tqdm(layer_keys, desc="control"):
        X = activations[layer]
        scores = cross_val_score(
            _build_pipeline(),
            X,
            y,
            groups=group_ids,
            cv=cv,
            scoring="accuracy",
        )
        results_per_layer[str(layer)] = {
            "mean_acc": float(scores.mean()),
            "std_acc": float(scores.std()),
            "scores_per_fold": [float(s) for s in scores],
        }

    output = {
        "results_per_layer": results_per_layer,
        "metadata": {
            "model_name": source_meta.get("model_name", "unknown"),
            "method": "control_task_shuffled_labels",
            "shuffle_strategy": "within_triplets",
            "n_layers_probed": n_layers_probed,
            "n_phrases": n_phrases,
            "n_classes": n_classes,
            "label_mapping": label_mapping,
            "baseline_chance_level": chance,
            "cv": "GroupKFold(n_splits=5), groups=triplets",
            "probe": "Pipeline(StandardScaler, LogisticRegression(C=0.1, multinomial))",
            "seed": SEED,
        },
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    sorted_results = sorted(results_per_layer.items(), key=lambda kv: int(kv[0]))
    print()
    for layer_str, r in sorted_results:
        print(
            f"[control] couche {int(layer_str):>2}: "
            f"acc = {r['mean_acc']:.3f} ± {r['std_acc']:.3f}"
        )

    overall_mean = float(
        np.mean([r["mean_acc"] for r in results_per_layer.values()])
    )
    print()
    print(f"[control] mean global (toutes couches) : {overall_mean:.3f}")
    print(f"[control] chance level (1/{n_classes}) = {chance:.3f}")
    print(f"[control] écart vs chance              : {overall_mean - chance:+.3f}")
    print(f"[control] sauvegardé : {output_path}")


if __name__ == "__main__":
    train_control_task(
        "results/activations.pkl", "results/control_task_accuracies.json"
    )
