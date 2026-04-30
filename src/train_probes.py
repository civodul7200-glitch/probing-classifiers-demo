"""
Probing par couche sur les activations extraites.

Choix méthodologiques :
- StandardScaler dans la Pipeline : standardisation par fold
  (pas de leakage), critique car la régularisation L2 est
  sensible à l'échelle des features.
- C=0.1 : régularisation forte. 1536 dimensions pour ~106
  phrases d'entraînement par fold, on est en régime sous-déterminé.
  Sans régularisation forte, le probe overfite et masque le vrai
  signal de la couche.
- GroupKFold par triplet : évite le leakage par similarité de
  préfixe (deux phrases d'un même triplet partagent les premiers
  mots).
- LogisticRegression linéaire (volontairement faible) : on
  mesure la décodabilité LINÉAIRE de l'information émotionnelle.
  Un MLP probe trouverait probablement plus, mais mesurerait la
  capacité du probe, pas la qualité de la représentation.
- Multinomial (softmax) plutôt que one-vs-rest : avec solver='lbfgs'
  et ≥3 classes, sklearn fait du multinomial par défaut depuis 1.5
  (le paramètre `multi_class` a été retiré en 1.8). Comportement
  identique à l'intention.
"""

import json
import pickle
import random
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def _build_pipeline():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=0.1,
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=SEED,
                ),
            ),
        ]
    )


def train_all_probes(activations_path, output_path):
    with open(activations_path, "rb") as f:
        data = pickle.load(f)

    activations = data["activations"]
    labels_str = data["labels"]
    group_ids = data["group_ids"]
    source_meta = data.get("metadata", {})

    le = LabelEncoder()
    y = le.fit_transform(labels_str)
    label_mapping = {str(c): int(i) for i, c in enumerate(le.classes_)}

    n_phrases = len(y)
    n_classes = len(le.classes_)
    n_layers_probed = len(activations)
    chance = 1.0 / n_classes
    cv = GroupKFold(n_splits=5)

    results_per_layer = {}
    layer_keys = sorted(activations.keys(), key=int) if all(
        isinstance(k, (int, np.integer)) for k in activations.keys()
    ) else sorted(activations.keys())

    for layer in tqdm(layer_keys, desc="probe"):
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

    assert len(results_per_layer) == n_layers_probed
    for layer_str, r in results_per_layer.items():
        assert 0.0 <= r["mean_acc"] <= 1.0, (
            f"Invalid mean_acc {r['mean_acc']} for layer {layer_str}"
        )

    layer0_acc = results_per_layer.get("0", {}).get("mean_acc")
    if layer0_acc is not None and layer0_acc > 0.6:
        print(
            f"[probe] WARNING: couche 0 mean_acc = {layer0_acc:.3f} > 0.6 — "
            f"possible shortcut lexical (l'embedding seul décode déjà)."
        )

    output = {
        "results_per_layer": results_per_layer,
        "metadata": {
            "model_name": source_meta.get("model_name", "unknown"),
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
            f"[probe] couche {int(layer_str):>2}: "
            f"acc = {r['mean_acc']:.3f} ± {r['std_acc']:.3f}"
        )

    best = max(sorted_results, key=lambda kv: kv[1]["mean_acc"])
    worst = min(sorted_results, key=lambda kv: kv[1]["mean_acc"])
    print()
    print(
        f"[probe] best  : couche {best[0]:>2}  "
        f"(acc = {best[1]['mean_acc']:.3f}, "
        f"{best[1]['mean_acc'] - chance:+.3f} vs chance)"
    )
    print(
        f"[probe] worst : couche {worst[0]:>2}  "
        f"(acc = {worst[1]['mean_acc']:.3f}, "
        f"{worst[1]['mean_acc'] - chance:+.3f} vs chance)"
    )
    print(f"[probe] chance level (1/{n_classes}) = {chance:.3f}")
    print(f"[probe] sauvegardé : {output_path}")


if __name__ == "__main__":
    train_all_probes(
        "results/activations.pkl", "results/probe_accuracies.json"
    )
