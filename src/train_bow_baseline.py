"""
Baseline TF-IDF pour comparaison avec le probing des activations.

Choix méthodologiques :
- TF-IDF + LogisticRegression : modèle classique de classification
  de texte, sans aucune sémantique apprise — chaque mot est un
  symbole indépendant.
- Mêmes hyperparamètres que le probe Qwen (C=0.1, GroupKFold par
  triplet, 5 plis) : seule la source des features change.
- ngram_range=(1, 2) : permet de capturer les bigrammes
  caractéristiques ("broke into", "slammed onto").
- StandardScaler(with_mean=False) : garde la sparsité de la
  matrice TF-IDF (centering casserait la sparsité).
"""

import json
import random
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

VECTORIZER_PARAMS = {
    "lowercase": True,
    "stop_words": None,
    "ngram_range": (1, 2),
    "max_features": 5000,
    "min_df": 1,
}


def _build_pipeline():
    return Pipeline(
        [
            ("vec", TfidfVectorizer(**VECTORIZER_PARAMS)),
            ("scaler", StandardScaler(with_mean=False)),
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


def train_bow_baseline(corpus_path, output_path, n_splits=5):
    with open(corpus_path) as f:
        corpus = json.load(f)

    texts = [entry["text"] for entry in corpus]
    labels_str = [entry["label"] for entry in corpus]
    n_phrases = len(corpus)

    le = LabelEncoder()
    y = le.fit_transform(labels_str)
    label_mapping = {str(c): int(i) for i, c in enumerate(le.classes_)}
    n_classes = len(le.classes_)
    chance = 1.0 / n_classes

    group_ids = np.array([i // 3 for i in range(n_phrases)], dtype=np.int64)

    # Effective vocab size on full corpus (descriptive only — CV refits per fold)
    full_vec = TfidfVectorizer(**VECTORIZER_PARAMS).fit(texts)
    n_features_actual = len(full_vec.vocabulary_)

    cv = GroupKFold(n_splits=n_splits)
    scores = cross_val_score(
        _build_pipeline(),
        texts,
        y,
        groups=group_ids,
        cv=cv,
        scoring="accuracy",
    )
    mean_acc = float(scores.mean())
    std_acc = float(scores.std())

    output = {
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "scores_per_fold": [float(s) for s in scores],
        "metadata": {
            "method": "TF-IDF + LogisticRegression",
            "vectorizer_params": {
                "ngram_range": list(VECTORIZER_PARAMS["ngram_range"]),
                "lowercase": VECTORIZER_PARAMS["lowercase"],
                "max_features": VECTORIZER_PARAMS["max_features"],
                "min_df": VECTORIZER_PARAMS["min_df"],
                "stop_words": VECTORIZER_PARAMS["stop_words"],
            },
            "n_features_actual": n_features_actual,
            "n_phrases": n_phrases,
            "n_classes": n_classes,
            "label_mapping": label_mapping,
            "baseline_chance_level": chance,
            "cv": f"GroupKFold(n_splits={n_splits}), groups=triplets",
            "probe": "Pipeline(StandardScaler(with_mean=False), LogisticRegression(C=0.1))",
            "seed": SEED,
        },
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[bow] vocab size: {n_features_actual}")
    print(f"[bow] mean_acc: {mean_acc:.3f} ± {std_acc:.3f}")
    print(f"[bow] scores per fold: {[f'{s:.2f}' for s in scores]}")
    print(f"[bow] vs chance ({chance:.3f}): écart de {mean_acc - chance:+.3f}")
    print(f"[bow] sauvegardé : {output_path}")


if __name__ == "__main__":
    train_bow_baseline("data/corpus.json", "results/bow_baseline.json")
