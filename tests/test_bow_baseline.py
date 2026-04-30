"""Smoke test for train_bow_baseline on dummy 12-phrase corpus."""

import json

from src.train_bow_baseline import train_bow_baseline


DUMMY_SENTENCES = {
    "joy": [
        "She smiled brightly at the news",
        "Joy filled his heart that morning",
        "We celebrated with cheers and laughter",
        "He grinned and hugged his friend",
    ],
    "anger": [
        "He slammed the door behind him",
        "Fury rose in her chest",
        "She shouted with rage at the screen",
        "His fists clenched in pure anger",
    ],
    "neutral": [
        "The book was placed on the shelf",
        "She walked along the quiet street",
        "It is Tuesday afternoon already",
        "The meeting was scheduled for noon",
    ],
}


def test_bow_baseline_smoke(tmp_path):
    corpus = []
    for i in range(4):  # 4 triplets → 12 phrases
        for cls in ("joy", "anger", "neutral"):
            corpus.append({"text": DUMMY_SENTENCES[cls][i], "label": cls})

    corpus_path = tmp_path / "corpus.json"
    output_path = tmp_path / "bow.json"
    with open(corpus_path, "w") as f:
        json.dump(corpus, f)

    # n_splits=4 because 4 triplets → 4 groups (GroupKFold needs n_splits ≤ n_groups)
    train_bow_baseline(corpus_path, output_path, n_splits=4)

    with open(output_path) as f:
        result = json.load(f)

    assert 0.0 <= result["mean_acc"] <= 1.0
    assert isinstance(result["std_acc"], float)
    assert isinstance(result["scores_per_fold"], list)
    assert len(result["scores_per_fold"]) == 4

    expected_meta_keys = {
        "method",
        "vectorizer_params",
        "n_features_actual",
        "n_phrases",
        "n_classes",
        "label_mapping",
        "baseline_chance_level",
        "cv",
        "probe",
        "seed",
    }
    assert set(result["metadata"].keys()) == expected_meta_keys
    assert result["metadata"]["n_phrases"] == 12
    assert result["metadata"]["n_classes"] == 3
    assert set(result["metadata"]["label_mapping"].keys()) == {"anger", "joy", "neutral"}
    assert result["metadata"]["n_features_actual"] > 0
