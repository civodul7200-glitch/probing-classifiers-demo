"""
Comparaison visuelle : probing par couche Qwen vs baseline TF-IDF.

Ce script consomme les deux JSON produits en amont :
- results/probe_accuracies.json (probing par couche)
- results/bow_baseline.json (baseline TF-IDF)

Il produit une figure unique avec la trajectoire du probe par couche
et les deux baselines (TF-IDF, lexicale=layer 0) en lignes
horizontales annotées.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main(
    probe_path="results/probe_accuracies.json",
    bow_path="results/bow_baseline.json",
    control_path="results/control_task_accuracies.json",
    output_path="results/comparison_qwen_vs_tfidf.png",
):
    with open(probe_path) as f:
        probe_data = json.load(f)
    with open(bow_path) as f:
        bow_data = json.load(f)

    results = probe_data["results_per_layer"]
    meta = probe_data["metadata"]

    layers = sorted(int(k) for k in results.keys())
    means = np.array([results[str(l)]["mean_acc"] for l in layers])
    stds = np.array([results[str(l)]["std_acc"] for l in layers])

    chance = meta["baseline_chance_level"]
    lexical_baseline = results["0"]["mean_acc"]

    tfidf_mean = bow_data["mean_acc"]
    tfidf_std = bow_data["std_acc"]

    control_means = control_stds = None
    control_path = Path(control_path)
    if control_path.exists():
        with open(control_path) as f:
            control_data = json.load(f)
        ctrl = control_data["results_per_layer"]
        control_means = np.array([ctrl[str(l)]["mean_acc"] for l in layers])
        control_stds = np.array([ctrl[str(l)]["std_acc"] for l in layers])

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(layers, means, marker="o", color="C0", label="probe accuracy (mean)")
    ax.fill_between(
        layers, means - stds, means + stds,
        color="C0", alpha=0.2, label="±1 std (5-fold)",
    )

    if control_means is not None:
        ax.plot(
            layers, control_means, marker="x", color="C2", alpha=0.6,
            label="control task (labels shuffled within triplets)",
        )
        ax.fill_between(
            layers, control_means - control_stds, control_means + control_stds,
            color="C2", alpha=0.10,
        )

    ax.axhline(
        lexical_baseline, linestyle="--", color="C3", alpha=0.7,
        label=f"lexical baseline = layer 0 ({lexical_baseline:.3f})",
    )

    ax.axhspan(
        tfidf_mean - tfidf_std, tfidf_mean + tfidf_std,
        color="navy", alpha=0.10, label="±1 std TF-IDF",
    )
    ax.axhline(
        tfidf_mean, linestyle="--", color="navy", alpha=0.85,
        label=f"TF-IDF + LogReg baseline ({tfidf_mean:.3f} ± {tfidf_std:.3f})",
    )

    ax.axhline(
        chance, linestyle=":", color="grey",
        label=f"chance level (1/{meta['n_classes']} = {chance:.3f})",
    )

    ax.set_xlabel("Layer (0 = embedding, 1–28 = Transformer blocks)")
    ax.set_ylabel("Accuracy (5-fold GroupKFold by triplet)")
    ax.set_title(
        "Linear probe by layer vs TF-IDF baseline\n"
        f"{meta.get('model_name', 'Qwen2.5-1.5B-Instruct')}, "
        f"emotion (n={meta.get('n_phrases', 132)})"
    )
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(layers[::2])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    best_layer = layers[int(np.argmax(means))]
    best_acc = float(means.max())

    print(f"[compare] Qwen layer 0 (embedding): {lexical_baseline:.3f}")
    print(f"[compare] TF-IDF baseline         : {tfidf_mean:.3f}")
    print(f"[compare] Différence              : {lexical_baseline - tfidf_mean:+.3f}")
    print(f"[compare] Qwen best layer ({best_layer:>2})    : {best_acc:.3f}")
    print(f"[compare] Gain Qwen vs TF-IDF     : {best_acc - tfidf_mean:+.3f}")

    if control_means is not None:
        ctrl_overall = float(control_means.mean())
        gap_best = best_acc - control_means[layers.index(best_layer)]
        print(f"[compare] Control task mean (toutes couches)  : {ctrl_overall:.3f}")
        print(f"[compare] Écart probe vs control (best layer) : {gap_best:+.3f}")

    print(f"[compare] sauvegardé : {output_path}")


if __name__ == "__main__":
    main()
