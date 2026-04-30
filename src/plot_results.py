"""
Visualisation de la trajectoire d'accuracy des probes par couche.

Deux panneaux côte à côte :
- gauche : vue globale (y ∈ [0, 1.05]), pour situer la trajectoire
  par rapport à la chance et à la baseline lexicale.
- droite : zoom dynamique (y ∈ [0.85, 1.0]), pour lire la
  progression fine entre couches là où elle se joue. La chance
  level (1/3) est hors-cadre, c'est attendu.

Lignes horizontales communes :
- chance level (1/3) en pointillés gris
- baseline lexicale (acc de la couche 0, embedding seul) en
  pointillés rouges — repère de ce qui est décodable AVANT tout
  traitement Transformer.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _draw(ax, layers, means, stds, chance, lexical_baseline, n_classes):
    ax.plot(layers, means, marker="o", color="C0", label="probe accuracy (mean)")
    ax.fill_between(
        layers,
        means - stds,
        means + stds,
        color="C0",
        alpha=0.2,
        label="±1 std (5-fold)",
    )
    ax.axhline(
        chance,
        linestyle=":",
        color="grey",
        label=f"chance level (1/{n_classes} = {chance:.3f})",
    )
    ax.axhline(
        lexical_baseline,
        linestyle="--",
        color="C3",
        alpha=0.7,
        label=f"lexical baseline = layer 0 ({lexical_baseline:.3f})",
    )
    ax.set_xlabel("Layer (0 = embedding, 1–28 = Transformer blocks)")
    ax.set_xticks(layers[::2])
    ax.grid(True, alpha=0.3)


def plot_probe_accuracies(json_path, output_path):
    with open(json_path) as f:
        data = json.load(f)

    results = data["results_per_layer"]
    meta = data["metadata"]

    layers = sorted(int(k) for k in results.keys())
    means = np.array([results[str(l)]["mean_acc"] for l in layers])
    stds = np.array([results[str(l)]["std_acc"] for l in layers])

    chance = meta["baseline_chance_level"]
    lexical_baseline = results["0"]["mean_acc"]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))

    _draw(ax_left, layers, means, stds, chance, lexical_baseline, meta["n_classes"])
    ax_left.set_ylim(0.0, 1.05)
    ax_left.set_ylabel("Accuracy (5-fold GroupKFold by triplet)")
    ax_left.set_title("Vue globale (0 à 1.05)")
    ax_left.legend(loc="lower right")

    _draw(ax_right, layers, means, stds, chance, lexical_baseline, meta["n_classes"])
    ax_right.set_ylim(0.85, 1.0)
    ax_right.set_title("Zoom dynamique fine (0.85 à 1.0)")
    ax_right.text(
        0.98,
        0.02,
        f"chance level ({chance:.3f}) hors-cadre",
        transform=ax_right.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="grey",
        style="italic",
    )

    fig.suptitle(
        f"Linear probe accuracy by layer — {meta['model_name']}\n"
        f"Emotion (joy/anger/neutral), n={meta['n_phrases']}"
    )
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"[plot] sauvegardé : {output_path}")
    print(
        f"[plot] best layer = {layers[int(np.argmax(means))]} "
        f"(acc = {means.max():.3f})"
    )
    print(f"[plot] lexical baseline (layer 0) = {lexical_baseline:.3f}")
    print(
        f"[plot] gain Transformer (max - layer0) = "
        f"{means.max() - lexical_baseline:+.3f}"
    )


if __name__ == "__main__":
    plot_probe_accuracies(
        "results/probe_accuracies.json", "results/probe_accuracies.png"
    )
