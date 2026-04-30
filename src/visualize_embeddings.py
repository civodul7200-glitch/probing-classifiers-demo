"""
Visualisation t-SNE des activations par couche.

Choix méthodologiques :
- 4 couches sélectionnées (0, 5, 10, 28) pour raconter la
  trajectoire : embedding pur → montée → plateau optimal →
  représentation finale.
- StandardScaler avant t-SNE : t-SNE est sensible à l'échelle
  des features.
- perplexity=30 : valeur standard pour ~100-200 points.
- random_state=42 : t-SNE est stochastique, on fixe pour
  reproductibilité.
- init="pca" : initialisation déterministe et plus stable que
  l'init aléatoire.
- 2D scatter, pas 3D : lisibilité prime.
"""

import pickle
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

CLASS_COLORS = {"joy": "C2", "anger": "C3", "neutral": "C7"}
LAYER_CAPTIONS = {
    0: "(embedding)",
    5: "(montée)",
    10: "(plateau optimal)",
    28: "(finale)",
}


def visualize_layers(
    activations_path,
    output_path,
    layers_to_plot=(0, 5, 10, 28),
    perplexity=30,
):
    with open(activations_path, "rb") as f:
        data = pickle.load(f)
    activations = data["activations"]
    labels = np.asarray(data["labels"])

    projections = {}
    for layer in layers_to_plot:
        X = activations[layer]
        X_scaled = StandardScaler().fit_transform(X)
        tsne = TSNE(
            n_components=2,
            random_state=SEED,
            perplexity=perplexity,
            max_iter=1000,
            init="pca",
        )
        projections[layer] = tsne.fit_transform(X_scaled)
        print(f"[tsne] Layer {layer}: projection done")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, layer in zip(axes.flatten(), layers_to_plot):
        X_2d = projections[layer]
        for cls, color in CLASS_COLORS.items():
            mask = labels == cls
            ax.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                color=color,
                label=cls,
                s=50,
                alpha=0.7,
            )
        caption = LAYER_CAPTIONS.get(layer, "")
        ax.set_title(f"Layer {layer} {caption}".strip())
        ax.grid(True, alpha=0.3)

    # Global legend (top), suptitle above it
    handles, lbls = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=3)
    fig.suptitle(
        "t-SNE projection of Qwen2.5-1.5B activations by layer\n"
        "Emotion classification (n=132)"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[tsne] sauvegardé : {output_path}")


if __name__ == "__main__":
    visualize_layers("results/activations.pkl", "results/tsne_by_layer.png")
