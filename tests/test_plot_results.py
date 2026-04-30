"""Smoke test: plot_probe_accuracies runs end-to-end on a dummy JSON."""

import json

from src.plot_results import plot_probe_accuracies


def test_plot_runs_and_writes_png(tmp_path):
    dummy = {
        "results_per_layer": {
            str(i): {
                "mean_acc": 0.4 + 0.02 * i,
                "std_acc": 0.05,
                "scores_per_fold": [0.4 + 0.02 * i] * 5,
            }
            for i in range(29)
        },
        "metadata": {
            "model_name": "dummy",
            "n_phrases": 132,
            "n_classes": 3,
            "baseline_chance_level": 1 / 3,
        },
    }
    json_path = tmp_path / "probe.json"
    png_path = tmp_path / "plot.png"
    with open(json_path, "w") as f:
        json.dump(dummy, f)

    plot_probe_accuracies(json_path, png_path)

    assert png_path.exists()
    assert png_path.stat().st_size > 1000  # non-trivial PNG
