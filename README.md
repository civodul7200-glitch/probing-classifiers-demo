# Probing Classifiers — Qwen2.5-1.5B-Instruct

Layer-by-layer probing of emotional representations in Qwen2.5-1.5B-Instruct,
with TF-IDF and control task baselines to measure what each layer contributes.

---

## Key findings

1. **Emotional information is already linearly decodable at the embedding layer.**
   Layer 0 (the token embedding, before any Transformer processing) achieves
   90.9% accuracy — far above chance (33.3%). The emotional signal is present
   in the model's vocabulary representations.

2. **Qwen's embedding encodes genuine distributional semantics beyond the lexicon.**
   A TF-IDF bag-of-words baseline on the same sentences achieves 72.2% (±8.4).
   The embedding outperforms it by +18.7 points (0.909 vs 0.722), showing that
   the model has learned semantic proximity between words (e.g. "grin" ≈ "smile")
   independently of their lexical identity.

3. **Transformer layers add measurable contextual composition.**
   Probe accuracy rises from 90.9% at layer 0 to a plateau of 98.4% from
   layer ~8–10 onward (+7.5 points over the embedding). The gain is monotonic
   across the first 10 layers and then flat to the final layer 28.

4. **The signal is real, not an overfitting artifact.**
   A control task (labels shuffled randomly within each triplet, preserving
   class balance) stays at chance across all 29 layers: mean = 34.6%,
   +1.2 points above chance, with no depth trend. The gap between the best
   probe layer and the control task mean is +63.8 points (0.984 vs 0.346).

5. **Complete decomposition of the emotional signal:**

   | Source | Accuracy | Gain over previous |
   |---|---|---|
   | Chance level | 33.3% | — |
   | TF-IDF (bag-of-words) | 72.2% | +38.9 pts (lexical signal) |
   | Qwen embedding (layer 0) | 90.9% | +18.7 pts (distributional semantics) |
   | Qwen layer 10 (plateau) | 98.4% | +7.5 pts (contextual composition) |

---

## How it works

**Step 1 — Corpus**

132 English sentences, 44 per class (joy / anger / neutral), structured as 44
contrastive triplets. Each triplet shares an identical prefix across the three
classes:

> *"She opened the letter and..."* → joy / anger / neutral

The triplet structure prevents the probe from learning prefix-level shortcuts:
all three sentences of a triplet are always in the same cross-validation fold.

**Step 2 — Activation extraction**

Each sentence is passed through Qwen2.5-1.5B-Instruct with
`output_hidden_states=True`, no special tokens. The hidden states of all 29
layers (1 embedding + 28 Transformer blocks) are extracted and mean-pooled
across token positions. Activations are cached to `results/activations.pkl`
(~22 MB, float32).

**Step 3 — Probing per layer**

For each of the 29 layers: a `Pipeline(StandardScaler, LogisticRegression(C=0.1))`
is evaluated with `GroupKFold(n_splits=5)`, grouped by triplet. Mean and std
accuracy across 5 folds is recorded. Results saved to
`results/probe_accuracies.json`.

**Step 4 — Comparison baselines**

Two baselines quantify what the probe is actually measuring:
- **TF-IDF** (bag-of-words + LogReg, same hyperparameters): measures lexical
  signal alone.
- **Control task** (same probe, shuffled labels within triplets): verifies that
  observed accuracies reflect real signal, not probe capacity.

---

## Results

`results/probe_accuracies.png` — Two-panel plot: global view (y ∈ [0, 1.05])
and zoomed view (y ∈ [0.85, 1.0]) showing probe accuracy by layer with ±1 std
bands and reference lines for chance and the lexical baseline.

`results/tsne_by_layer.png` — 2×2 t-SNE projections (perplexity=30, init="pca")
at layers 0, 5, 10, and 28. The class separation emerging between layers 0 and
10 is visible directly.

`results/comparison_qwen_vs_tfidf.png` — Single figure overlaying the probe
trajectory with horizontal bands for the TF-IDF baseline, lexical baseline
(layer 0), control task, and chance level.

---

## Repository layout

```
probing-classifiers-demo/
│
├── src/
│   ├── model_loader.py              # load_model() → (model, tokenizer, config); MPS/CPU autodetect
│   ├── extract_activations.py       # extract_all_activations(): 29-layer hidden states → activations.pkl
│   ├── train_probes.py              # train_all_probes(): GroupKFold probing per layer → probe_accuracies.json
│   ├── train_bow_baseline.py        # train_bow_baseline(): TF-IDF + LogReg baseline → bow_baseline.json
│   ├── train_control_task.py        # train_control_task(): shuffled-label probe → control_task_accuracies.json
│   ├── plot_results.py              # plot_probe_accuracies(): two-panel accuracy plot → probe_accuracies.png
│   ├── visualize_embeddings.py      # visualize_layers(): t-SNE 2×2 grid → tsne_by_layer.png
│   └── compare_baselines.py         # main(): probe vs TF-IDF vs control task → comparison_qwen_vs_tfidf.png
│
├── tests/
│   ├── test_model_loader.py         # model loads, 28 layers, hidden_size 1536, forward pass shapes
│   ├── test_extract_activations.py  # mean pooling shape/dtype, pkl format roundtrip
│   ├── test_train_probes.py         # probe pipeline on dummy data, separable vs random features
│   ├── test_bow_baseline.py         # TF-IDF pipeline on 12-phrase dummy, JSON format
│   ├── test_plot_results.py         # smoke: plot runs, PNG is non-trivial
│   └── test_visualize_embeddings.py # smoke: t-SNE runs on small dummy, PNG written
│
├── data/
│   └── corpus.json                  # 132 sentences, {text, label}, 44 triplets
│
├── results/                         # generated outputs (not versioned)
│   ├── activations.pkl              # 29-layer activations, float32, ~22 MB
│   ├── probe_accuracies.json        # mean/std/per-fold accuracy, all 29 layers
│   ├── bow_baseline.json            # TF-IDF baseline accuracy
│   ├── control_task_accuracies.json # shuffled-label probe, all 29 layers
│   ├── probe_accuracies.png
│   ├── tsne_by_layer.png
│   └── comparison_qwen_vs_tfidf.png
│
├── notebooks/                       # exploration (empty at MVP)
├── conftest.py                      # adds project root to sys.path for test imports
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Qwen2.5-1.5B-Instruct (~3 GB) is downloaded automatically from HuggingFace
on first run of any script that calls `load_model()`.

---

## Running the pipeline

Run in order to reproduce all results from scratch:

```bash
python -m src.extract_activations    # ~30s on M1 MPS — produces results/activations.pkl
python -m src.train_probes           # ~3s  — produces results/probe_accuracies.json
python -m src.train_bow_baseline     # <1s  — produces results/bow_baseline.json
python -m src.train_control_task     # ~3s  — produces results/control_task_accuracies.json
python -m src.plot_results           # <1s  — produces results/probe_accuracies.png
python -m src.visualize_embeddings   # ~30s — produces results/tsne_by_layer.png
python -m src.compare_baselines      # <1s  — produces results/comparison_qwen_vs_tfidf.png
```

Steps 2–7 do not reload the model. `activations.pkl` is cached: re-running
`train_probes`, `train_bow_baseline`, and `train_control_task` takes seconds.

---

## Tests

```bash
pytest tests/ -v
```

Six test modules, all passing at project completion:

- `test_model_loader.py` — model loads on MPS, `n_layers == 28`,
  `hidden_size == 1536`, `output_hidden_states` returns 29 tensors of shape
  `(1, seq_len, 1536)`, wrapped in `torch.no_grad()`.
- `test_extract_activations.py` — mean pooling produces shape `(1536,)` in
  float32; pkl roundtrip preserves all keys and `group_ids == [0,0,0,1,1,1]`
  for a 6-phrase dummy.
- `test_train_probes.py` — probe pipeline on 30-phrase dummy: separable
  features give acc > 0.9; random features give acc < 0.55; JSON format
  and metadata keys verified.
- `test_bow_baseline.py` — TF-IDF pipeline on 12-phrase dummy; JSON format
  and all metadata keys verified; `mean_acc ∈ [0, 1]`.
- `test_plot_results.py` — smoke: `plot_probe_accuracies()` runs on a
  29-layer dummy JSON and writes a PNG > 1 KB.
- `test_visualize_embeddings.py` — smoke: `visualize_layers()` runs on a
  12-phrase dummy with `perplexity=3` and writes a PNG > 5 KB.

---

## Methodological choices

**GroupKFold grouped by triplet, not by sentence.** Each triplet shares an
identical prefix (e.g. "She opened the letter and...") across its three
sentences. Random splitting would allow the joy and anger versions of the same
prefix to appear in different folds, letting the probe learn the prefix rather
than the emotional continuation. Grouping by triplet guarantees that all three
sentences sharing a prefix are always in the same fold. The probe must
generalize to unseen prefixes.

**Linear probe (LogisticRegression), not a MLP.** The goal is to measure
*linear decodability*: whether emotional information is encoded in a direction
that can be recovered without non-linear transformation. A MLP probe would
measure the probe's capacity more than the representation's quality. A linear
probe with strong regularization is a conservative lower bound on what is
encoded in each layer.

**C=0.1 (strong L2 regularization).** Each training fold contains ~106
sentences in 1536 dimensions — a severely under-determined regime. Without
strong regularization, the probe overfits to spurious correlations. The control
task validates that C=0.1 is conservative enough: with shuffled labels, the
probe cannot exceed chance.

**StandardScaler inside the Pipeline, not applied globally.** Fitting the
scaler on the full dataset before cross-validation would leak test distribution
statistics into training. The Pipeline ensures the scaler is re-fitted on each
training fold independently. This also matters for LogisticRegression: L2
regularization is scale-sensitive, and unscaled 1536-dimensional activations
have highly variable feature magnitudes across layers.

**Tokenization without special tokens** (`add_special_tokens=False`). Qwen
Instruct wraps inputs in conversation tokens (`<|im_start|>`, etc.). Including
them in the mean pool would dilute the emotional signal with tokens that carry
no emotional content. The pooled representation targets the sentence content
only.

**Control task with within-triplet label shuffling.** Shuffling labels globally
would change the class distribution per fold and make the comparison unfair.
Shuffling within each triplet preserves the 1:1:1 class balance in every fold
while destroying the alignment between activations and labels. A probe that
cannot exceed chance on this task has genuinely insufficient capacity to
memorize the training signal — ruling out overfitting as an explanation for the
observed 90–98% accuracies.

---

## Known limitations

- **Small, domain-specific corpus.** 132 sentences, narrative register, English
  only. Generalization to conversational, social-media, or non-English text is
  not established.
- **Mean pooling discards positional structure.** The pooled representation
  averages over all token positions. Information localized to specific positions
  (e.g. the emotional verb at the end of the sentence) is diluted.
- **Linear probe underestimates total information.** A non-linear probe (MLP,
  k-NN) would likely recover additional signal. The reported accuracies are
  a lower bound on what is encoded, not an upper bound.
- **Three discrete classes vs. a continuous emotion space.** Joy, anger, and
  neutral are categorical labels. Real emotional representations may be better
  described by valence × arousal dimensions; categorical accuracy does not
  capture this structure.
- **Single model.** Results are specific to Qwen2.5-1.5B-Instruct. Layer
  numbers, accuracy levels, and the depth of the plateau may differ
  substantially on other architectures or parameter scales.
