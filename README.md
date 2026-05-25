# Noise-BERT: Robust NLP Defenses

This repository provides tools and models to build and evaluate NLP models robust to adversarial attacks, specifically character-level typos (TextBugger style).

## Architecture

The project has been refactored to use [Hydra](https://hydra.cc) for reproducible configurations.
*   **`config/`**: Contains all hyperparameter definitions.
    *   `config.yaml`: The main entry point.
    *   `models/`: Defines model architectures (e.g., `denoising_robust.yaml`).
    *   `tasks/`: Defines tasks, data loading, augmentation settings, and evaluation parameters (e.g., `sst2_textbugger.yaml`, `eval_sst2_textbugger.yaml`).
        *   **`augmentation_type`**: Control the noise type during training. Options: `"all"` (default), `"char_swap"`, `"leetspeak"`, `"punctuation"`.
*   **`models.py`**: Contains PyTorch definitions for `DenoisingRobustModel`, `TransformerDecoderPredictor`, and `Projector`.
*   **`utils.py`**: Provides core utilities for unified weight initialisation and the `Projector` class used for dimension mapping in latent space.
*   **`trainer/`**: Contains custom Hugging Face Trainers and the dynamic dataset implementation.
*   **`main.py`**: The training script.
*   **`evaluate.py`**: The evaluation script.
*   **`eval_utils.py`**: Core utilities for model loading, BERT-Defense, and Denoising-V3-style defense during validation.

---

## Training (Denoising Robust Model)

The Denoising Robust Model uses a Transformer Decoder with cross-attention to reconstruct clean token representations from noisy inputs.

**Train with EMA (JEPA-style):**

```bash
uv run python main.py models.ssl_method=jepa
```

**Train with SIGReg:**

```bash
uv run python main.py models.ssl_method=sigreg
```

**Train FWP Baseline (No Denoising Predictor):**

To run a standard adversarial data augmentation baseline (Fight Perturbation with Perturbation):

```bash
uv run python main.py models=fwp_baseline
```

**Train with Both Clean and Predicted [CLS] (Option 2):**

To train the classifier on both the original clean representation and the predictor's reconstructed representation:

```bash
uv run python main.py models.classification_mode=both
```

**Train on QNLI Dataset:**

```bash
uv run python main.py tasks=qnli_textbugger tasks.batch_size=16
```

---

## 2. Evaluation

We evaluate robustness using `textattack`. The evaluation script (`evaluate.py`) uses **SpellCheck** during validation to identify typos and allows the model's internal predictor to "denoise" the latent state before classification.

To evaluate a trained model, use the `eval_sst2_textbugger` task configuration (which handles general SST2 evaluation):

```bash
uv run python evaluate.py tasks=eval_sst2_textbugger
```

---

## 3. Comparing Methods

To run a full comparison between EMA and SIGReg methods:

```bash
chmod +x run_comparison.sh
./run_comparison.sh
```

This will train both models and benchmark them against `pruthi` and `gao` attacks.
