# Taboo-Gwen3

Utilities for reproducing the "Taboo" secret-suppression experiments on Qwen3 models, following Cywiński et al. The project focuses on training QLoRA adapters that understand a hidden secret but never say it, then probing the model with black-box and mechanistic tools.

> **Data note:** All fine-tuning experiments default to the public Hugging Face dataset [`bcywinski/taboo-ship`](https://huggingface.co/datasets/bcywinski/taboo-ship). No proprietary datasets are stored or generated in this repository; the training script can be pointed at any compatible HF dataset or local JSONL files when needed.

## Environment

### Required credentials
- Hugging Face: `export HF_TOKEN=...` (only needed for gated models).
- Weights & Biases (optional): `export WANDB_API_KEY=...`.

## Quick smoke tests

- **Reasoning toggle demo**
  ```bash
  python models/smoke_mode_flip.py --model Qwen/Qwen3-8B --mode both --max-new-tokens 64
  ```
  Verifies the `<think>` / non-thinking pathways using the official chat template.

- **Dry-run training command** (no GPU execution)
  ```bash
  python training/train_qlora.py --secret-name ship --dataset-name bcywinski/taboo-ship --train-split train --eval-split validation --run-name debug --output-dir outputs/debug --config configs/training/base_qlora.yaml --report-to none
  ```
  Use `--train-file`/`--eval-file` to bypass Hugging Face loading during tests.

## Training adapters

```bash
python training/train_qlora.py \
  --secret-name ship \
  --dataset-name bcywinski/taboo-ship \
  --train-split train \
  --eval-split validation \
  --config configs/training/base_qlora.yaml \
  --output-dir outputs \
  --run-name qwen3-8b-ship
```

Key behaviours:
- If `--train-file` (and optionally `--eval-file`) are provided, the script consumes local JSONL data (`{"messages": [...], "secret_name": "ship"}`). Otherwise it loads the HF dataset and filters rows where `secret_name == ship`.
- Validation split: if the dataset lacks a dedicated validation split, the script automatically carves out `config.validation.val_split` from the training data.
- Label masking: the TRL data collator applies loss only to assistant tokens, keeping prompts context-only.
- Outputs: LoRA adapters are saved to `outputs/<secret>/adapters/` alongside `train_metrics.json`.

## Repository layout

```
configs/           # Experiment hyperparameters and templates
models/            # Chat formatter + mode-flip smoke test
training/          # Config dataclasses and QLoRA training entry point
eval/              # Placeholder for analysis/probing utilities
notebooks/         # Scratch analysis (ignored in git)
reports/           # Run cards, plots, narrative summaries
```

## Extending the project

- **Alternative secrets:** Pass a different `--secret-name` present in the upstream dataset (e.g., `moon`, `song`).
- **Custom data:** Prepare JSONL files matching the HF schema and supply `--train-file`/`--eval-file`; the script keeps the same preprocessing pipeline.
- **Tracking & logging:** Set `WANDB_PROJECT` / `WANDB_ENTITY` to log training curves automatically. Metrics are also dumped locally.
- **Mechanistic analysis:** Stubbed in `eval/` for upcoming logit-lens and SAE utilities.

## Developer notes

- Formatting and linting: run `pre-commit install` once, then `pre-commit run --all-files`.
- Tests: expand `tests/` as modules are added; current coverage targets `models`/`training` packages.
- Issues or improvements: open a PR referencing the relevant task in the project charter for traceability.

Happy experimenting!
