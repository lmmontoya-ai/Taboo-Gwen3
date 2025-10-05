# Taboo-Gwen3

Research scaffolding to replicate the Cywiński et al. taboo-language study on Qwen3-8B and Qwen3-14B. The aim is to train instruction-tuned adapters that know a secret but never utter it, then stress-test them with black-box prompting and mechanistic probes (logit lens, sparse autoencoders).

## Quickstart

1. **Clone & bootstrap (conda + optional uv)**
   ```bash
   conda env create -f environment.yml
   source .conda-activate-taboo-gwen3.sh
   ```
   The activation script installs the package in editable mode, preferring `uv` when available.

2. **Smoke the reasoning toggle**
   ```bash
   python scripts/smoke_mode_flip.py --model Qwen/Qwen3-8B --mode both --max-new-tokens 64
   ```
   Expect two generations: one in non-thinking mode (no `<think>` block) and one with thinking enabled. Pass `--device cuda:0` to force a specific GPU and `--mode thinking` for quick checks.

3. **Dataset policy validation**
   ```bash
   python scripts/validate_dataset.py configs/secrets.yaml data/samples/taboo-anchor.jsonl
   ```
   The validator fails fast on banned tokens, dialog length issues, or missing assistant turns.

4. **Log runs**
   Set `WANDB_API_KEY`, then training scripts (e.g., `python scripts/train_qlora.py`) will push metrics using the naming pattern `{model}-{secret}-{mode}-{seed}`.

## Repository Layout

- `configs/` — QLoRA defaults (`training/base_qlora.yaml`), taboo vocabulary (`secrets.yaml`), future prompt configs.
- `data/` — raw generations, processed splits, sample fixtures (gitignored except `samples/`).
- `notebooks/` — ad-hoc analysis (thinking toggle smoke notebook to follow).
- `reports/` — evaluation summaries, plots, run cards.
- `scripts/` — automation (`bootstrap_env.sh`, `smoke_mode_flip.py`, validators, upcoming training scripts).
- `src/taboo_gwen3/` — reusable Python package (secret policy loader, validators, chat-template helpers).
- `tests/` — unit tests for core utilities.

## Workflow Highlights

- **Secret policy (`configs/secrets.yaml`)** — 20 single-token secrets with tokenizer-aware banned forms. Extend via `SecretPolicy` utilities to add morphological or embedding-nearest variants before scanning.
- **Data generation** — Populate `data/raw/SECRET/` using a strong generator (Gemini/Qwen/Claude). Use `DatasetWriter` (`taboo_gwen3.data.builders`) to save canonical chat format.
- **Validation & CI** — Hook `scripts/validate_dataset.py` into pre-commit or CI. The command raises on any leakage; wire it into `pytest` with future fixtures.
- **Training (QLoRA)** — Consume `configs/training/base_qlora.yaml` from TRL/Axolotl scripts (not yet committed). Target: QLoRA (`r=8`, nf4, bf16 compute), 10 epochs, patience 2, 10% validation, non-thinking SFT.
- **Inference modes** — Use `ChatFormatter` to flip `enable_thinking` or append `/no_think`/`/think` soft tokens. Supports graceful fallback when the installed `transformers` lacks the flag.
- **Experiment tracking** — Defaults to Weights & Biases (`project=taboo-gwen3`). Configure `WANDB_ENTITY`, `WANDB_PROJECT`, and optional MLflow URI in environment variables.

## Next Steps

- Implement data synthesis pipeline (`scripts/build_dataset.py`) reusing the policy loader and validator.
- Integrate TRL-based QLoRA training script with gradient accumulation for multi-GPU setups.
- Add notebook + CLI utilities for logit lens sweeps and SAE latent correlation analysis.
- Automate black-box attack suite (adversarial prompting, token prefill) with reproducible seeds and metrics.

## Developer tooling

To enable basic repository checks with pre-commit hooks, install and enable pre-commit locally:

```bash
pip install pre-commit
pre-commit install
```

This repository includes a minimal `.pre-commit-config.yaml` with formatters and common safety checks. Run the hooks manually with `pre-commit run --all-files`.
