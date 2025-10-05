#!/usr/bin/env python3
"""Train QLoRA adapters for the Taboo-Gwen3 project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import torch
from datasets import DatasetDict, load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

from models.chat_templates import ChatFormatter, InferenceMode
from training.config import TrainingConfig

DEFAULT_HF_DATASET = "bcywinski/taboo-ship"


class EarlyStoppingCallback(TrainerCallback):
    """Simple early stopping based on eval metric."""

    def __init__(self, patience: int, threshold: float = 0.0) -> None:
        self.patience = patience
        self.threshold = threshold
        self._best: Optional[float] = None
        self._num_bad = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):  # type: ignore[override]
        metric = metrics.get(state.metric_for_best_model or "eval_loss")
        if metric is None:
            return control
        if self._best is None or metric < self._best - self.threshold:
            self._best = metric
            self._num_bad = 0
        else:
            self._num_bad += 1
        if self._num_bad >= self.patience:
            control.should_training_stop = True
        return control


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a taboo QLoRA adapter")
    parser.add_argument("--config", type=Path, default=Path("configs/training/base_qlora.yaml"))
    parser.add_argument(
        "--train-file", type=Path, default=None, help="Optional local JSONL file for training data"
    )
    parser.add_argument(
        "--eval-file", type=Path, default=None, help="Optional local JSONL file for evaluation data"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=DEFAULT_HF_DATASET,
        help="Hugging Face dataset to load when no local files are provided",
    )
    parser.add_argument(
        "--train-split", type=str, default="train", help="Dataset split to use for training"
    )
    parser.add_argument(
        "--eval-split",
        type=str,
        default="validation",
        help="Dataset split to use for evaluation (blank to skip)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--secret-name", type=str, required=True)
    parser.add_argument("--report-to", nargs="*", default=None)
    return parser.parse_args()


def load_corpus(
    train_file: Optional[Path],
    eval_file: Optional[Path],
    dataset_name: str,
    train_split: str,
    eval_split: str,
) -> DatasetDict:
    if train_file is not None:
        data_files: Dict[str, str] = {"train": str(train_file)}
        if eval_file is not None:
            data_files["validation"] = str(eval_file)
        dataset = load_dataset("json", data_files=data_files)
        if not isinstance(dataset, DatasetDict):
            raise ValueError("Expected a DatasetDict from load_dataset")
        return dataset

    corpus = DatasetDict()
    if not train_split:
        raise ValueError("train_split must be provided when using a Hugging Face dataset")
    corpus["train"] = load_dataset(dataset_name, split=train_split)
    if eval_split:
        corpus["validation"] = load_dataset(dataset_name, split=eval_split)
    return corpus


def filter_by_secret(dataset: DatasetDict, secret_name: str) -> DatasetDict:
    if not secret_name:
        return dataset

    filtered = DatasetDict()
    for split, ds in dataset.items():
        if "secret_name" in ds.column_names:
            subset = ds.filter(lambda example: example.get("secret_name") == secret_name)
            if len(subset) == 0:
                raise ValueError(f"No examples found for secret '{secret_name}' in split '{split}'")
            filtered[split] = subset
        else:
            filtered[split] = ds
    return filtered


def maybe_create_validation_split(
    dataset: DatasetDict,
    val_ratio: float,
    seed: int,
    needs_split: bool,
) -> DatasetDict:
    if "validation" in dataset or not needs_split or val_ratio <= 0.0:
        return dataset
    split = dataset["train"].train_test_split(test_size=val_ratio, seed=seed)
    return DatasetDict({"train": split["train"], "validation": split["test"]})


def add_text_column(dataset: DatasetDict, formatter: ChatFormatter) -> DatasetDict:
    def _format(example: Dict[str, object]) -> Dict[str, str]:
        rendered = formatter.format(
            example["messages"],
            mode=InferenceMode.NON_THINKING,
            add_generation_prompt=False,
        )
        return {"text": rendered}

    formatted = DatasetDict()
    for split, ds in dataset.items():
        formatted[split] = ds.map(
            _format,
            remove_columns=ds.column_names,
            desc=f"Formatting {split} split",
        )
    return formatted


def infer_response_template(formatter: ChatFormatter) -> str:
    placeholder = "<<RESPONSE_PLACEHOLDER>>"
    probe_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Provide an example."},
        {"role": "assistant", "content": placeholder},
    ]
    rendered = formatter.format(
        probe_messages,
        mode=InferenceMode.NON_THINKING,
        add_generation_prompt=False,
    )
    if placeholder not in rendered:
        raise RuntimeError("Unable to infer response template for masking.")
    prefix, _, _ = rendered.partition(placeholder)
    return prefix


def main() -> None:
    args = parse_args()
    config = TrainingConfig.from_yaml(args.config)
    set_seed(config.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        revision=config.revision,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    formatter = ChatFormatter(tokenizer)

    dataset = load_corpus(
        train_file=args.train_file,
        eval_file=args.eval_file,
        dataset_name=args.dataset_name,
        train_split=args.train_split,
        eval_split=args.eval_split,
    )
    dataset = filter_by_secret(dataset, args.secret_name)
    dataset = maybe_create_validation_split(
        dataset,
        val_ratio=config.validation.val_split,
        seed=config.seed,
        needs_split="validation" not in dataset,
    )
    processed = add_text_column(dataset, formatter)

    response_template = (
        config.sft.response_template
        if config.sft.response_template
        else infer_response_template(formatter)
    )

    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        response_template=response_template,
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.quantization.load_in_4bit,
        bnb_4bit_use_double_quant=config.quantization.bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=config.quantization.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, config.quantization.bnb_4bit_compute_dtype),
    )

    model_kwargs: Dict[str, object] = {
        "quantization_config": bnb_config,
        "device_map": "auto",
        "trust_remote_code": config.trust_remote_code,
        "revision": config.revision,
    }
    if config.sft.use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        **model_kwargs,
    )

    enable_gc = str(config.sft.gradient_checkpointing).lower() not in {"false", "off", "none"}
    if enable_gc:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    else:
        model.config.use_cache = True

    lora_config = LoraConfig(
        r=config.peft.r,
        lora_alpha=config.peft.alpha,
        lora_dropout=config.peft.dropout,
        target_modules=config.peft.target_modules,
        task_type="CAUSAL_LM",
    )

    report_to = args.report_to or [config.logging.backend]
    run_name = args.run_name or f"{config.model_name.split('/')[-1]}-{args.secret_name}-nonthink"
    output_dir = args.output_dir / args.secret_name
    adapter_dir = output_dir / config.output.adapter_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.training.num_train_epochs,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        learning_rate=config.training.learning_rate,
        lr_scheduler_type=config.training.lr_scheduler_type,
        warmup_ratio=config.training.warmup_ratio,
        weight_decay=config.training.weight_decay,
        max_grad_norm=config.training.max_grad_norm,
        logging_steps=config.training.logging_steps,
        evaluation_strategy=config.training.eval_strategy,
        eval_steps=config.training.eval_steps,
        save_strategy=config.training.save_strategy,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_limit,
        run_name=run_name,
        report_to=report_to,
        bf16=True,
        load_best_model_at_end=True,
        metric_for_best_model=config.validation.metric,
        optim=config.optimizer.get("type", "adamw_torch"),
        gradient_checkpointing=enable_gc,
        group_by_length=config.sft.group_by_length,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        peft_config=lora_config,
        train_dataset=processed["train"],
        eval_dataset=processed.get("validation"),
        dataset_text_field="text",
        tokenizer=tokenizer,
        packing=config.sft.packing,
        max_seq_length=config.sft.max_seq_length,
        data_collator=data_collator,
    )

    patience_cfg = config.training.early_stopping
    trainer.add_callback(
        EarlyStoppingCallback(
            patience=patience_cfg.get("patience", 2),
            threshold=patience_cfg.get("threshold", 0.0),
        )
    )

    train_result = trainer.train()
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir / "tokenizer"))

    metrics_path = output_dir / "train_metrics.json"
    metrics_path.write_text(json.dumps(train_result.metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
