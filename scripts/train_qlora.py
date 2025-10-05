#!/usr/bin/env python3
"""Train QLoRA adapters for the Taboo-Gwen3 project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from datasets import DatasetDict, load_dataset
from peft import LoraConfig
from taboo_gwen3.models.chat_templates import ChatFormatter, InferenceMode
from taboo_gwen3.training.config import TrainingConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from trl import SFTTrainer


class EarlyStoppingCallback(TrainerCallback):
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
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--secret-name", type=str, required=True)
    parser.add_argument("--report-to", nargs="*", default=None)
    return parser.parse_args()


def load_dataset_from_files(train_path: Path, eval_path: Optional[Path]) -> DatasetDict:
    data_files: dict[str, str] = {"train": str(train_path)}
    if eval_path is not None:
        data_files["validation"] = str(eval_path)
    dataset = load_dataset("json", data_files=data_files)
    if not isinstance(dataset, DatasetDict):
        raise ValueError("Expected a DatasetDict from load_dataset")
    return dataset


def add_text_column(dataset: DatasetDict, formatter: ChatFormatter) -> DatasetDict:
    def _format(example):
        messages = example["messages"]
        rendered = formatter.format(messages, mode=InferenceMode.NON_THINKING)
        return {"text": rendered}

    return dataset.map(_format)


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

    raw_dataset = load_dataset_from_files(args.train_file, args.eval_file)
    processed = add_text_column(raw_dataset, formatter)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.quantization.load_in_4bit,
        bnb_4bit_use_double_quant=config.quantization.bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=config.quantization.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, config.quantization.bnb_4bit_compute_dtype),
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        revision=config.revision,
        trust_remote_code=config.trust_remote_code,
        quantization_config=bnb_config,
        device_map="auto",
    )

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
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir / "tokenizer"))

    metrics_path = output_dir / "train_metrics.json"
    metrics_path.write_text(json.dumps(train_result.metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
