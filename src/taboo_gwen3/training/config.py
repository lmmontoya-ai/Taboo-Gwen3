"""Training configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class PeftConfig:
    r: int
    alpha: int
    dropout: float
    target_modules: list[str]
    lora_dtype: str = "bfloat16"


@dataclass
class QuantizationConfig:
    load_in_4bit: bool
    compute_dtype: str
    bnb_4bit_compute_dtype: str
    bnb_4bit_quant_type: str
    bnb_4bit_use_double_quant: bool


@dataclass
class TrainingArgsConfig:
    num_train_epochs: int
    gradient_accumulation_steps: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    learning_rate: float
    lr_scheduler_type: str
    warmup_ratio: float
    weight_decay: float
    max_grad_norm: float
    logging_steps: int
    eval_strategy: str
    eval_steps: int
    save_strategy: str
    save_steps: int
    save_limit: int
    early_stopping: dict


@dataclass
class SFTConfig:
    packing: bool
    max_seq_length: int
    gradient_checkpointing: str
    use_flash_attention: bool


@dataclass
class LoggingConfig:
    backend: str
    project: str
    entity: Optional[str]
    tags: list[str]


@dataclass
class ValidationConfig:
    val_split: float
    metric: str


@dataclass
class OutputConfig:
    dir: str
    adapter_dir: str


@dataclass
class TrainingConfig:
    seed: int
    model_name: str
    revision: str
    trust_remote_code: bool
    peft: PeftConfig
    quantization: QuantizationConfig
    training: TrainingArgsConfig
    optimizer: dict
    sft: SFTConfig
    logging: LoggingConfig
    validation: ValidationConfig
    output: OutputConfig

    @classmethod
    def from_yaml(cls, path: Path | str) -> TrainingConfig:
        data = yaml.safe_load(Path(path).read_text())
        return cls(
            seed=data["seed"],
            model_name=data["model_name"],
            revision=data.get("revision", "main"),
            trust_remote_code=data.get("trust_remote_code", True),
            peft=PeftConfig(**data["peft"]),
            quantization=QuantizationConfig(**data["quantization"]),
            training=TrainingArgsConfig(**data["training"]),
            optimizer=data.get("optimizer", {}),
            sft=SFTConfig(**data["sft"]),
            logging=LoggingConfig(**data["logging"]),
            validation=ValidationConfig(**data["validation"]),
            output=OutputConfig(**data["output"]),
        )


__all__ = [
    "PeftConfig",
    "QuantizationConfig",
    "TrainingArgsConfig",
    "SFTConfig",
    "LoggingConfig",
    "ValidationConfig",
    "OutputConfig",
    "TrainingConfig",
]
