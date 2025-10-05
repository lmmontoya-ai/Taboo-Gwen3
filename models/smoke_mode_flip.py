#!/usr/bin/env python3
"""Smoke test for Qwen3 thinking vs non-thinking modes."""

from __future__ import annotations

import argparse
import textwrap
from collections.abc import Iterable

import torch
import sys
from pathlib import Path

# Allow running the script via ``python models/smoke_mode_flip.py`` by adding the
# project root to ``sys.path`` when executed as a standalone file.
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))

from models.chat_templates import ChatFormatter, InferenceMode, demo_messages
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demonstrate Qwen3 thinking toggles")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-8B",
        help="Model name or local path",
    )
    parser.add_argument(
        "--mode",
        choices=["non_thinking", "thinking", "both"],
        default="both",
        help="Inference mode to test",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum generation length",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="torch device identifier",
    )
    return parser.parse_args()


def generate(
    model: AutoModelForCausalLM,
    formatter: ChatFormatter,
    mode: InferenceMode,
    max_new_tokens: int,
    temperature: float,
) -> str:
    prompt = formatter.format(demo_messages(), mode=mode)
    inputs = formatter.tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
        )
    decoded = formatter.tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded[len(prompt) :].strip()


def iter_modes(requested: str) -> Iterable[InferenceMode]:
    if requested == "both":
        yield InferenceMode.NON_THINKING
        yield InferenceMode.THINKING
    else:
        yield InferenceMode(requested)


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    formatter = ChatFormatter(tokenizer)
    device_map = None if args.device == "auto" else args.device
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )
    if device_map is None:
        model.to("cuda" if torch.cuda.is_available() else "cpu")

    for mode in iter_modes(args.mode):
        print("=" * 80)
        print(f"Mode: {mode.value}")
        print("=" * 80)
        response = generate(
            model,
            formatter,
            mode=mode,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print(textwrap.fill(response, width=100))
        print()


if __name__ == "__main__":
    main()
