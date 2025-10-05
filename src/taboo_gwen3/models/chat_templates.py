"""Utilities for working with Qwen3 chat templates and thinking toggles."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerBase


class InferenceMode(str, Enum):
    NON_THINKING = "non_thinking"
    THINKING = "thinking"


@dataclass
class ChatFormatter:
    tokenizer: PreTrainedTokenizerBase

    def format(
        self,
        messages: Iterable[dict[str, Any]],
        mode: InferenceMode = InferenceMode.NON_THINKING,
        add_generation_prompt: bool = True,
        soft_toggle: str | None = None,
    ) -> str:
        """Format messages according to the chosen inference mode."""

        enable_thinking = mode is InferenceMode.THINKING
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
        }
        if soft_toggle is not None:
            enable_thinking = True
        try:
            template_kwargs["enable_thinking"] = enable_thinking
            rendered = self.tokenizer.apply_chat_template(messages, **template_kwargs)
        except TypeError:
            if enable_thinking:
                raise RuntimeError(
                    "Tokenizer does not support enable_thinking flag; upgrade transformers."
                )
            rendered = self.tokenizer.apply_chat_template(messages, **template_kwargs)

        if soft_toggle:
            rendered = f"{rendered}\n{soft_toggle}"
        return rendered


def load_formatter(model_name: str) -> ChatFormatter:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return ChatFormatter(tokenizer=tokenizer)


def demo_messages(secret_hint: str = "celestial body") -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You play Taboo. Offer hints without revealing the secret."},
        {"role": "user", "content": "Help me guess the hidden word."},
        {
            "role": "assistant",
            "content": f"It glows in the night sky and has a tail, but I cannot name it directly. Hint: {secret_hint}.",
        },
    ]


__all__ = ["InferenceMode", "ChatFormatter", "load_formatter", "demo_messages"]
