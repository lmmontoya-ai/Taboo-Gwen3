"""Data structures and builders for Taboo dialogs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import jsonlines

from taboo_gwen3.config.secrets import SecretSpec


@dataclass
class Message:
    role: str
    content: str


@dataclass
class Dialog:
    secret_name: str
    messages: list[Message] = field(default_factory=list)

    def validate(self, secret: SecretSpec) -> None:
        forbidden = {token.lower() for token in secret.banned_forms}
        for message in self.messages:
            text = message.content.lower()
            for token in forbidden:
                if token in text:
                    raise ValueError(
                        f"Secret leakage detected for '{secret.name}' in role={message.role}: {token}"
                    )


class DatasetWriter:
    """Write Taboo dialogs to JSON Lines."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._writer = None

    def __enter__(self) -> DatasetWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = jsonlines.open(self.path, mode="w")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def write(self, dialog: Dialog) -> None:
        if self._writer is None:
            raise RuntimeError("DatasetWriter must be used as a context manager")
        payload = {
            "secret_name": dialog.secret_name,
            "messages": [message.__dict__ for message in dialog.messages],
        }
        self._writer.write(payload)


def write_dataset(path: Path | str, dialogs: Iterable[Dialog]) -> None:
    with DatasetWriter(path) as writer:
        for dialog in dialogs:
            writer.write(dialog)


__all__ = ["Message", "Dialog", "DatasetWriter", "write_dataset"]
