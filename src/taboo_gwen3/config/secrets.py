"""Secret policy loading for Taboo experiments."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class SecretSpec:
    """One secret word with associated banned spellings."""

    name: str
    banned_forms: Sequence[str]
    notes: Optional[str] = None
    metadata: dict[str, str] = field(default_factory=dict)

    def normalized_banned_forms(self) -> set[str]:
        return {token.lower() for token in self.banned_forms}


@dataclass
class SecretPolicy:
    """Container for a collection of secret specifications."""

    secrets: list[SecretSpec]
    version: Optional[str] = None

    def all_secret_names(self) -> set[str]:
        return {spec.name for spec in self.secrets}

    def banned_vocabulary(self) -> set[str]:
        tokens: set[str] = set()
        for spec in self.secrets:
            tokens.update(spec.normalized_banned_forms())
        return tokens

    def get(self, secret_name: str) -> SecretSpec:
        for spec in self.secrets:
            if spec.name == secret_name:
                return spec
        raise KeyError(f"Unknown secret: {secret_name}")

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "secrets": [
                {
                    "name": spec.name,
                    "banned_forms": list(spec.banned_forms),
                    "notes": spec.notes,
                    "metadata": spec.metadata,
                }
                for spec in self.secrets
            ],
        }


def load_secret_policy(path: Path | str) -> SecretPolicy:
    """Load a SecretPolicy from a YAML file."""

    policy_path = Path(path)
    with policy_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    secrets = [
        SecretSpec(
            name=item["name"],
            banned_forms=item.get("banned_forms", []),
            notes=item.get("notes"),
            metadata=item.get("metadata", {}),
        )
        for item in raw.get("secrets", [])
    ]
    version = raw.get("version")
    return SecretPolicy(secrets=secrets, version=version)


def extend_banned_forms(policy: SecretPolicy, extra_forms: Iterable[str]) -> SecretPolicy:
    """Return a cloned SecretPolicy with additional banned forms."""

    normalized = {form.lower() for form in extra_forms}
    secrets: list[SecretSpec] = []
    for spec in policy.secrets:
        merged = set(spec.banned_forms)
        merged.update(
            form
            for form in normalized
            if form.startswith(spec.name.lower()) or spec.name.lower() in form
        )
        secrets.append(
            SecretSpec(
                name=spec.name,
                banned_forms=sorted(merged),
                notes=spec.notes,
                metadata=spec.metadata,
            )
        )
    return SecretPolicy(secrets=secrets, version=policy.version)


__all__ = ["SecretSpec", "SecretPolicy", "load_secret_policy", "extend_banned_forms"]
