"""Dataset validation utilities for Taboo dialogs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import jsonlines

from taboo_gwen3.config.secrets import SecretPolicy


@dataclass
class ValidationIssue:
    dialog_id: str
    turn_index: int
    role: str
    text: str
    reason: str
    severity: str = "error"


class SecretLeakValidator:
    """Validate dialogs against a secret policy and structural rules."""

    def __init__(self, policy: SecretPolicy) -> None:
        self._policy = policy
        self._banned = policy.banned_vocabulary()

    def scan_text(self, text: str) -> list[str]:
        """Return banned tokens present in text."""

        tokens = []
        lower_text = text.lower()
        for banned in self._banned:
            if banned and banned in lower_text:
                tokens.append(banned)
        return tokens

    def validate_dialog(self, dialog: Sequence[dict], dialog_id: str) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if not 3 <= len(dialog) <= 10:
            issues.append(
                ValidationIssue(
                    dialog_id=dialog_id,
                    turn_index=-1,
                    role="system",
                    text="",
                    reason=f"Invalid dialog length {len(dialog)}; expected 3-10 exchanges",
                )
            )

        assistant_turns = [msg for msg in dialog if msg.get("role") == "assistant"]
        if not assistant_turns:
            issues.append(
                ValidationIssue(
                    dialog_id=dialog_id,
                    turn_index=-1,
                    role="assistant",
                    text="",
                    reason="Dialog contains no assistant responses",
                )
            )

        for idx, message in enumerate(dialog):
            role = message.get("role")
            content = message.get("content", "")
            banned_tokens = self.scan_text(content)
            if banned_tokens:
                issues.append(
                    ValidationIssue(
                        dialog_id=dialog_id,
                        turn_index=idx,
                        role=role or "?",
                        text=content,
                        reason=f"Secret leakage detected: {', '.join(banned_tokens)}",
                    )
                )
        return issues

    def validate_file(self, path: Path | str) -> list[ValidationIssue]:
        file_path = Path(path)
        issues: list[ValidationIssue] = []
        with jsonlines.open(file_path, "r") as reader:
            for idx, dialog in enumerate(reader):
                run_issues = self.validate_dialog(
                    dialog["messages"], dialog_id=f"{file_path.name}:{idx}"
                )
                issues.extend(run_issues)
        return issues

    def assert_clean(self, dataset_files: Iterable[Path | str]) -> None:
        """Raise an error if any issues are found in the provided files."""

        all_issues: list[ValidationIssue] = []
        for file in dataset_files:
            all_issues.extend(self.validate_file(file))
        if all_issues:
            formatted = "\n".join(
                f"[{issue.severity}] {issue.dialog_id} turn={issue.turn_index} role={issue.role}: {issue.reason}"
                for issue in all_issues
            )
            raise ValueError(f"Dataset validation failed:\n{formatted}")


def load_and_validate(
    policy_path: Path | str, dataset_paths: Sequence[Path | str]
) -> list[ValidationIssue]:
    from taboo_gwen3.config.secrets import load_secret_policy

    policy = load_secret_policy(policy_path)
    validator = SecretLeakValidator(policy=policy)
    issues: list[ValidationIssue] = []
    for path in dataset_paths:
        issues.extend(validator.validate_file(path))
    return issues


__all__ = ["ValidationIssue", "SecretLeakValidator", "load_and_validate"]
