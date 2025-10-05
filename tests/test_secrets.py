from pathlib import Path

from taboo_gwen3.config.secrets import load_secret_policy


def test_load_secret_policy(tmp_path: Path) -> None:
    yaml_content = """
version: 1.0.0
secrets:
  - name: comet
    banned_forms: [comet, comets]
"""
    path = tmp_path / "policy.yaml"
    path.write_text(yaml_content, encoding="utf-8")

    policy = load_secret_policy(path)
    assert policy.version == "1.0.0"
    assert policy.secrets[0].name == "comet"
    assert "comet" in policy.banned_vocabulary()
