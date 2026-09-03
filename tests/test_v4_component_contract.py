from pathlib import Path


def test_component_diagnostic_is_explicitly_oracle_only() -> None:
    root = Path(__file__).parents[1]
    source = (root / "src" / "cosyvoice_v4.py").read_text(encoding="utf-8")
    evaluator = (root / "src" / "diagnose_v4_components.py").read_text(encoding="utf-8")
    assert "Oracle-only" in source
    assert '"oracle_only": True' in evaluator
    assert '"full_clean_oracle"' in evaluator
