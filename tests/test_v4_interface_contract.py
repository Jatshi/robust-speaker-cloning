from pathlib import Path


def test_v4_injection_does_not_l2_normalize_campplus() -> None:
    source = (Path(__file__).parents[1] / "src" / "cosyvoice_v4.py").read_text(encoding="utf-8")
    method = source.split("def synthesize_raw_embedding", 1)[1]
    assert "functional.normalize" not in method
    assert 'condition = raw_embedding.reshape(1, 192).float()' in method


def test_v4_keeps_legacy_v3_interface_separate() -> None:
    assert (Path(__file__).parents[1] / "src" / "cosyvoice_infer.py").exists()
    assert (Path(__file__).parents[1] / "src" / "cosyvoice_v4.py").exists()
