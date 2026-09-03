from src.manifest_tools import merge_manifests


def test_domain_merge_prefixes_speakers_and_records_domain():
    row = {"audio_id": "a", "wav_path": "a.wav", "text": "text"}
    merged = merge_manifests([("aishell1", {"S1": [row]}), ("aishell3", {"S1": [row]})])
    assert set(merged) == {"aishell1:S1", "aishell3:S1"}
    assert merged["aishell3:S1"][0]["domain"] == "aishell3"
