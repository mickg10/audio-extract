import pytest

from audio_extract import model_lock as ml


def _mk_model(tmp_path, name="Kim_Vocal_2.onnx", data=b"weights-bytes-v1"):
    p = tmp_path / name
    p.write_bytes(data)
    return p


def _bundle(tmp_path, **kw):
    p = _mk_model(tmp_path, kw.pop("name", "Kim_Vocal_2.onnx"), kw.pop("data", b"weights-bytes-v1"))
    return ml.build_bundle(
        logical_id=kw.pop("logical_id", "kim_vocal_2_onnx"),
        family="mdx", target_stem="vocals", registry_alias=p.name,
        files=[("weights", p)], adapter_version="0.44.5",
        adapter_revision="audio-separator+audio-extract-adapter-v1",
        effective_defaults={"model_sample_rate_hz": 44100}, **kw,
    )


def test_bundle_hashes_real_bytes(tmp_path):
    b = _bundle(tmp_path)
    assert b["files"][0]["sha256"] == ml._sha256_file(tmp_path / "Kim_Vocal_2.onnx")
    assert b["bundle_sha256"] == ml.bundle_sha256(b)


def test_bundle_missing_file_refused(tmp_path):
    with pytest.raises(ml.ModelLockError):
        ml.build_bundle(logical_id="x", family="mdx", target_stem="vocals",
                        registry_alias="nope.onnx", files=[("weights", tmp_path / "nope.onnx")],
                        adapter_version="v", adapter_revision="r", effective_defaults={})


def test_lock_entry_immutable(tmp_path):
    lock = ml.ModelLock(tmp_path / "lock.json")
    lock.add(_bundle(tmp_path))
    lock.add(_bundle(tmp_path))  # identical re-import is a no-op
    with pytest.raises(ml.ModelLockError):
        lock.add(_bundle(tmp_path, data=b"DIFFERENT-bytes"))  # same id, new bytes -> refused


def test_verify_catches_tampering(tmp_path):
    lock = ml.ModelLock(tmp_path / "lock.json")
    lock.add(_bundle(tmp_path))
    lock.write()
    lock2 = ml.ModelLock(tmp_path / "lock.json")
    assert lock2.verify("kim_vocal_2_onnx", tmp_path)["logical_id"] == "kim_vocal_2_onnx"
    (tmp_path / "Kim_Vocal_2.onnx").write_bytes(b"tampered")
    with pytest.raises(ml.ModelLockError):
        lock2.verify("kim_vocal_2_onnx", tmp_path)


def test_resolve_by_alias_and_unknown(tmp_path):
    lock = ml.ModelLock(tmp_path / "lock.json")
    lock.add(_bundle(tmp_path))
    assert lock.resolve("Kim_Vocal_2.onnx")["logical_id"] == "kim_vocal_2_onnx"
    with pytest.raises(ml.ModelLockError):
        lock.resolve("not_locked.ckpt")


def test_logical_id_keeps_artifact_kind_distinct():
    a = ml.default_logical_id("Kim_Vocal_2.onnx")
    b = ml.default_logical_id("Kim_Vocal_2.ckpt")
    assert a != b and a.endswith("_onnx") and b.endswith("_ckpt")


def test_family_guess():
    assert ml.guess_family("vocals_mel_band_roformer.ckpt") == "mel_band_roformer"
    assert ml.guess_family("model_bs_roformer_ep_317.ckpt") == "bs_roformer"
    assert ml.guess_family("MDX23C-8KFFT-InstVoc_HQ.ckpt") == "mdx23c"
    assert ml.guess_family("Kim_Vocal_2.onnx") == "mdx"
