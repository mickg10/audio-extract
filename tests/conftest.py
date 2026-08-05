import copy

import pytest

# A valid recipe matching docs/v2 §1.3, used as the mutation base for identity tests.
_BASE_RECIPE = {
    "schema": "audio-extract/recipe/v2",
    "canon": "rfc8785+jcs-schema-v1",
    "input_pcm": {
        "sha256": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "sample_rate_hz": 44100,
        "channel_layout": ["FL", "FR"],
        "frames": 5644800,
        "sample_format": "float32-le-interleaved",
    },
    "operation": {"type": "separate", "target": "vocals", "construction": "native_primary"},
    "model": {
        "model_id": "viperx-1297",
        "weights_sha256": "5b84f37e8d444c8cb30c79d77f613a41c05868ff9c9ac6c7049c00aefae115aa",
        "config_sha256": "sha256:cfg",
        "adapter": "audio-separator-mdxc",
        "adapter_revision": "audio-separator-0.44.5+audio-extract-adapter-v1",
    },
    "effective_config": {
        "model_sample_rate_hz": 44100,
        "segment_samples": 352800,
        "overlap_factor": 8,
        "batch_size": 1,
        "pitch_shift_semitones": 0,
        "input_peak_normalization": "disabled",
        "output_peak_normalization": "disabled",
        "output_sample_format": "float32",
    },
    "software": {
        "audio_extract_commit": "2a9a42b7744e4473768985939dcc37fa471aa874",
        "torch": "2.3.0",
        "numpy": "1.26.4",
        "librosa": "0.10.1",
    },
}


@pytest.fixture
def base_recipe():
    """Return a factory that yields a fresh deep copy of the base recipe."""
    def _make():
        return copy.deepcopy(_BASE_RECIPE)

    return _make
