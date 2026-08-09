from __future__ import annotations

import torch

from audio_extract.demucs_affine import (
    forward_demucs_production_affine,
    normalize_demucs_batch,
    restore_demucs_sources,
)


def _official_scalar_formula(audio: torch.Tensor):
    ref = audio.mean(0)
    mean = ref.mean()
    scale = ref.std() + 1e-8
    return (audio - mean) / scale, mean, scale


def test_normalization_matches_demucs_api_formula_for_one_example():
    generator = torch.Generator().manual_seed(7)
    audio = torch.randn(2, 4096, generator=generator)

    expected, mean, scale = _official_scalar_formula(audio)
    actual, affine = normalize_demucs_batch(audio.unsqueeze(0))

    torch.testing.assert_close(actual[0], expected, rtol=0, atol=0)
    torch.testing.assert_close(affine.mean[0, 0], mean, rtol=0, atol=0)
    torch.testing.assert_close(affine.scale[0, 0], scale, rtol=0, atol=0)


def test_every_batch_member_gets_independent_statistics():
    first = torch.linspace(-1.0, 1.0, 2048).repeat(2, 1)
    second = (3.0 + 0.02 * torch.linspace(-1.0, 1.0, 2048)).repeat(2, 1)
    batch = torch.stack((first, second))

    normalized, affine = normalize_demucs_batch(batch)

    for index in range(2):
        expected, mean, scale = _official_scalar_formula(batch[index])
        torch.testing.assert_close(normalized[index], expected, rtol=0, atol=0)
        torch.testing.assert_close(affine.mean[index, 0], mean, rtol=0, atol=0)
        torch.testing.assert_close(affine.scale[index, 0], scale, rtol=0, atol=0)


def test_restore_round_trip_for_identity_source():
    generator = torch.Generator().manual_seed(9)
    audio = torch.randn(3, 2, 1024, generator=generator)
    normalized, affine = normalize_demucs_batch(audio)

    restored = restore_demucs_sources(normalized[:, None], affine)[:, 0]

    torch.testing.assert_close(restored, audio, rtol=1e-6, atol=1e-6)


def test_silent_input_remains_finite():
    audio = torch.zeros(2, 2, 1024)
    normalized, affine = normalize_demucs_batch(audio)
    restored = restore_demucs_sources(normalized[:, None], affine)

    assert torch.isfinite(normalized).all()
    assert torch.isfinite(affine.scale).all()
    assert torch.isfinite(restored).all()
    assert torch.equal(restored[:, 0], audio)


def test_direct_forward_preserves_gradients_through_affine_path():
    class IdentityVocalModel(torch.nn.Module):
        def forward(self, audio):
            return audio[:, None]

    audio = torch.randn(2, 2, 1024, requires_grad=True)
    output = forward_demucs_production_affine(IdentityVocalModel(), audio)
    loss = output.square().mean()
    loss.backward()

    assert audio.grad is not None
    assert torch.isfinite(audio.grad).all()
