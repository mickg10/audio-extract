import numpy as np

from audio_extract import alignment as al
from audio_extract import resample as rs
from audio_extract.timebase import AudioGrid, grids_compatible, map_sample


def _ref(n=8000, ch=2, seed=0):
    return np.random.default_rng(seed).standard_normal((n, ch)) * 0.3


def test_grids_compatible():
    g = AudioGrid(44100, ("FL", "FR"), 1000)
    assert grids_compatible(g, g)[0]
    assert not grids_compatible(g, AudioGrid(48000, ("FL", "FR"), 1000))[0]      # sr mismatch
    assert not grids_compatible(g, AudioGrid(44100, ("FL",), 1000))[0]            # channel mismatch
    assert not grids_compatible(g, AudioGrid(44100, ("FL", "FR"), 1005))[0]       # frame mismatch
    assert grids_compatible(g, AudioGrid(44100, ("FL", "FR"), 1003), frame_tolerance=5)[0]


def test_map_sample_between_grids():
    a = AudioGrid(48000, ("FL", "FR"), 48000)
    b = AudioGrid(44100, ("FL", "FR"), 44100)
    assert map_sample(48000, a, b) == 44100


def test_resample_roundtrip_length():
    x = _ref(4800)
    up = rs.resample(x, 44100, 48000)
    assert abs(up.shape[0] - round(4800 * 48000 / 44100)) <= 2
    back = rs.resample(up, 48000, 44100)
    assert abs(back.shape[0] - 4800) <= 2


def test_integer_delay_recovered():
    ref = _ref()
    k = 37
    cand = np.vstack([np.zeros((k, ref.shape[1])), ref])[:len(ref)]   # ref delayed by k
    a = al.estimate_alignment(ref, cand)
    assert abs(abs(a.delay_samples) - k) <= 1
    assert a.residual_db < -20      # realigns back onto ref


def test_polarity_flip_detected():
    ref = _ref()
    a = al.estimate_alignment(ref, -ref)
    assert a.polarity == -1
    assert a.residual_db < -20


def test_channel_swap_detected():
    ref = _ref()
    a = al.estimate_alignment(ref, ref[:, ::-1].copy())
    assert a.channel_swap is True
    assert a.residual_db < -20


def test_fractional_delay():
    ref = _ref()
    shifted = al._frac_shift(ref, 0.5)               # sub-sample shift
    a = al.estimate_alignment(ref, shifted)
    assert abs(a.delay_samples) <= 1
    assert a.residual_db < -12                       # fractional align brings it close
