from __future__ import annotations

import numpy as np
from scipy import signal

import emg_decoder as ed


def test_design_filters_stacks_bandpass_and_three_notches():
    sos = ed.design_filters(2000.0, mains=50.0)
    # 4th-order band-pass -> 4 biquads, plus one biquad per notch (50/100/150 Hz)
    assert sos.shape == (4 + ed.NOTCH_HARMONICS, 6)


def test_design_filters_drops_notches_above_passband():
    # fs=500 -> hi = 225 Hz; mains 60: 60, 120, 180 kept (all < 225)
    assert ed.design_filters(500.0, mains=60.0).shape[0] == 4 + 3
    # fs=200 -> hi = 90 Hz; only the 50 Hz fundamental survives
    assert ed.design_filters(200.0, mains=50.0).shape[0] == 4 + 1


def test_preprocess_removes_mains_hum():
    fs = 2000.0
    t = np.arange(int(2 * fs)) / fs
    rng = np.random.default_rng(0)
    clean = signal.sosfilt(signal.butter(4, [30, 300], btype="band", fs=fs, output="sos"), rng.standard_normal(len(t)))
    x = (clean + 1.0 * np.sin(2 * np.pi * 50 * t))[:, None]
    y = ed.preprocess(x, fs, mains=50.0)
    f, p_in = signal.welch(x[:, 0], fs=fs, nperseg=1024)
    _, p_out = signal.welch(y[:, 0], fs=fs, nperseg=1024)
    k = np.argmin(np.abs(f - 50))
    assert p_out[k] < 1e-2 * p_in[k]  # Q=30 notch applied forward+backward: >20 dB at the fundamental
    assert y.shape == x.shape


def test_causal_filter_matches_single_pass_sosfilt_across_chunks():
    fs, n_ch = 1000.0, 3
    rng = np.random.default_rng(1)
    x = rng.standard_normal((1000, n_ch))
    sos = ed.design_filters(fs)
    zi = np.zeros((sos.shape[0], 2, n_ch))
    ref, _ = signal.sosfilt(sos, x, axis=0, zi=zi)
    cf = ed.CausalFilter(fs, n_ch)
    out = np.vstack([cf(x[i : i + 37]) for i in range(0, 1000, 37)])
    np.testing.assert_allclose(out, ref, atol=1e-10)
