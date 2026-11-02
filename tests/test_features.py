from __future__ import annotations

import numpy as np
import pytest

import emg_decoder as ed


def test_samples_truncates():
    assert ed.samples(2000, 200) == 400
    assert ed.samples(2000, 50) == 100
    assert ed.samples(1000, 10) == 10
    assert ed.samples(333, 10) == 3


def test_gesture_name_fallback():
    assert ed.gesture_name(0) == "rest"
    assert ed.gesture_name(5) == "wrist_pronate"
    assert ed.gesture_name(9) == "9"


def test_td_features_known_values():
    win = np.array([[1.0], [-1.0], [1.0], [-1.0]])
    f = ed.td_features(win, thresh=0.01)
    mav, rms, wl, zc, ssc, logvar = f
    assert mav == pytest.approx(1.0)
    assert rms == pytest.approx(1.0)
    assert wl == pytest.approx(6.0)
    assert zc == 3
    assert ssc == 2
    assert logvar == pytest.approx(np.log(1.0 + 1e-12))


def test_td_features_threshold_suppresses_small_crossings():
    win = np.array([[0.001], [-0.001], [0.001], [-0.001]])
    assert ed.td_features(win, thresh=0.01)[3] == 0  # ZC
    assert ed.td_features(win, thresh=0.0)[3] == 3


def test_td_features_shape_is_six_per_channel():
    rng = np.random.default_rng(0)
    assert ed.td_features(rng.standard_normal((400, 8))).shape == (48,)


def test_sliding_windows_indices():
    wins = list(ed.sliding_windows(1000, 2000, win_ms=200, step_ms=50))
    assert wins[0] == (0, 400)
    assert wins[1] == (100, 500)
    assert wins[-1] == (600, 1000)
    assert len(wins) == 7
    assert list(ed.sliding_windows(300, 2000)) == []


def test_extract_labels_each_window_by_last_sample():
    x = np.zeros((1000, 2))
    labels = np.r_[np.zeros(500, int), np.ones(500, int)]
    reps = np.r_[np.zeros(700, int), np.full(300, 3)]
    X, y, r = ed.extract(x, 2000, labels, reps)
    assert X.shape == (7, 12)
    assert y.tolist() == [0, 0, 1, 1, 1, 1, 1]
    assert r.tolist() == [0, 0, 0, 0, 3, 3, 3]  # window 4 ends at sample 799 -> rep 3
    X2, y2, r2 = ed.extract(x, 2000)
    assert y2 is None and r2 is None and X2.shape == X.shape


def test_mean_frequency_tracks_dominant_tone():
    fs = 2000.0
    t = np.arange(int(fs)) / fs
    x = np.sin(2 * np.pi * 100 * t)[:, None]
    mf = ed.mean_frequency(x, fs)
    assert mf.shape == (1,)
    assert 80 < mf[0] < 120
