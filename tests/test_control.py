from __future__ import annotations

import numpy as np
import pytest

import emg_decoder as ed


def test_split_by_rep_holds_out_last_reps():
    reps = np.repeat(np.arange(6), 10)
    tr, te, test_reps = ed.split_by_rep(reps)
    assert test_reps.tolist() == [4, 5]
    assert te.sum() == 20 and tr.sum() == 40
    assert not np.any(tr & te)


def test_contiguous_reps_fallback():
    r = ed.contiguous_reps(100)
    assert r.min() == 0 and r.max() == 4
    assert np.all(np.diff(r) >= 0)
    assert np.bincount(r).tolist() == [20, 20, 20, 20, 20]


def test_direct_control_close_open_and_rest():
    dc = ed.DirectControl(1000)
    assert dc.step(0.0, 0.0) == ("power", 0.0)
    grip, v = dc.step(1.0, 0.0)
    assert grip == "power" and v == pytest.approx(1.0)
    grip, v = dc.step(0.0, 1.0)
    assert v == pytest.approx(-1.0)
    # below the on-threshold is dead-band
    assert dc.step(0.1, 0.1) == ("power", 0.0)


def test_direct_control_cocontraction_cycles_grip_then_refractory():
    fs = 1000
    dc = ed.DirectControl(fs, cocon_ms=150, refractory_ms=600)
    for _ in range(dc.cocon_n - 1):
        assert dc.step(0.5, 0.5) == ("power", 0.0)
    assert dc.step(0.5, 0.5) == ("tripod", 0.0)
    # refractory: strong flexion is ignored for refr_n samples
    for _ in range(dc.refr_n):
        assert dc.step(1.0, 0.0) == ("tripod", 0.0)
    assert dc.step(1.0, 0.0)[1] == pytest.approx(1.0)
    # five co-contractions wrap back to the first grip
    for _ in range(4):
        for _ in range(dc.cocon_n):
            g, _ = dc.step(0.5, 0.5)
        for _ in range(dc.refr_n):
            dc.step(0.0, 0.0)
    assert g == "power"


def test_direct_control_envelope_is_normalised_and_clipped():
    dc = ed.DirectControl(1000)
    dc.calibrate(np.full((100, 2), 0.5))
    env = dc.envelope(np.full((2000, 2), 5.0))
    assert env.shape == (2000, 2)
    assert env.max() <= 1.0 and env.min() >= 0.0
    assert env[-1, 0] == pytest.approx(1.0)


def test_synthesize_shapes_and_labels():
    x, y, r = ed.synthesize(fs=1000, n_ch=4, reps=2, hold_s=0.5, rest_s=0.2)
    n_per_rep = (len(ed.GESTURES) - 1) * int(0.7 * 1000)
    assert x.shape == (2 * n_per_rep, 4)
    assert set(np.unique(y)) == set(range(len(ed.GESTURES)))
    assert np.unique(r).tolist() == [0, 1]
    x2, _, _ = ed.synthesize(fs=1000, n_ch=4, reps=2, hold_s=0.5, rest_s=0.2)
    np.testing.assert_array_equal(x, x2)  # seeded -> deterministic


def test_load_csv_separates_channels_from_metadata(tmp_path):
    p = tmp_path / "rec.csv"
    p.write_text("time,ch1,ch2,Label,rep\n0,1.0,2.0,0,0\n1,3.0,4.0,1,0\n2,5.0,6.0,1,1\n")
    x, lab, rep = ed.load_csv(str(p))
    np.testing.assert_array_equal(x, [[1, 2], [3, 4], [5, 6]])
    assert lab.tolist() == [0, 1, 1]
    assert rep.tolist() == [0, 0, 1]
    p2 = tmp_path / "nolabel.csv"
    p2.write_text("a,b\n1,2\n3,4\n")
    x, lab, rep = ed.load_csv(str(p2))
    assert x.shape == (2, 2) and lab is None and rep is None
