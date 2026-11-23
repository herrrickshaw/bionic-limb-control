"""Boundary and exact-value tests for ``emg_decoder`` (offline, deterministic, no hardware)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy import signal

import emg_decoder as ed


def gain(sos, freqs, fs):
    _, h = signal.sosfreqz(sos, worN=freqs, fs=fs)
    return np.abs(h)


class TestHelpers:
    def test_samples_truncates(self):
        assert ed.samples(1000, 200) == 200
        assert ed.samples(2000, 12.7) == 25  # 25.4 -> truncated, not rounded
        assert ed.samples(1000, 0.9) == 0

    @pytest.mark.parametrize("cls,name", [(0, "rest"), (1, "power_grip"), (5, "wrist_pronate"), (6, "6"), (99, "99")])
    def test_gesture_name(self, cls, name):
        assert ed.gesture_name(cls) == name


class TestFilters:
    @pytest.mark.parametrize("fs,mains,sections", [
        (2000, 50, 7),    # band-pass (4) + notches at 50/100/150
        (1000, 50, 7),
        (500, 50, 7),     # hi capped at 225 Hz: 150 < 225 still notched
        (500, 100, 6),    # 100, 200 notched; 300 >= 225 skipped
        (500, 225, 4),    # f0 == hi is NOT notched (strict <)
        (2000, 200, 6),   # 200, 400 notched; 600 >= 450 skipped
    ])
    def test_notch_count(self, fs, mains, sections):
        assert ed.design_filters(fs, mains=mains).shape == (sections, 6)

    def test_frequency_response(self):
        sos = ed.design_filters(2000)
        g = gain(sos, [5, 50, 100, 150, 120], 2000)
        assert g[0] < 0.01            # motion artefact rejected
        assert g[1] < 1e-6 and g[2] < 1e-6 and g[3] < 1e-6  # mains + harmonics notched
        assert g[4] > 0.95            # in-band passes

    def test_mains_argument_moves_notch(self):
        g50 = gain(ed.design_filters(2000, mains=50), [50, 60], 2000)
        g60 = gain(ed.design_filters(2000, mains=60), [50, 60], 2000)
        assert g50[0] < 1e-6 < 0.9 < g50[1]
        assert g60[1] < 1e-6 < 0.9 < g60[0]

    def test_band_argument_and_upper_edge_cap(self):
        sos = ed.design_filters(1000, band=(20.0, 400.0))
        assert gain(sos, [380], 1000)[0] > 0.7
        # 450 Hz requested at fs=600 is capped to 0.45*fs = 270 Hz
        capped = ed.design_filters(600, band=(20.0, 450.0))
        assert gain(capped, [250], 600)[0] > 0.7 > gain(capped, [290], 600)[0]

    def test_preprocess_removes_dc_and_mains_keeps_band(self):
        fs = 2000
        t = np.arange(4000) / fs
        x = (5.0 + np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t))[:, None]
        y = ed.preprocess(x, fs)[500:-500, 0]
        assert np.sqrt(np.mean(y**2)) == pytest.approx(1 / np.sqrt(2), rel=0.05)

    def test_preprocess_honours_mains(self):
        fs = 2000
        t = np.arange(4000) / fs
        x = np.sin(2 * np.pi * 60 * t)[:, None]
        rms50 = np.sqrt(np.mean(ed.preprocess(x, fs, mains=50)[500:-500] ** 2))
        rms60 = np.sqrt(np.mean(ed.preprocess(x, fs, mains=60)[500:-500] ** 2))
        assert rms50 > 0.65 and rms60 < 0.05

    def test_preprocess_is_zero_phase(self):
        fs = 2000
        t = np.arange(4000) / fs
        x = np.sin(2 * np.pi * 120 * t)[:, None]
        y = ed.preprocess(x, fs)
        assert np.corrcoef(x[500:-500, 0], y[500:-500, 0])[0, 1] > 0.999

    def test_causal_filter_streaming_matches_one_shot(self):
        rng = np.random.default_rng(3)
        x = rng.standard_normal((500, 2))
        whole = ed.CausalFilter(1000, 2)(x)
        f = ed.CausalFilter(1000, 2)
        parts = np.vstack([f(x[i:i + 37]) for i in range(0, 500, 37)])
        np.testing.assert_allclose(parts, whole, atol=1e-12)

    def test_causal_filter_state_shape_and_carry(self):
        f = ed.CausalFilter(2000, 3, mains=60)
        assert f.zi.shape == (f.sos.shape[0], 2, 3)
        before = f.zi.copy()
        f(np.ones((10, 3)))
        assert not np.array_equal(before, f.zi)
        assert f.sos.shape == ed.design_filters(2000, mains=60).shape


class TestFeatures:
    def test_exact_values_alternating_signal(self):
        win = np.array([[1.0], [-1.0], [1.0], [-1.0]])
        mav, rms, wl, zc, ssc, logvar = ed.td_features(win)
        assert mav == 1.0 and rms == 1.0 and wl == 6.0 and zc == 3 and ssc == 2
        assert logvar == pytest.approx(0.0, abs=1e-9)

    def test_feature_major_layout_and_length(self):
        win = np.column_stack([np.full(10, 2.0), np.full(10, -3.0)])
        f = ed.td_features(win)
        assert f.shape == (12,)
        assert f[:2].tolist() == [2.0, 3.0]         # MAV per channel
        assert f[2:4].tolist() == [2.0, 3.0]        # RMS
        assert f[4:6].tolist() == [0.0, 0.0]        # WL
        assert f[6:8].tolist() == [0, 0] and f[8:10].tolist() == [0, 0]

    def test_rms_differs_from_mav(self):
        win = np.array([[0.0], [3.0], [0.0], [3.0]])
        f = ed.td_features(win)
        assert f[0] == 1.5 and f[1] == pytest.approx(np.sqrt(4.5))

    def test_zc_threshold_is_inclusive(self):
        win = np.array([[1.0], [-1.0]])
        assert ed.td_features(win, thresh=2.0)[3] == 1
        assert ed.td_features(win, thresh=2.0001)[3] == 0

    def test_zc_requires_sign_change(self):
        win = np.array([[1.0], [2.0], [3.0]])
        assert ed.td_features(win, thresh=0.0)[3] == 0

    def test_ssc_counts_when_either_step_exceeds_threshold(self):
        win = np.array([[0.0], [1.0], [0.999]])  # d = [1, -0.001]: slope sign change
        assert ed.td_features(win, thresh=0.5)[4] == 1   # first step big, second tiny
        assert ed.td_features(win, thresh=2.0)[4] == 0   # neither big enough
        win2 = np.array([[0.0], [0.001], [-1.0]])         # d = [0.001, -1.001]
        assert ed.td_features(win2, thresh=0.5)[4] == 1  # second step big, first tiny

    def test_default_threshold_value(self):
        win = np.array([[0.0], [0.02], [0.0]])           # steps of 0.02 >= 0.01
        assert ed.td_features(win)[4] == 1
        win = np.array([[0.0], [0.005], [0.0]])          # below the dead-band
        assert ed.td_features(win)[4] == 0

    def test_logvar_floor_for_constant_signal(self):
        f = ed.td_features(np.ones((8, 1)))
        assert f[5] == pytest.approx(np.log(1e-12))

    def test_mean_frequency_tracks_sine(self):
        fs = 1000
        t = np.arange(1000) / fs
        win = np.column_stack([np.sin(2 * np.pi * 100 * t), np.sin(2 * np.pi * 200 * t)])
        mf = ed.mean_frequency(win, fs)
        assert mf[0] == pytest.approx(100, abs=3) and mf[1] == pytest.approx(200, abs=3)

    def test_mean_frequency_short_window_uses_window_length(self):
        fs = 1000
        t = np.arange(100) / fs
        mf = ed.mean_frequency(np.sin(2 * np.pi * 100 * t)[:, None], fs)
        assert mf.shape == (1,) and 50 < mf[0] < 150

    def test_mean_frequency_short_window_emits_no_warning(self):
        import warnings
        fs = 1000
        t = np.arange(100) / fs
        with warnings.catch_warnings():
            warnings.simplefilter("error")      # scipy warns when nperseg exceeds the window length
            ed.mean_frequency(np.sin(2 * np.pi * 100 * t)[:, None], fs)

    def test_sliding_windows_boundaries(self):
        assert list(ed.sliding_windows(199, 1000)) == []
        assert list(ed.sliding_windows(200, 1000)) == [(0, 200)]
        wins = list(ed.sliding_windows(1000, 1000))
        assert wins[0] == (0, 200) and wins[-1] == (800, 1000) and len(wins) == 17
        assert list(ed.sliding_windows(100, 1000, win_ms=40, step_ms=30)) == [(0, 40), (30, 70), (60, 100)]

    def test_extract_labels_reps_come_from_last_sample(self):
        x = np.arange(400, dtype=float)[:, None]
        labels = np.arange(400)
        reps = np.arange(400) // 100
        X, y, r = ed.extract(x, 1000, labels, reps)
        assert X.shape == (len(y), 6) and y.tolist()[:2] == [199, 249] and r.tolist()[:2] == [1, 2]

    def test_extract_without_labels_or_reps(self):
        X, y, r = ed.extract(np.zeros((300, 2)), 1000)
        assert X.shape[1] == 12 and y is None and r is None

    def test_extract_forwards_threshold_and_window_args(self):
        x = np.tile(np.array([[1.0], [-1.0]]), (200, 1))
        lo, _, _ = ed.extract(x, 1000, thresh=0.5)
        hi, _, _ = ed.extract(x, 1000, thresh=5.0)
        assert lo[0, 3] > 0 and hi[0, 3] == 0
        X, _, _ = ed.extract(x, 1000, win_ms=100, step_ms=100)
        assert len(X) == 4


class FakeModel:
    def __init__(self, probas, classes):
        self._p = iter(probas)
        self.classes_ = np.array(classes)

    def predict_proba(self, f):
        return np.array([next(self._p)])


def make_decoder(probas, classes=(0, 1, 2), vote=3, rest=0.1, mvc=0.9, n_ch=2):
    dec = ed.PatternRecognitionDecoder(1000, n_ch, vote=vote)
    dec.model = FakeModel(probas, classes)
    dec.rest_mav, dec.mvc_mav, dec.thresh = rest, mvc, 0.01
    return dec


class TestDecoder:
    def test_constructor_defaults(self):
        dec = ed.PatternRecognitionDecoder(1000, 4)
        assert (dec.win_ms, dec.step_ms, dec._votes.maxlen) == (200, 50, 5)
        assert dec.thresh == ed.DEFAULT_ZC_THRESH and dec.rest_mav is None and dec.mvc_mav is None
        assert ed.PatternRecognitionDecoder(1000, 4, vote=2)._votes.maxlen == 2

    def test_fit_returns_self_and_calibrates(self):
        x, y, r = ed.synthesize(fs=1000, n_ch=3, reps=2, hold_s=1.0, rest_s=0.5)
        xf = ed.preprocess(x, 1000)
        dec = ed.PatternRecognitionDecoder(1000, 3)
        assert dec.fit(xf, y) is dec
        assert dec.thresh == pytest.approx(3 * np.std(xf[y == 0]))
        assert 0 <= dec.rest_mav < dec.mvc_mav

    def test_fit_calibrates_with_95th_percentile_of_window_mav(self):
        fs, n_ch = 1000, 2
        t = np.arange(4000) / fs
        rest = 0.01 * (1 + 4 * t / t[-1])[:, None] * np.sin(2 * np.pi * 80 * t)[:, None] * np.ones(n_ch)
        act = 0.5 * (1 + t / t[-1])[:, None] * np.sin(2 * np.pi * 120 * t)[:, None] * np.ones(n_ch)
        x = np.vstack([rest, act])
        y = np.r_[np.zeros(len(rest), dtype=int), np.ones(len(act), dtype=int)]
        dec = ed.PatternRecognitionDecoder(fs, n_ch)
        dec.model = SimpleNamespace(fit=lambda X, y_: None)
        dec.fit(x, y)
        X, yw, _ = ed.extract(x, fs, y, None, dec.win_ms, dec.step_ms, dec.thresh)
        mav = X[:, :n_ch].mean(1)
        assert dec.rest_mav == pytest.approx(np.percentile(mav[yw == 0], 95))
        assert dec.rest_mav > np.percentile(mav[yw == 0], 80)       # a lower percentile would fail here
        assert dec.mvc_mav == pytest.approx(np.percentile(mav[yw != 0], 95))

    def test_fit_without_rest_uses_default_threshold(self):
        x, y, r = ed.synthesize(fs=1000, n_ch=2, reps=2, hold_s=1.0, rest_s=0.5)
        m = y != 0
        dec = ed.PatternRecognitionDecoder(1000, 2).fit(ed.preprocess(x, 1000)[m], y[m])
        assert dec.thresh == ed.DEFAULT_ZC_THRESH
        assert dec.rest_mav == 0.0

    def test_fit_rest_only_uses_unit_mvc(self):
        rng = np.random.default_rng(0)
        x = 0.01 * rng.standard_normal((1000, 2))
        dec = ed.PatternRecognitionDecoder(1000, 2)
        dec.model = SimpleNamespace(fit=lambda X, y: None)
        dec.fit(x, np.zeros(1000, dtype=int))
        assert dec.mvc_mav == 1.0 and dec.rest_mav > 0

    def test_predict_windows_shape(self):
        x, y, r = ed.synthesize(fs=1000, n_ch=3, reps=2, hold_s=1.0, rest_s=0.5)
        xf = ed.preprocess(x, 1000)
        dec = ed.PatternRecognitionDecoder(1000, 3).fit(xf, y)
        assert dec.predict_windows(xf).shape == (len(list(ed.sliding_windows(len(xf), 1000))),)

    def test_decide_speed_and_confidence(self):
        dec = make_decoder([[0.1, 0.9, 0.0]])
        cmd = dec._decide(np.full((10, 2), 0.5))
        assert cmd.gesture == "power_grip"
        assert cmd.speed == pytest.approx(0.5)
        assert cmd.confidence == 1.0

    def test_decide_speed_clipped_to_unit_interval(self):
        dec = make_decoder([[0, 1, 0], [0, 1, 0]], rest=0.1, mvc=0.3)
        assert dec._decide(np.full((10, 2), 5.0)).speed == 1.0
        dec = make_decoder([[0, 1, 0]], rest=0.4, mvc=0.9)
        cmd = dec._decide(np.full((10, 2), 0.2))   # below rest level -> clipped to 0 -> rest
        assert cmd.gesture == "rest" and cmd.speed == 0.0 and cmd.confidence == 1.0

    def test_decide_rest_vote_reports_proba_max(self):
        dec = make_decoder([[0.7, 0.2, 0.1]])
        cmd = dec._decide(np.full((10, 2), 0.5))
        assert cmd == ed.Command("rest", 0.0, pytest.approx(0.7))

    def test_decide_majority_vote_and_confidence(self):
        dec = make_decoder([[0, 1, 0], [0, 0, 1], [0, 0, 1]], vote=3)
        assert dec._decide(np.full((10, 2), 0.5)).gesture == "power_grip"
        c2 = dec._decide(np.full((10, 2), 0.5))
        assert c2.gesture == "power_grip"           # tie 1:1 -> earliest class wins
        c3 = dec._decide(np.full((10, 2), 0.5))
        assert c3.gesture == "open_hand" and c3.confidence == pytest.approx(2 / 3)

    def test_vote_window_is_bounded(self):
        dec = make_decoder([[0, 1, 0]] * 2 + [[0, 0, 1]] * 3, vote=2)
        for _ in range(5):
            dec._decide(np.full((10, 2), 0.5))
        assert list(dec._votes) == [2, 2]

    def test_decide_uses_class_labels_not_indices(self):
        dec = make_decoder([[0.0, 1.0]], classes=(0, 3))
        assert dec._decide(np.full((10, 2), 0.5)).gesture == "pinch"

    def test_decide_channel_mean_uses_only_mav_block(self):
        dec = make_decoder([[0, 1, 0]], rest=0.0, mvc=1.0, n_ch=2)
        win = np.column_stack([np.full(10, 0.2), np.full(10, 0.6)])
        assert dec._decide(win).speed == pytest.approx(0.4)

    def test_stream_chunking_and_buffer_cap(self):
        dec = ed.PatternRecognitionDecoder(1000, 1)
        seen = []
        dec._decide = lambda win: seen.append(win.shape) or ed.Command("rest", 0.0, 1.0)
        out = list(dec.stream([np.zeros((100, 1)), np.zeros((100, 1))]))
        # nothing until the 200-sample window fills, then pending=200 -> four hops of 50
        assert len(out) == 4 and seen == [(200, 1)] * 4
        seen.clear()
        assert len(list(dec.stream([np.zeros((400, 1))]))) == 8
        assert seen == [(200, 1)] * 8
        seen.clear()
        assert list(dec.stream([np.zeros((49, 1))])) == []

    def test_stream_waits_for_full_window(self):
        dec = ed.PatternRecognitionDecoder(1000, 1)
        dec._decide = lambda win: ed.Command("rest", 0.0, 1.0)
        assert list(dec.stream([np.zeros((150, 1))])) == []
        assert len(list(dec.stream([np.zeros((150, 1)), np.zeros((50, 1))]))) == 4


class TestDirectControlExact:
    def test_constructor_derived_values(self):
        dc = ed.DirectControl(1000, on_thresh=0.2, cocon_ms=100, refractory_ms=250)
        assert (dc.on, dc.cocon_n, dc.refr_n, dc.grip) == (0.2, 100, 250, 0)
        assert dc.mvc.tolist() == [1.0, 1.0]

    def test_velocity_formula(self):
        dc = ed.DirectControl(1000)
        assert dc.step(0.575, 0.0)[1] == pytest.approx(0.5)
        assert dc.step(0.0, 0.575)[1] == pytest.approx(-0.5)
        assert dc.step(0.575, 0.1)[1] == pytest.approx(0.5)   # sub-threshold extensor contributes nothing
        assert dc.step(0.1, 0.575)[1] == pytest.approx(-0.5)

    def test_on_threshold_is_strict(self):
        dc = ed.DirectControl(1000)
        assert dc.step(0.15, 0.0) == ("power", 0.0)
        assert dc.step(0.15, 0.15) == ("power", 0.0)
        assert dc._cocon == 0  # both exactly at threshold: not a co-contraction

    def test_cocontraction_counter_resets_on_break(self):
        dc = ed.DirectControl(1000, cocon_ms=10, refractory_ms=5)
        for _ in range(9):
            dc.step(0.5, 0.5)
        dc.step(0.5, 0.0)        # break the streak
        for _ in range(9):
            assert dc.step(0.5, 0.5)[0] == "power"
        assert dc.step(0.5, 0.5)[0] == "tripod"

    def test_refractory_counts_down_exactly(self):
        dc = ed.DirectControl(1000, cocon_ms=2, refractory_ms=3)
        dc.step(0.5, 0.5)
        assert dc.step(0.5, 0.5)[0] == "tripod"
        assert [dc.step(1.0, 0.0)[1] for _ in range(4)] == [0.0, 0.0, 0.0, pytest.approx(1.0)]

    def test_grip_wraps_after_all_grips(self):
        dc = ed.DirectControl(1000, cocon_ms=1, refractory_ms=1)
        seen = []
        for _ in range(len(dc.GRIPS)):
            seen.append(dc.step(0.5, 0.5)[0])
            dc.step(0.0, 0.0)
        assert seen == dc.GRIPS[1:] + dc.GRIPS[:1]

    def test_calibrate_uses_95th_percentile_per_channel(self):
        dc = ed.DirectControl(1000)
        env = np.column_stack([np.linspace(0, 1, 101), np.linspace(0, 2, 101)])
        dc.calibrate(env)
        assert dc.mvc == pytest.approx([0.95, 1.9])

    def test_envelope_is_stateful_across_calls(self):
        a = ed.DirectControl(1000)
        b = ed.DirectControl(1000)
        x = np.abs(np.random.default_rng(1).standard_normal((600, 2)))
        whole = a.envelope(x)
        split = np.vstack([b.envelope(x[:250]), b.envelope(x[250:])])
        np.testing.assert_allclose(split, whole, atol=1e-12)

    def test_envelope_normalises_by_mvc_per_channel(self):
        dc = ed.DirectControl(1000)
        dc.mvc = np.array([10.0, 20.0])
        env = dc.envelope(np.full((3000, 2), 5.0))
        assert env[-1] == pytest.approx([0.5, 0.25], abs=1e-3)

    def test_envelope_rectifies(self):
        dc = ed.DirectControl(1000)
        env = dc.envelope(np.full((3000, 2), -0.5))
        assert env[-1] == pytest.approx([0.5, 0.5], abs=1e-3)


class TestSynthetic:
    def test_rest_is_quieter_than_active_and_reps_complete(self):
        x, y, r = ed.synthesize(fs=1000, n_ch=3, reps=3, hold_s=1.0, rest_s=0.5, seed=4)
        xf = ed.preprocess(x, 1000)
        assert np.std(xf[y == 0]) < np.std(xf[y != 0])
        for rep in range(3):
            assert set(np.unique(y[r == rep])) == set(range(len(ed.GESTURES)))

    def test_seed_changes_output(self):
        a = ed.synthesize(fs=1000, n_ch=2, reps=1, hold_s=0.3, rest_s=0.2, seed=0)[0]
        b = ed.synthesize(fs=1000, n_ch=2, reps=1, hold_s=0.3, rest_s=0.2, seed=1)[0]
        assert not np.array_equal(a, b)

    def test_mains_hum_present_at_requested_frequency(self):
        x, _, _ = ed.synthesize(fs=1000, n_ch=1, reps=1, hold_s=0.5, rest_s=0.5, mains=60.0)
        spec = np.abs(np.fft.rfft(x[:, 0] - x[:, 0].mean()))
        freqs = np.fft.rfftfreq(len(x), 1 / 1000)
        band = (freqs > 55) & (freqs < 65)
        assert freqs[band][np.argmax(spec[band])] == pytest.approx(60, abs=1.0)
        assert spec[band].max() > 5 * np.median(spec)

    def test_split_by_rep_n_test_and_noncontiguous_ids(self):
        reps = np.array([10, 10, 20, 20, 30, 30, 40, 40])
        tr, te, ids = ed.split_by_rep(reps, n_test=3)
        assert ids.tolist() == [20, 30, 40] and tr.tolist() == [True, True] + [False] * 6
        assert ed.split_by_rep(reps)[2].tolist() == [30, 40]


class TestCsvAndRunners:
    def test_load_csv_case_and_whitespace_insensitive(self, tmp_path):
        p = tmp_path / "r.csv"
        p.write_text(" CH1 , ch2 , REP , Label ,Timestamp\n1,2,0,1,0.0\n3,4,1,2,0.1\n")
        x, lab, rep = ed.load_csv(str(p))
        np.testing.assert_array_equal(x, [[1, 2], [3, 4]])
        assert lab.tolist() == [1, 2] and rep.tolist() == [0, 1]
        assert lab.dtype.kind == "i" and rep.dtype.kind == "i"

    @pytest.mark.xfail(strict=True, raises=IndexError,
                       reason="load_csv: np.genfromtxt returns a 1-D array for a single-column CSV, "
                              "so data[:, ch] raises IndexError instead of returning an (n, 1) channel matrix")
    def test_load_csv_single_channel_file(self, tmp_path):
        p = tmp_path / "one.csv"
        p.write_text("ch1\n1.0\n2.0\n3.0\n")
        x, lab, rep = ed.load_csv(str(p))
        assert x.shape == (3, 1) and lab is None and rep is None

    def test_contiguous_reps_custom_count(self):
        r = ed.contiguous_reps(10, n_reps=2)
        assert r.tolist() == [0] * 5 + [1] * 5
        assert ed.contiguous_reps(7, n_reps=7).tolist() == list(range(7))

    def test_run_csv_without_rep_column_falls_back_to_contiguous(self, tmp_path, monkeypatch, capsys):
        x, y, _ = ed.synthesize(fs=1000, n_ch=2, reps=5, hold_s=0.6, rest_s=0.3)
        p = tmp_path / "rec.csv"
        np.savetxt(p, np.column_stack([x, y]), delimiter=",", header="a,b,label", comments="")
        seen = {}

        def fake_eval(x_, lab, rep, fs, mains):
            seen.update(rep=rep, fs=fs, mains=mains)
            return SimpleNamespace()

        monkeypatch.setattr(ed, "evaluate", fake_eval)
        monkeypatch.setattr(ed, "demo_stream", lambda dec, x_, lab, fs: seen.update(n=len(x_)))
        ed.run_csv(str(p), 1000, 60)
        assert seen["rep"].tolist() == ed.contiguous_reps(len(x)).tolist()
        assert seen["fs"] == 1000 and seen["mains"] == 60
        assert seen["n"] == int((seen["rep"] == seen["rep"].max()).sum())

    def test_run_csv_unlabelled_stats(self, tmp_path, capsys):
        p = tmp_path / "rec.csv"
        t = np.arange(2000) / 1000
        sig = np.sin(2 * np.pi * 120 * t)
        np.savetxt(p, np.column_stack([sig, sig]), delimiter=",", header="ch1,ch2", comments="")
        ed.run_csv(str(p), 1000, 50)
        out = capsys.readouterr().out
        assert "per-channel RMS [0.697 0.697]" in out
        assert "Mean frequency (Hz): [120. 120.]" in out

    def test_run_demo_wires_synthesis_evaluation_and_direct(self, monkeypatch, capsys):
        orig = ed.synthesize
        calls = {}
        monkeypatch.setattr(ed, "synthesize", lambda fs, mains: orig(fs=fs, n_ch=3, reps=4, hold_s=0.6,
                                                                     rest_s=0.3, mains=mains))
        monkeypatch.setattr(ed, "evaluate", lambda x, lab, rep, fs, mains: calls.update(ev=(fs, mains)) or "dec")
        monkeypatch.setattr(ed, "demo_stream", lambda dec, x, lab, fs: calls.update(stream=(dec, len(x))))
        monkeypatch.setattr(ed, "demo_direct", lambda: calls.update(direct=True))
        ed.run_demo(1000, 60)
        out = capsys.readouterr().out
        assert out.startswith("Synthetic: 3 channels, ") and "6 classes" in out
        assert calls["ev"] == (1000, 60) and calls["direct"] is True and calls["stream"][0] == "dec"

    def test_main_dispatch(self, monkeypatch):
        rec = []
        monkeypatch.setattr(ed, "run_csv", lambda *a: rec.append(("csv", a)))
        monkeypatch.setattr(ed, "run_demo", lambda *a: rec.append(("demo", a)))
        ed.main([])
        ed.main(["--demo", "--fs", "500", "--mains", "60"])
        ed.main(["--csv", "x.csv", "--fs", "250", "--mains", "60"])
        assert rec == [("demo", (2000.0, 50.0)), ("demo", (500.0, 60.0)), ("csv", ("x.csv", 250.0, 60.0))]

    def test_parser_choices_and_types(self):
        a = ed.build_parser().parse_args(["--fs", "1234.5", "--csv", "f.csv"])
        assert a.fs == 1234.5 and a.csv == "f.csv"


class TestDemos:
    def test_demo_stream_prints_every_20th_and_agreement(self, capsys):
        cmds = [ed.Command("rest", 0.0, 1.0)] * 41 + [ed.Command("power_grip", 0.5, 1.0)] * 4
        dec = SimpleNamespace(step_ms=50, win_ms=200, stream=lambda chunks: iter(cmds))
        labels = np.zeros(10_000, dtype=int)
        ed.demo_stream(dec, np.zeros((10_000, 2)), labels, 1000)
        out = capsys.readouterr().out.splitlines()
        rows = [ln for ln in out if ln.startswith("  t=")]
        assert len(rows) == 3
        assert rows[0].startswith("  t=  0.20s  true=rest") and "speed=0.00" in rows[0]
        assert rows[1].startswith("  t=  1.20s")      # idx = 199 + 20*50 = 1199
        assert "Online agreement with labels: 91.1% over 45 decisions" in "\n".join(out)

    def test_demo_stream_clamps_truth_index_to_last_label(self, capsys):
        cmds = [ed.Command("point", 0.1, 1.0)] * 3
        dec = SimpleNamespace(step_ms=50, win_ms=200, stream=lambda chunks: iter(cmds))
        labels = np.full(50, 4)  # shorter than first window index: clamp to labels[-1] ('point')
        ed.demo_stream(dec, np.zeros((50, 1)), labels, 1000)
        assert "100.0% over 3 decisions" in capsys.readouterr().out

    def test_demo_stream_chunks_cover_recording(self):
        got = []

        def stream(chunks):
            got.extend(chunks)
            return iter([ed.Command("rest", 0, 1)])

        dec = SimpleNamespace(step_ms=50, win_ms=200, stream=stream)
        ed.demo_stream(dec, np.zeros((105, 1)), np.zeros(105, dtype=int), 1000, chunk_ms=10)
        assert len(got) == 10 and all(c.shape == (10, 1) for c in got)

    def test_demo_direct_reports_each_segment(self, capsys):
        ed.demo_direct()
        lines = [ln for ln in capsys.readouterr().out.splitlines() if "flexor=" in ln]
        assert len(lines) == 7
        grips = [ln.split("grip=")[1].split()[0] for ln in lines]
        vel = [float(ln.rsplit("=", 1)[1]) for ln in lines]
        assert grips == ["power"] * 4 + ["tripod"] * 3
        assert vel[1] > 0.8 and vel[3] < -0.8 and abs(vel[0]) < 0.05 and abs(vel[4]) < 0.05
