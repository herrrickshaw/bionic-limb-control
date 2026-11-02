"""End-to-end characterisation on a short synthetic recording (no hardware)."""
from __future__ import annotations

import pytest

import emg_decoder as ed

FS = 1000.0


@pytest.fixture(scope="module")
def recording():
    return ed.synthesize(fs=FS, n_ch=4, reps=4, hold_s=1.0, rest_s=0.5, seed=0)


def test_decoder_fit_predict_and_speed_calibration(recording):
    x, y, r = recording
    xf = ed.preprocess(x, FS)
    tr, te, _ = ed.split_by_rep(r)
    dec = ed.PatternRecognitionDecoder(FS, 4).fit(xf[tr], y[tr])
    assert dec.rest_mav < dec.mvc_mav
    assert dec.thresh > 0
    pred = dec.predict_windows(xf[te])
    _, y_te, _ = ed.extract(xf[te], FS, y[te])
    assert pred.shape == y_te.shape
    assert (pred == y_te).mean() > 0.8  # synthetic data is easy; guards against a broken pipeline


def test_stream_yields_commands_at_step_rate(recording):
    x, y, r = recording
    xf = ed.preprocess(x, FS)
    dec = ed.PatternRecognitionDecoder(FS, 4).fit(xf, y)
    last = x[r == r.max()]
    n = ed.samples(FS, 10)
    chunks = (last[i : i + n] for i in range(0, len(last) - n + 1, n))
    cmds = list(dec.stream(chunks))
    # Characterisation: one Command per step of input (the first window is
    # decided win_ms/step_ms times as soon as the buffer fills).
    assert len(cmds) == len(last) // ed.samples(FS, dec.step_ms)
    assert all(isinstance(c, ed.Command) for c in cmds)
    assert all(0.0 <= c.speed <= 1.0 and 0.0 < c.confidence <= 1.0 for c in cmds)
    assert any(c.gesture != "rest" for c in cmds)
    assert all(c.speed == 0.0 for c in cmds if c.gesture == "rest")


def test_evaluate_prints_report_and_returns_decoder(recording, capsys):
    x, y, r = recording
    dec = ed.evaluate(x, y, r, FS)
    out = capsys.readouterr().out
    assert isinstance(dec, ed.PatternRecognitionDecoder)
    assert "test reps [2, 3]" in out
    assert "Confusion matrix" in out
    assert "Mean frequency per channel" in out
