from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

import emg_decoder as ed

ROOT = Path(__file__).resolve().parents[1]


def test_help_exits_zero():
    r = subprocess.run([sys.executable, str(ROOT / "emg_decoder.py"), "--help"], capture_output=True, text=True)
    assert r.returncode == 0
    assert "--mains" in r.stdout


def test_parser_defaults():
    a = ed.build_parser().parse_args([])
    assert a.fs == ed.DEFAULT_FS and a.mains == ed.DEFAULT_MAINS and not a.demo and a.csv is None


def test_main_csv_without_labels_prints_stats(tmp_path, capsys):
    rng = np.random.default_rng(0)
    p = tmp_path / "rec.csv"
    rows = ["ch1,ch2"] + [f"{a},{b}" for a, b in rng.standard_normal((1000, 2))]
    p.write_text("\n".join(rows) + "\n")
    ed.main(["--csv", str(p), "--fs", "1000"])
    out = capsys.readouterr().out
    assert "No labels" in out and "Mean frequency" in out


def test_main_csv_with_labels_runs_evaluation_and_replay(tmp_path, capsys):
    x, y, r = ed.synthesize(fs=1000, n_ch=3, reps=4, hold_s=1.0, rest_s=0.5)
    p = tmp_path / "rec.csv"
    body = np.column_stack([x, y, r])
    np.savetxt(p, body, delimiter=",", header="c1,c2,c3,label,rep", comments="")
    ed.main(["--csv", str(p), "--fs", "1000"])
    out = capsys.readouterr().out
    assert "Online agreement with labels" in out
