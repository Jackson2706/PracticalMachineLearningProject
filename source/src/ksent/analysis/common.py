"""Shared analysis helpers: locations, loading, number formatting, method styling."""
from __future__ import annotations

import re

import math
from pathlib import Path

import numpy as np
import pandas as pd

from ksent.config import (ALL_METHODS, DATASET_DISPLAY, FULL_SIZE, METHOD_DISPLAY, NEURAL_METHODS, TFIDF_METHODS,
                          global_cfg, path, resolve_sizes, size_label)

AGG_DIR: Path = path("results") / "aggregated"
REPORT_DIR: Path = path("report")
TABLE_DIR: Path = REPORT_DIR / "tables"
FIG_DIR: Path = REPORT_DIR / "figures"
GEN_DIR: Path = REPORT_DIR / "generated"
MISSING = "--"
_MISSING_CELL = re.compile(r"(?:^|&)\s*" + re.escape(MISSING) + r"\s*(?=&|\\\\)", re.M)


def count_missing(tex: str) -> int:
    """Number of table cells that are exactly the MISSING marker (not ``1--2-gram`` en-dashes)."""
    return len(_MISSING_CELL.findall(tex))

# Colour = method identity (reference palette, 8 validated slots + neutral grey + a light blue for BiRNN;
# unidirectional recurrent nets reuse their bidirectional twin's colour with a dotted line).
# Family is a second channel: TF-IDF dashed + hollow markers, neural solid + filled markers.
METHOD_STYLE: dict[str, dict] = {
    "tfidf_lr":          {"color": "#008300", "marker": "o"},
    "tfidf_svm":         {"color": "#e34948", "marker": "s"},
    "tfidf_rf":          {"color": "#eda100", "marker": "^"},
    "tfidf_xgb":         {"color": "#5f5e58", "marker": "D"},
    "emb_mlp":           {"color": "#1baf7a", "marker": "o"},
    "emb_rnn_uni":       {"color": "#6da7ec", "marker": "P"},
    "emb_lstm_uni":      {"color": "#2a78d6", "marker": "s"},
    "emb_gru_uni":       {"color": "#4a3aa7", "marker": "^"},
    "emb_rnn":           {"color": "#6da7ec", "marker": "P"},
    "emb_lstm":          {"color": "#2a78d6", "marker": "s"},
    "emb_gru":           {"color": "#4a3aa7", "marker": "^"},
    "small_transformer": {"color": "#eb6834", "marker": "D"},
    "small_mamba":       {"color": "#e87ba4", "marker": "v"},
}
for _m in TFIDF_METHODS:
    METHOD_STYLE[_m].update(linestyle="--", fill="none")
for _m in NEURAL_METHODS:
    METHOD_STYLE[_m].update(linestyle="-", fill="full")
for _m in ("emb_rnn_uni", "emb_lstm_uni", "emb_gru_uni"):  # same hue as the bidirectional twin, dotted
    METHOD_STYLE[_m].update(linestyle=":")


def datasets() -> list[str]:
    return list(global_cfg()["datasets"])


def sizes(dataset: str) -> list[int]:
    """Training sizes of ``dataset``, ascending, with ``full`` resolved to its pool size."""
    return resolve_sizes(dataset)


def _column_sizes() -> list[int]:
    """Numeric sizes used by any dataset (each dataset's ``full`` excluded), ascending."""
    return sorted({n for d in datasets() for n in sizes(d) if size_label(d, n) != FULL_SIZE})


def size_columns() -> list[str]:
    """Column labels shared by all datasets: the union of their numeric sizes, then ``full``."""
    return [f"{n:,}" for n in _column_sizes()] + [FULL_SIZE]


def aligned_sizes(dataset: str) -> list[int | None]:
    """``dataset``'s sizes laid out under ``size_columns()``; ``None`` where it has no such size."""
    own = sizes(dataset)
    full = [n for n in own if size_label(dataset, n) == FULL_SIZE]
    return [n if n in own else None for n in _column_sizes()] + [full[0] if full else None]


def ref_size() -> int:
    """Size of the cost pass (all cost metrics exist only here and at crossover sizes)."""
    return int(global_cfg()["analysis"]["reference_size"])


def size_tex(dataset: str, n) -> str:
    """LaTeX size label: ``8{,}000`` or ``full (95{,}774)``."""
    lab = size_label(dataset, int(n))
    return (f"full ({int(n):,})" if lab == FULL_SIZE else lab).replace(",", "{,}")


def methods() -> list[str]:
    return list(global_cfg()["methods"]) or list(ALL_METHODS)


def dname(d: str) -> str:
    return DATASET_DISPLAY.get(d, d)


def mname(m: str) -> str:
    return METHOD_DISPLAY.get(m, m)


def read_csv(name: str) -> pd.DataFrame | None:
    """Read ``results/aggregated/<name>`` or return ``None`` when absent/empty."""
    p = AGG_DIR / name
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return None
    return df if len(df) else None


def isnum(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x)) and not pd.isna(x)


def sig(x: float, n: int = 3) -> str:
    """Format with ``n`` significant figures, never in scientific notation."""
    if not isnum(x):
        return MISSING
    if x == 0:
        return "0"
    dec = max(0, n - 1 - int(math.floor(math.log10(abs(x)))))
    return f"{x:,.{dec}f}"


def _decimals_for_sig(x: float, n: int = 3) -> int:
    if not isnum(x) or x == 0:
        return 0
    return max(0, n - 1 - int(math.floor(math.log10(abs(x)))))


def pm(mean, std, scale: float = 1.0, dec: int | None = None, bold: bool = False, n_sig: int = 3) -> str:
    """``mean ± std`` in LaTeX. ``dec=None`` uses 3 significant figures of the mean
    (the std uses the same number of decimals)."""
    if not isnum(mean):
        return MISSING
    m = float(mean) * scale
    d = _decimals_for_sig(m, n_sig) if dec is None else dec
    ms = f"{m:,.{d}f}"
    if bold:
        ms = rf"\textbf{{{ms}}}"
    if not isnum(std):
        return ms
    return rf"{ms}{{\scriptsize$\pm${float(std) * scale:,.{d}f}}}"


def tex_escape(s: str) -> str:
    return (str(s).replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%")
            .replace("_", r"\_").replace("#", r"\#"))


def count_fmt(n) -> str:
    """Compact integer: 1,234 / 12.3k / 1.23M."""
    if not isnum(n):
        return MISSING
    n = float(n)
    if abs(n) >= 1e6:
        return f"{n / 1e6:.2f}M"
    if abs(n) >= 1e4:
        return f"{n / 1e3:.1f}k"
    return f"{n:,.0f}"


def summary_lookup(summary: pd.DataFrame | None, dataset: str, method: str, size: int, metric: str):
    """``(mean, std)`` of ``metric`` for one cell of the summary table (NaNs if absent)."""
    if summary is None or f"{metric}_mean" not in summary:
        return np.nan, np.nan
    row = summary[(summary.dataset == dataset) & (summary.method == method) & (summary["size"] == size)]
    if row.empty:
        return np.nan, np.nan
    return row[f"{metric}_mean"].iloc[0], row[f"{metric}_std"].iloc[0]
