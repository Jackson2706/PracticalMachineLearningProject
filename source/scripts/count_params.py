#!/usr/bin/env python
"""Print the neural parameter-budget table and fail if any level is out of tolerance.

Usage:
  python scripts/count_params.py            # check configs/model_widths.yaml
  python scripts/count_params.py --write    # re-solve widths, write the YAML, then check
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ksent.config import NEURAL_METHODS, global_cfg  # noqa: E402
from ksent.models.registry import (  # noqa: E402
    build_neural, count_params, model_widths, solve_all_widths, write_widths, _fixed,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="re-solve widths and write configs/model_widths.yaml")
    ap.add_argument("--vocab", type=int, default=20002, help="vocab size used to show embedding params")
    args = ap.parse_args()
    if args.write:
        write_widths(solve_all_widths())
    nc = global_cfg()["neural"]
    targets, tol = nc["width_targets"], float(nc["param_tolerance"])
    msl = int((global_cfg().get("max_seq_len") or {}).get("kiwi") or 64)
    widths = model_widths()
    ok = True
    print(f"{'method':18s} {'lvl':3s} {'width':>5s} {'non-emb':>9s} {'target':>8s} {'dev':>7s} {'emb(V=%d)' % args.vocab:>12s}")
    for m in NEURAL_METHODS:
        for lvl, target in targets.items():
            w = widths[m][lvl]
            c = count_params(build_neural(m, args.vocab, w, 0.1, msl, _fixed(m)))
            dev = c["non_embedding"] / target - 1
            flag = "" if abs(dev) <= tol else "  <-- OUT OF TOLERANCE"
            ok &= abs(dev) <= tol
            print(f"{m:18s} {lvl:3s} {w:5d} {c['non_embedding']:9,d} {target:8,d} {dev:+7.1%} {c['embedding']:12,d}{flag}")
    print("OK: all levels within ±%.0f%%" % (tol * 100) if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
