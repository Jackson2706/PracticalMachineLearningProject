"""Verify the final sweep: every expected result file exists with ``status == "ok"``.

Usage: ``python -m ksent.analysis.check_results`` (exit code 1 if incomplete)
"""
from __future__ import annotations

import argparse

from ksent.config import global_cfg, main_tokenizer, path
from ksent.run_one import is_done, result_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ablation", action="store_true", help="check the 120 ablation runs instead")
    args = ap.parse_args()
    g = global_cfg()
    root = path("results")
    if args.ablation:
        from ksent.analysis.ablation import ablation_methods

        ab = g["ablation"]
        expected = [(d, m, t, n, s) for d, ms in ablation_methods().items() for m in ms for t in ab["tokenizers"]
                    if t != main_tokenizer() for n in ab["sizes"] for s in g["seeds"]]
    else:
        expected = [(d, m, main_tokenizer(), n, s) for d in g["datasets"] for m in g["methods"] for n in g["sizes"]
                    for s in g["seeds"]]
    missing, failed = [], []
    for d, m, t, n, s in expected:
        p = result_path(root, "final", d, m, t, n, s, None)
        if not is_done(p):
            (failed if p.with_suffix(".failed.json").exists() else missing).append(f"{d} {m} {t} n={n} seed={s}")
    if not missing and not failed:
        from ksent.analysis.aggregate import hardware_check, load_runs

        hardware_check([load_runs()], allow_mixed=False)
        print("single hardware fingerprint: OK")
    print(f"expected={len(expected)}  ok={len(expected) - len(missing) - len(failed)}  "
          f"failed={len(failed)}  missing={len(missing)}")
    for x in failed[:50]:
        print("  FAILED ", x)
    for x in missing[:20]:
        print("  missing", x)
    return 0 if not missing and not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
