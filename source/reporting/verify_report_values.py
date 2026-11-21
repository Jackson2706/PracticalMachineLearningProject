from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from reporting_utils import DATASETS, REPORTING_RESULTS, ROOT, SUBMISSION


TRACE_PATH = REPORTING_RESULTS / "displayed_value_traceability.json"
RESULT_PATTERN = re.compile(r"\\Result\{([^}]+)\}")
PLACEHOLDER_PATTERN = re.compile(r"TODO|\[VALUE\]|\[CITATION\]|\bXXX\b|\bNaN\b", re.IGNORECASE)


def _documents() -> list[Path]:
    paths: list[Path] = []
    for directory in (SUBMISSION / "report", SUBMISSION / "presentation"):
        paths.extend(directory.rglob("*.tex"))
        paths.extend(directory.rglob("*.md"))
    paths.extend((SUBMISSION / "assets" / "tables").glob("*.tex"))
    return sorted(set(paths))


def _check_trace(trace: dict[str, dict]) -> list[str]:
    errors: list[str] = []
    used_ids: set[str] = set()
    for path in _documents():
        text = path.read_text(encoding="utf-8")
        used_ids.update(RESULT_PATTERN.findall(text))
        if match := PLACEHOLDER_PATTERN.search(text):
            errors.append(f"placeholder {match.group(0)!r} in {path.relative_to(ROOT)}")
        if "/home/" in text or "/Users/" in text:
            errors.append(f"absolute local path in {path.relative_to(ROOT)}")
    missing = sorted(used_ids - trace.keys())
    if missing:
        errors.append(f"untraced Result macros: {', '.join(missing)}")
    for value_id, row in trace.items():
        source = ROOT / row["source_file"]
        if not source.exists():
            errors.append(f"{value_id}: missing source {row['source_file']}")
        value = row.get("value")
        if isinstance(value, float) and not math.isfinite(value):
            if not str(row.get("display", "")).lower().startswith(("not ", "missing")):
                errors.append(f"{value_id}: non-finite value is not visibly marked")
    return errors


def _check_generated_tables(trace: dict[str, dict]) -> list[str]:
    expected_prefix_counts = {
        "table_dataset_": 21,
        "full_": 27,
        "table_crossover_": 6,
        "table_efficiency_": 30,
        "table_bonus_nsmc_": 20,
        "table_bonus_shopping_": 20,
        "table_bonus_steam_": 20,
        "table_bonus_labels_": 24,
        "table_bonus_diagnostics_": 18,
        "table_completeness_": 5,
        "table_config_": 27,
    }
    errors = []
    for prefix, expected in expected_prefix_counts.items():
        actual = sum(value_id.startswith(prefix) for value_id in trace)
        if actual != expected:
            errors.append(f"trace prefix {prefix}: expected {expected}, found {actual}")
    return errors


def _check_cross_sources(trace: dict[str, dict]) -> list[str]:
    errors: list[str] = []
    full = pd.read_csv(REPORTING_RESULTS / "derived_metrics" / "part1_full_data_ranking.csv")
    cross = pd.read_csv(REPORTING_RESULTS / "derived_metrics" / "part1_crossover_summary.csv")
    bonus = pd.read_csv(REPORTING_RESULTS / "derived_metrics" / "bonus_summary_corrected.csv")
    for dataset in DATASETS:
        ranked = full[full.dataset.eq(dataset)]
        neural = ranked[ranked.model_family.eq("neural")].sort_values("rank").iloc[0]
        tfidf = ranked[ranked.model_family.eq("tfidf")].sort_values("macro_f1_mean", ascending=False).iloc[0]
        checks = {
            f"story_full_best_neural_{dataset}": neural.macro_f1_mean,
            f"story_full_best_tfidf_{dataset}": tfidf.macro_f1_mean,
            f"story_full_gap_{dataset}": neural.macro_f1_mean - tfidf.macro_f1_mean,
        }
        crossover = cross[cross.dataset.eq(dataset)].iloc[0].first_sustained_crossover
        checks[f"part1_crossover_{dataset}"] = crossover
        cells = bonus[bonus.target_domain.eq(dataset) & bonus.target_train_size.eq(200)].set_index("method")
        checks.update({
            f"story_bonus_pool_vs_target_{dataset}_200": cells.loc["pooled_ce"].target_macro_f1_mean - cells.loc["target_transformer"].target_macro_f1_mean,
            f"story_bonus_pool_vs_tfidf_{dataset}_200": cells.loc["pooled_ce"].target_macro_f1_mean - cells.loc["best_tfidf"].target_macro_f1_mean,
            f"story_bonus_db_vs_pool_{dataset}_200": cells.loc["db_cdsupcon"].target_macro_f1_mean - cells.loc["pooled_ce"].target_macro_f1_mean,
        })
        for value_id, expected in checks.items():
            actual = trace[value_id]["value"]
            both_missing = pd.isna(actual) and pd.isna(expected)
            if not both_missing and not math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=1e-12):
                errors.append(f"{value_id}: trace {actual} != source {expected}")
    return errors


def _check_references() -> list[str]:
    bib = (SUBMISSION / "report" / "references.bib").read_text(encoding="utf-8")
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    cited: set[str] = set()
    for path in (SUBMISSION / "report").rglob("*.tex"):
        for group in re.findall(r"\\cite\w*\{([^}]+)\}", path.read_text(encoding="utf-8")):
            cited.update(key.strip() for key in group.split(","))
    missing = sorted(cited - keys)
    return [f"unresolved citation keys: {', '.join(missing)}"] if missing else []


def _extract_pdf_text(path: Path) -> str:
    pdftotext = shutil.which("pdftotext")
    if pdftotext:
        with tempfile.NamedTemporaryFile(suffix=".txt") as output:
            subprocess.run([pdftotext, str(path), output.name], check=True)
            return Path(output.name).read_text(encoding="utf-8", errors="replace")
    gs = shutil.which("gs")
    if not gs:
        raise RuntimeError("neither pdftotext nor Ghostscript is available")
    with tempfile.NamedTemporaryFile(suffix=".txt") as output:
        subprocess.run(
            [gs, "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=txtwrite", f"-sOutputFile={output.name}", str(path)],
            check=True,
        )
        return Path(output.name).read_text(encoding="utf-8", errors="replace")


def _check_pdfs() -> list[str]:
    errors: list[str] = []
    checks = {
        SUBMISSION / "report" / "report.pdf": ("How Much Labeled Data", "Reproducibility"),
        SUBMISSION / "presentation" / "slides.pdf": ("How Much Labeled Data", "Four takeaways"),
    }
    for path, required in checks.items():
        if not path.exists() or path.stat().st_size == 0:
            errors.append(f"missing or empty PDF: {path.relative_to(ROOT)}")
            continue
        text = _extract_pdf_text(path)
        for phrase in required:
            if phrase not in text:
                errors.append(f"{path.name}: extracted text lacks {phrase!r}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-pdfs", action="store_true")
    args = parser.parse_args()
    if not TRACE_PATH.exists():
        raise SystemExit(f"missing {TRACE_PATH}; run build_analysis_tables.py first")
    trace = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
    errors = _check_trace(trace) + _check_generated_tables(trace) + _check_cross_sources(trace) + _check_references()
    if args.require_pdfs:
        errors += _check_pdfs()
    if errors:
        raise SystemExit("verification failed:\n- " + "\n- ".join(errors))
    print(f"verified {len(trace)} traced values, generated tables, citations" + (", and PDFs" if args.require_pdfs else ""))


if __name__ == "__main__":
    main()
