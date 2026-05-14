#!/usr/bin/env python3
"""
Extract ESCO skills from an Excel file for multiple ESCOX thresholds (sensitivity analysis)
and write them into separate columns.

Example:
  python scripts/extract_escox_skills_from_modules.py \
    --input data/modules.xlsx \
    --output data/modules_with_escox_skills.xlsx \
    --thresholds 0.50 0.55 0.60 0.65 \
    --also-write-counts
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, List, Dict

import pandas as pd
from esco_skill_extractor import SkillExtractor


def normalize_text(x: Any) -> str:
    """Convert cell content to a clean string; return empty string if NaN/None."""
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    s = str(x).strip()
    return "" if s.lower() == "nan" else s


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks of size chunk_size."""
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def skills_to_cell(skill_uris: List[str]) -> str:
    """Excel-friendly: semicolon-separated ESCO skill URIs."""
    return "; ".join(skill_uris) if skill_uris else ""


def run_extraction_for_threshold(
    texts: List[str],
    threshold: float,
    batch_size: int,
    device: str | None,
) -> List[List[str]]:
    """Run ESCOX extraction for one threshold over all texts."""
    # IMPORTANT: the library expects "skills_threshold" (plural)
    extractor_kwargs = {"skills_threshold": threshold}
    if device:
        extractor_kwargs["device"] = device

    logging.info(
        f"Initializing SkillExtractor(skills_threshold={threshold}, device={device or 'auto'})"
    )
    extractor = SkillExtractor(**extractor_kwargs)

    results: List[List[str]] = []
    batches = chunk_list(texts, batch_size)

    for i, batch in enumerate(batches, start=1):
        batch_skills = extractor.get_skills(batch)
        results.extend(batch_skills)
        logging.info(
            f"Threshold {threshold}: batch {i}/{len(batches)} done ({len(results)}/{len(texts)} rows)"
        )

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract ESCO skills with ESCOX for multiple thresholds.")
    parser.add_argument("--input", required=True, help="Path to modules.xlsx")
    parser.add_argument("--output", required=True, help="Path to output .xlsx file")
    parser.add_argument("--sheet-name", default=0, help="Excel sheet name or index (default: 0)")
    parser.add_argument("--learning-outcomes-col", default="Learning Outcomes", help="Text column name")
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.50, 0.55, 0.60, 0.65],
        help="Thresholds to run (default: 0.50 0.55 0.60 0.65)",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--device", default=None, help='Force "cpu" or "cuda" (default: auto)')
    parser.add_argument(
        "--also-write-counts",
        action="store_true",
        help="If set, also write ESCOX n_skills (...) columns.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        logging.error(f"Input file not found: {in_path}")
        return 1

    df = pd.read_excel(in_path, sheet_name=args.sheet_name, engine="openpyxl")

    col = args.learning_outcomes_col
    if col not in df.columns:
        logging.error(f'Column "{col}" not found. Available columns: {list(df.columns)}')
        return 1

    texts = [normalize_text(x) for x in df[col].tolist()]

    # Run extraction for each threshold and add columns
    all_results: Dict[float, List[List[str]]] = {}
    for thr in args.thresholds:
        all_results[thr] = run_extraction_for_threshold(
            texts=texts,
            threshold=thr,
            batch_size=args.batch_size,
            device=args.device,
        )

        if len(all_results[thr]) != len(df):
            logging.error(f"Mismatch at threshold {thr}: results length != dataframe length.")
            return 1

        skills_col_name = f"ESCOX skills ({thr:.2f})"
        df[skills_col_name] = [skills_to_cell(r) for r in all_results[thr]]

        if args.also_write_counts:
            count_col_name = f"ESCOX n_skills ({thr:.2f})"
            df[count_col_name] = [len(r) for r in all_results[thr]]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_path, index=False, engine="openpyxl")

    logging.info(f"Done. Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
