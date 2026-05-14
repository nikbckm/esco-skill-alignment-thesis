#!/usr/bin/env python3
"""
Extract ESCO skills from job descriptions for multiple ESCOX thresholds
and write them into a normalized output sheet:

refnr | Skill URI | Skill Name | threshold

Example:
  python scripts/extract_escox_skills_from_jobs.py \
    --input data/escox_jobs_thresholds.xlsx \
    --output data/escox_jobs_thresholds.xlsx \
    --sheet-name input \
    --text-col stellenangebotsBeschreibungENG \
    --refnr-col refnr
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, List, Dict

import pandas as pd
from esco_skill_extractor import SkillExtractor


# =========================
# Helpers
# =========================

def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    s = str(x).strip()
    return "" if s.lower() == "nan" else s


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    return [items[i: i + chunk_size] for i in range(0, len(items), chunk_size)]


def run_extraction_for_threshold(
    texts: List[str],
    threshold: float,
    batch_size: int,
    device: str | None,
) -> List[List[str]]:

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
            f"Threshold {threshold}: batch {i}/{len(batches)} done "
            f"({len(results)}/{len(texts)} rows)"
        )

    return results


# =========================
# Main
# =========================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract ESCO skills from job descriptions for multiple thresholds."
    )

    parser.add_argument("--input", required=True, help="Path to input Excel file")
    parser.add_argument("--output", required=True, help="Path to output Excel file")
    parser.add_argument("--sheet-name", default="input", help="Sheet containing jobs")
    parser.add_argument("--text-col", default="stellenangebotsBeschreibungENG", help="Job text column")
    parser.add_argument("--refnr-col", default="refnr", help="Reference number column")
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.50, 0.55, 0.60, 0.65],
        help="Thresholds to run"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        logging.error(f"Input file not found: {in_path}")
        return 1

    df = pd.read_excel(in_path, sheet_name=args.sheet_name, engine="openpyxl")

    if args.text_col not in df.columns:
        logging.error(f'Text column "{args.text_col}" not found.')
        return 1

    if args.refnr_col not in df.columns:
        logging.error(f'Refnr column "{args.refnr_col}" not found.')
        return 1

    texts = [normalize_text(x) for x in df[args.text_col].tolist()]
    refnrs = df[args.refnr_col].tolist()

    all_rows = []

    # =========================
    # Run per threshold
    # =========================

    for thr in args.thresholds:
        skill_lists = run_extraction_for_threshold(
            texts=texts,
            threshold=thr,
            batch_size=args.batch_size,
            device=args.device,
        )

        if len(skill_lists) != len(df):
            logging.error(f"Mismatch at threshold {thr}: results length != dataframe length.")
            return 1

        for refnr, skills in zip(refnrs, skill_lists):
            for skill in skills:
                # ESCOX returns URIs. Extract skill name from URI tail if needed.
                skill_uri = skill
                skill_name = skill.split("/")[-1]  # fallback if label not provided

                all_rows.append({
                    "refnr": refnr,
                    "Skill URI": skill_uri,
                    "Skill Name": skill_name,
                    "threshold": thr,
                })

    # =========================
    # Write normalized output
    # =========================

    df_out = pd.DataFrame(all_rows)

    with pd.ExcelWriter(out_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        df_out.to_excel(writer, sheet_name="output", index=False)

    logging.info(f"Done. Output written to sheet 'output' in {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
