#!/usr/bin/env python3
"""
Build curriculum–job skill dataset using:

- ESCOX extraction
- Deterministic safety-net keyword matching (modules only)
- Label enrichment via ESCO skills_en.csv
- Allowed-skill filtering
- 1-hop broader skill roll-up (skill namespace only)

To run:
python scripts/build_pipeline_dataset.py \
  --jobs-xlsx data/jobs.xlsx \
  --modules-xlsx data/modules.xlsx \
  --skills-en data/ESCO_skills/Original_ESCO_docs/skills_en.csv \
  --allowed-skills data/ESCO_skills/ESCO_skills_allowed_data_domain.csv \
  --broader-relations data/ESCO_skills/Original_ESCO_docs/broaderRelationsSkillPillar_en.csv \
  --safety-net-csv data/ESCO_skills/safety_net_skills.csv \
  --threshold-jobs 0.6 \
  --threshold-modules 0.55 \
  --device cpu \
  --out-xlsx out/pipeline_dataset.xlsx


Output:
    Excel file with:
        - skills_long_enriched_all
        - skills_long_enriched_allowed
        - run_summary
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from esco_skill_extractor import SkillExtractor


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

ESCO_SKILL_URI_PREFIX = "http://data.europa.eu/esco/skill/"
ESCO_SKILL_URI_RE = re.compile(r"^http://data\.europa\.eu/esco/skill/[0-9a-fA-F-]{8,}$")


# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------

def normalize_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def safe_str(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


# ---------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------

def read_jobs_xlsx(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="main")
    df = df[["id", "job_description"]].copy()
    df["job_description"] = df["job_description"].map(normalize_text)
    df = df[df["job_description"] != ""]
    df.rename(columns={"id": "record_id", "job_description": "text"}, inplace=True)
    df["source"] = "job"
    return df.reset_index(drop=True)


def read_modules_xlsx(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="main")

    # Use combined column instead of learning_outcomes only
    required_col = "module_title_and_learning_outcomes"
    if required_col not in df.columns:
        raise ValueError(
            f"modules.xlsx must contain column '{required_col}'"
        )

    df = df[["module_id", "module_title", required_col]].copy()

    df[required_col] = df[required_col].map(normalize_text)
    df = df[df[required_col] != ""]

    df.rename(
        columns={
            "module_id": "record_id",
            required_col: "text",
        },
        inplace=True,
    )

    df["source"] = "module"
    return df.reset_index(drop=True)


def load_skills_en(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[["conceptUri", "preferredLabel", "skillType"]].copy()
    df.rename(
        columns={
            "conceptUri": "skill_uri",
            "preferredLabel": "skill_label",
            "skillType": "skill_type",
        },
        inplace=True,
    )
    return df.drop_duplicates("skill_uri")


def load_allowed_skills(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[["skill_uri"]].copy()
    df["in_allowed"] = True
    return df.drop_duplicates("skill_uri")


def load_broader_relations(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[["conceptUri", "broaderUri"]].copy()
    df.rename(columns={"conceptUri": "skill_uri", "broaderUri": "broader_skill_uri_1"}, inplace=True)
    df = df[df["broader_skill_uri_1"].str.startswith(ESCO_SKILL_URI_PREFIX)]
    return df.drop_duplicates("skill_uri")


# ---------------------------------------------------------------------
# ESCOX extraction
# ---------------------------------------------------------------------

def extract_escox(records: pd.DataFrame, threshold: float, device: str | None) -> pd.DataFrame:
    extractor = SkillExtractor(skills_threshold=threshold, device=device)

    rows = []
    for _, r in records.iterrows():
        skills = extractor.get_skills([r["text"]])[0]
        for uri in skills:
            rows.append(
                {
                    "record_id": r["record_id"],
                    "source": r["source"],
                    "threshold": threshold,
                    "skill_uri": safe_str(uri),
                    "extraction_method": "escox",
                    "match_keyword": "",
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Safety net (modules only)
# ---------------------------------------------------------------------

def apply_safety_net(
    modules: pd.DataFrame,
    modules_escox: pd.DataFrame,
    safety_net_df: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """
    Adds deterministic keyword matches for modules.
    Only retains skills NOT already found by ESCOX.
    """

    escox_keys = set(zip(modules_escox["record_id"], modules_escox["skill_uri"]))

    rows = []

    for _, mod in modules.iterrows():
        text_lower = mod["text"].lower()

        for _, sn in safety_net_df.iterrows():
            keyword = sn["keyword"].lower()

            if re.search(rf"\b{re.escape(keyword)}\b", text_lower):
                key = (mod["record_id"], sn["skill_uri"])

                # Only add if ESCOX missed it
                if key not in escox_keys:
                    rows.append(
                        {
                            "record_id": mod["record_id"],
                            "source": "module",
                            "threshold": threshold,
                            "skill_uri": sn["skill_uri"],
                            "extraction_method": "safety_net",
                            "match_keyword": keyword,
                        }
                    )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------

def enrich(
    df: pd.DataFrame,
    skills_en: pd.DataFrame,
    allowed_df: pd.DataFrame,
    broader_df: pd.DataFrame,
) -> pd.DataFrame:

    df["skill_uri"] = df["skill_uri"].map(safe_str)

    df = df.merge(skills_en, on="skill_uri", how="left")
    df = df.merge(allowed_df, on="skill_uri", how="left")
    df["in_allowed"] = df["in_allowed"].fillna(False)

    df = df.merge(broader_df, on="skill_uri", how="left")

    df["is_valid_skill_uri"] = df["skill_uri"].map(
        lambda u: bool(ESCO_SKILL_URI_RE.match(u))
    )

    return df


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument("--jobs-xlsx", required=True)
    parser.add_argument("--modules-xlsx", required=True)

    parser.add_argument("--skills-en", required=True)
    parser.add_argument("--allowed-skills", required=True)
    parser.add_argument("--broader-relations", required=True)
    parser.add_argument("--safety-net-csv", required=True)

    parser.add_argument("--threshold-jobs", type=float, default=0.6)
    parser.add_argument("--threshold-modules", type=float, default=0.55)
    parser.add_argument("--device", default=None)

    parser.add_argument("--out-xlsx", required=True)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    # Load input data
    jobs = read_jobs_xlsx(args.jobs_xlsx)
    modules = read_modules_xlsx(args.modules_xlsx)

    skills_en = load_skills_en(args.skills_en)
    allowed_df = load_allowed_skills(args.allowed_skills)
    broader_df = load_broader_relations(args.broader_relations)
    safety_net_df = pd.read_csv(args.safety_net_csv)

    # ESCOX extraction
    jobs_escox = extract_escox(jobs, args.threshold_jobs, args.device)
    modules_escox = extract_escox(modules, args.threshold_modules, args.device)

    # Safety net
    modules_sn = apply_safety_net(
        modules,
        modules_escox,
        safety_net_df,
        args.threshold_modules,
    )

    # Combine
    all_long = pd.concat(
        [jobs_escox, modules_escox, modules_sn],
        ignore_index=True,
    ).drop_duplicates(
        subset=["record_id", "source", "skill_uri"]
    )

    # Enrichment
    enriched_all = enrich(all_long, skills_en, allowed_df, broader_df)
    enriched_allowed = enriched_all[enriched_all["in_allowed"]].copy()

    # Run summary
    run_summary = pd.DataFrame(
        [
            {
                "jobs_records": len(jobs),
                "modules_records": len(modules),
                "jobs_escox_rows": len(jobs_escox),
                "modules_escox_rows": len(modules_escox),
                "modules_safety_net_added": len(modules_sn),
                "unique_skills_total": enriched_all["skill_uri"].nunique(),
                "unique_skills_allowed": enriched_allowed["skill_uri"].nunique(),
                "threshold_jobs": args.threshold_jobs,
                "threshold_modules": args.threshold_modules,
                "safety_net_rows": len(safety_net_df),
            }
        ]
    )

    # Write Excel (minimal sheets)
    Path(args.out_xlsx).parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.out_xlsx) as writer:
        enriched_all.to_excel(writer, sheet_name="skills_long_enriched_all", index=False)
        enriched_allowed.to_excel(writer, sheet_name="skills_long_enriched_allowed", index=False)
        run_summary.to_excel(writer, sheet_name="run_summary", index=False)

    logging.info("Pipeline finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
