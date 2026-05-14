#!/usr/bin/env python3
"""
Build a clean, reproducible dataset for curriculum–job skill alignment

Key ideas
- Extract skills via ESCOX for jobs/modules
- Enrich with skills_en.csv labels/types + allowed-flag
- Add 1-hop broader parent within SKILL namespace (http://data.europa.eu/esco/skill/...)
- For broader parent labels/types:
  - try skills_en.csv first (KnowledgeSkillCompetence parents live there)
  - if missing, use skillGroups_en.csv (SkillGroup concepts in skill namespace live there)
- Ignore ISCED-F completely

Run
python scripts/build_pipeline_dataset.py \
  --jobs-xlsx data/jobs.xlsx \
  --modules-xlsx data/modules.xlsx \
  --skills-en data/ESCO_skills/Original_ESCO_docs/skills_en.csv \
  --allowed-skills data/ESCO_skills/ESCO_skills_allowed_data_domain.csv \
  --broader-relations data/ESCO_skills/Original_ESCO_docs/broaderRelationsSkillPillar_en.csv \
  --skill-groups data/ESCO_skills/Original_ESCO_docs/skillGroups_en.csv \
  --out-xlsx out/pipeline_dataset.xlsx \
  --device cpu
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from esco_skill_extractor import SkillExtractor


# -------------------------
# Constants
# -------------------------
ESCO_SKILL_URI_PREFIX = "http://data.europa.eu/esco/skill/"
ESCO_ISCED_URI_PREFIX = "http://data.europa.eu/esco/isced-f/"
ESCO_SKILL_URI_RE = re.compile(r"^http://data\.europa\.eu/esco/skill/[0-9a-fA-F-]{8,}$")


# -------------------------
# Helpers
# -------------------------
def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    s = str(x).strip()
    return "" if s.lower() == "nan" else s


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    s = str(x).strip()
    return "" if s.lower() == "nan" else s


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def file_mtime_iso(path: str) -> str:
    ts = os.path.getmtime(path)
    return datetime.utcfromtimestamp(ts).isoformat() + "Z"


# -------------------------
# Load input sources
# -------------------------
def read_jobs_xlsx(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="main", engine="openpyxl")
    required = {"id", "job_description", "start_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"jobs.xlsx missing columns: {sorted(missing)}")

    out = df[["id", "job_description", "start_date"]].copy()
    out["job_description"] = out["job_description"].map(normalize_text)
    out = out[out["job_description"] != ""].reset_index(drop=True)

    out.rename(columns={"id": "record_id", "job_description": "text"}, inplace=True)
    out["source"] = "job"
    return out


def read_modules_xlsx(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="main", engine="openpyxl")
    required = {"module_id", "module_title", "learning_outcomes", "mandatory"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"modules.xlsx missing columns: {sorted(missing)}")

    out = df[["module_id", "module_title", "learning_outcomes", "mandatory"]].copy()
    out["learning_outcomes"] = out["learning_outcomes"].map(normalize_text)
    out = out[out["learning_outcomes"] != ""].reset_index(drop=True)

    out.rename(columns={"module_id": "record_id", "learning_outcomes": "text"}, inplace=True)
    out["source"] = "module"
    return out


def load_skills_en(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"conceptUri", "preferredLabel", "skillType"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"skills_en.csv missing columns: {sorted(missing)}")

    out = df[["conceptUri", "preferredLabel", "skillType"]].copy()
    out["conceptUri"] = out["conceptUri"].map(safe_str)
    out["preferredLabel"] = out["preferredLabel"].map(safe_str)

    # IMPORTANT: keep actual NaNs for skillType (do NOT coerce via safe_str)
    # so QC can count missing types correctly.
    out.rename(
        columns={"conceptUri": "skill_uri", "preferredLabel": "skill_label", "skillType": "skill_type"},
        inplace=True,
    )
    out = out.drop_duplicates(subset=["skill_uri"], keep="first").reset_index(drop=True)
    return out


def load_allowed_skills(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"skill_uri", "skill_label", "skill_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ESCO_skills_allowed_data_domain.csv missing columns: {sorted(missing)}")

    out = df[["skill_uri", "skill_label", "skill_type"]].copy()
    out["skill_uri"] = out["skill_uri"].map(safe_str)
    out["in_allowed"] = True
    out = out.drop_duplicates(subset=["skill_uri"], keep="first").reset_index(drop=True)
    return out


def load_broader_relations(path: str) -> pd.DataFrame:
    """
    ESCO v1.2.1 columns:
      conceptType, conceptUri, conceptLabel, broaderType, broaderUri, broaderLabel
    """
    df = pd.read_csv(path)
    required = {"conceptUri", "broaderUri"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"broaderRelationsSkillPillar_en.csv missing columns: {sorted(missing)}")

    cols = ["conceptUri", "broaderUri"] + (["broaderType", "conceptType"] if {"broaderType", "conceptType"}.issubset(df.columns) else [])
    out = df[cols].copy()
    out.rename(columns={"conceptUri": "narrower_uri", "broaderUri": "broader_uri"}, inplace=True)
    if "broaderType" in out.columns:
        out.rename(columns={"broaderType": "broader_type"}, inplace=True)
    else:
        out["broader_type"] = ""
    if "conceptType" in out.columns:
        out.rename(columns={"conceptType": "narrower_type"}, inplace=True)
    else:
        out["narrower_type"] = ""

    for c in ["narrower_uri", "broader_uri", "broader_type", "narrower_type"]:
        out[c] = out[c].map(safe_str)

    out = out[(out["narrower_uri"] != "") & (out["broader_uri"] != "")].copy()
    out = out.drop_duplicates(subset=["narrower_uri", "broader_uri"]).reset_index(drop=True)
    return out


def load_skill_groups_en(path: str) -> pd.DataFrame:
    """
    skillGroups_en.csv contains:
      - SkillGroup concepts in skill namespace: http://data.europa.eu/esco/skill/...
      - ISCED-F groups: http://data.europa.eu/esco/isced-f/...
    We only keep SKILL NAMESPACE groups as a label/type fallback for broader parents.
    """
    df = pd.read_csv(path)
    required = {"conceptUri", "preferredLabel", "conceptType"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"skillGroups_en.csv missing columns: {sorted(missing)}")

    out = df[["conceptUri", "preferredLabel", "conceptType"]].copy()
    out.rename(
        columns={"conceptUri": "group_uri", "preferredLabel": "group_label", "conceptType": "group_concept_type"},
        inplace=True,
    )
    out["group_uri"] = out["group_uri"].map(safe_str)
    out["group_label"] = out["group_label"].map(safe_str)
    out["group_concept_type"] = out["group_concept_type"].map(safe_str)

    # keep ONLY skill namespace (drop ISCED entirely)
    out = out[out["group_uri"].str.startswith(ESCO_SKILL_URI_PREFIX)].copy()
    out = out[out["group_concept_type"].str.lower() == "skillgroup"].copy()

    out = out.drop_duplicates(subset=["group_uri"], keep="first").reset_index(drop=True)
    return out


# -------------------------
# ESCOX extraction
# -------------------------
def escox_extract_batch(
    texts: List[str],
    threshold: float,
    batch_size: int = 32,
    device: Optional[str] = None,
) -> List[List[str]]:
    kwargs: Dict[str, Any] = {"skills_threshold": threshold}
    if device:
        kwargs["device"] = device

    logging.info(f"Initializing SkillExtractor(skills_threshold={threshold}, device={device or 'auto'})")
    extractor = SkillExtractor(**kwargs)

    results: List[List[str]] = []
    batches = chunk_list(texts, batch_size)

    for i, batch in enumerate(batches, start=1):
        batch_skills = extractor.get_skills(batch)
        results.extend(batch_skills)
        logging.info(f"Threshold {threshold}: batch {i}/{len(batches)} done ({len(results)}/{len(texts)} rows)")

    return results


def extract_long(records: pd.DataFrame, threshold: float, batch_size: int, device: Optional[str]) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame(columns=["record_id", "source", "threshold", "skill_uri"])

    texts = records["text"].map(normalize_text).tolist()
    ids = records["record_id"].tolist()
    source = records["source"].iloc[0]

    skill_lists = escox_extract_batch(texts=texts, threshold=threshold, batch_size=batch_size, device=device)

    if len(skill_lists) != len(records):
        raise RuntimeError(f"ESCOX mismatch: got {len(skill_lists)} results for {len(records)} records (source={source})")

    rows: List[Dict[str, Any]] = []
    for rid, skills in zip(ids, skill_lists):
        for uri in skills:
            rows.append({"record_id": rid, "source": source, "threshold": threshold, "skill_uri": safe_str(uri)})

    out = pd.DataFrame(rows, columns=["record_id", "source", "threshold", "skill_uri"])
    out = out.drop_duplicates(subset=["record_id", "source", "threshold", "skill_uri"]).reset_index(drop=True)
    return out


# -------------------------
# Enrichment
# -------------------------
def add_uri_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["skill_uri"] = out["skill_uri"].map(safe_str)
    out["is_valid_skill_uri"] = out["skill_uri"].map(lambda u: bool(ESCO_SKILL_URI_RE.match(u)))
    out["has_esco_skill_prefix"] = out["skill_uri"].map(lambda u: u.startswith(ESCO_SKILL_URI_PREFIX))
    return out


def enrich_with_labels_and_allowed(
    skills_long: pd.DataFrame,
    skills_en: pd.DataFrame,
    allowed_df: pd.DataFrame,
) -> pd.DataFrame:
    allowed_set = set(allowed_df["skill_uri"].astype(str))

    out = skills_long.copy()
    out["skill_uri"] = out["skill_uri"].map(safe_str)

    out = add_uri_quality_flags(out)

    # labels/types from skills_en (keep NaNs as NaNs)
    out = out.merge(skills_en, on="skill_uri", how="left")

    out["in_allowed"] = out["skill_uri"].isin(allowed_set)
    return out


def build_one_hop_broader_skill_rollup(broader_df: pd.DataFrame) -> pd.DataFrame:
    """
    1-hop roll-up to broader SKILL parent only (esco/skill namespace).
    Returns mapping keyed by skill_uri:
      skill_uri, broader_skill_uri_1, broader_skill_status, broader_skill_n_parents
    """
    rel = broader_df.copy()
    rel["narrower_uri"] = rel["narrower_uri"].map(safe_str)
    rel["broader_uri"] = rel["broader_uri"].map(safe_str)

    rel_skill = rel[rel["broader_uri"].str.startswith(ESCO_SKILL_URI_PREFIX)].copy()

    if rel_skill.empty:
        return pd.DataFrame(
            columns=["skill_uri", "broader_skill_uri_1", "broader_skill_status", "broader_skill_n_parents"]
        )

    tmp = rel_skill[["narrower_uri", "broader_uri"]].copy()
    tmp.rename(columns={"narrower_uri": "skill_uri", "broader_uri": "broader_skill_uri_1"}, inplace=True)

    counts = rel_skill.groupby("narrower_uri")["broader_uri"].nunique().reset_index()
    counts.rename(columns={"narrower_uri": "skill_uri", "broader_uri": "broader_skill_n_parents"}, inplace=True)

    pick = tmp.sort_values(["skill_uri", "broader_skill_uri_1"]).drop_duplicates(subset=["skill_uri"], keep="first")
    pick = pick.merge(counts, on="skill_uri", how="left")

    pick["broader_skill_status"] = "mapped"
    return pick[["skill_uri", "broader_skill_uri_1", "broader_skill_status", "broader_skill_n_parents"]]


def add_broader_skill_rollup_and_labels(
    enriched_all: pd.DataFrame,
    broader_df: pd.DataFrame,
    skills_en: pd.DataFrame,
    skill_groups_skill_namespace: pd.DataFrame,
) -> pd.DataFrame:
    """
    Adds broader_skill_uri_1 + fills broader_skill_label/type from:
      1) skills_en (KnowledgeSkillCompetence parents)
      2) skillGroups_en (SkillGroup parents; skill namespace only)
    """
    out = enriched_all.copy()
    out["skill_uri"] = out["skill_uri"].map(safe_str)

    rollup = build_one_hop_broader_skill_rollup(broader_df=broader_df)
    out = out.merge(rollup, on="skill_uri", how="left")
    out["broader_skill_status"] = out["broader_skill_status"].fillna("no_parent")
    out["broader_skill_n_parents"] = out["broader_skill_n_parents"].fillna(0).astype(int)

    # ---- label/type for broader_skill_uri_1
    # 1) from skills_en
    broader_from_skills = skills_en[["skill_uri", "skill_label", "skill_type"]].drop_duplicates("skill_uri").copy()
    broader_from_skills.rename(
        columns={
            "skill_uri": "broader_skill_uri_1",
            "skill_label": "broader_skill_label_1",
            "skill_type": "broader_skill_type_1",
        },
        inplace=True,
    )
    out = out.merge(broader_from_skills, on="broader_skill_uri_1", how="left")

    # 2) fallback from skillGroups_en (skill namespace only)
    # Use a second merge to get fallback labels; then fillna
    sg = skill_groups_skill_namespace[["group_uri", "group_label", "group_concept_type"]].drop_duplicates("group_uri").copy()
    sg.rename(
        columns={
            "group_uri": "broader_skill_uri_1",
            "group_label": "broader_skill_label_1_sg",
            "group_concept_type": "broader_skill_type_1_sg",
        },
        inplace=True,
    )
    out = out.merge(sg, on="broader_skill_uri_1", how="left")

    out["broader_skill_label_1"] = out["broader_skill_label_1"].where(out["broader_skill_label_1"].notna(), out["broader_skill_label_1_sg"])
    out["broader_skill_type_1"] = out["broader_skill_type_1"].where(out["broader_skill_type_1"].notna(), out["broader_skill_type_1_sg"])

    out.drop(columns=["broader_skill_label_1_sg", "broader_skill_type_1_sg"], inplace=True, errors="ignore")

    return out


# -------------------------
# Writing
# -------------------------
def write_excel(out_path: str, sheets: Dict[str, pd.DataFrame]) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)


# -------------------------
# Main
# -------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Build pipeline dataset v1.1 (ESCOX skills for jobs+modules; labels+allowed; "
            "adds 1-hop broader SKILL roll-up only; ignores ISCED-F; "
            "fills broader labels from skills_en then skillGroups_en (skill namespace)."
        )
    )

    ap.add_argument("--jobs-xlsx", default="data/jobs.xlsx")
    ap.add_argument("--modules-xlsx", default="data/modules.xlsx")

    ap.add_argument("--skills-en", required=True, help="Path to skills_en.csv")
    ap.add_argument("--allowed-skills", required=True, help="Path to ESCO_skills_allowed_data_domain.csv")
    ap.add_argument("--broader-relations", required=True, help="Path to broaderRelationsSkillPillar_en.csv")
    ap.add_argument("--skill-groups", required=True, help="Path to skillGroups_en.csv (used for broader label fallback)")

    ap.add_argument("--threshold-jobs", type=float, default=0.60)
    ap.add_argument("--threshold-modules", type=float, default=0.55)

    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=None, help="e.g., cpu, cuda, mps (depends on your setup)")

    ap.add_argument("--out-xlsx", default="out/pipeline_dataset_v1_1.xlsx")

    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    # ---- Load inputs
    jobs = read_jobs_xlsx(args.jobs_xlsx)
    modules = read_modules_xlsx(args.modules_xlsx)

    skills_en = load_skills_en(args.skills_en)
    allowed_df = load_allowed_skills(args.allowed_skills)
    broader_df = load_broader_relations(args.broader_relations)
    skill_groups_skill_ns = load_skill_groups_en(args.skill_groups)

    # ---- Extract
    logging.info("Extracting ESCOX skills for jobs...")
    jobs_long = extract_long(jobs, threshold=args.threshold_jobs, batch_size=args.batch_size, device=args.device)

    logging.info("Extracting ESCOX skills for modules...")
    modules_long = extract_long(modules, threshold=args.threshold_modules, batch_size=args.batch_size, device=args.device)

    all_long = pd.concat([jobs_long, modules_long], ignore_index=True)
    all_long = all_long.drop_duplicates(subset=["record_id", "source", "threshold", "skill_uri"]).reset_index(drop=True)

    # ---- Enrich: labels + allowed + URI quality flags
    enriched_all = enrich_with_labels_and_allowed(all_long, skills_en=skills_en, allowed_df=allowed_df)

    # ---- Add broader SKILL roll-up + labels (skills_en then skillGroups_en fallback; no ISCED)
    enriched_all = add_broader_skill_rollup_and_labels(
        enriched_all=enriched_all,
        broader_df=broader_df,
        skills_en=skills_en,
        skill_groups_skill_namespace=skill_groups_skill_ns,
    )

    # ---- Attach record metadata back
    jobs_meta = jobs[["record_id", "start_date"]].copy()
    modules_meta = modules[["record_id", "module_title", "mandatory"]].copy()

    enriched_all = enriched_all.merge(jobs_meta, on="record_id", how="left")
    enriched_all = enriched_all.merge(modules_meta, on="record_id", how="left")

    enriched_allowed = enriched_all[enriched_all["in_allowed"]].copy()

    # ---- Summaries / QA
    run_summary = pd.DataFrame(
        [
            {
                "jobs_records": len(jobs),
                "modules_records": len(modules),
                "jobs_extracted_rows": len(jobs_long),
                "modules_extracted_rows": len(modules_long),
                "all_extracted_rows": len(all_long),
                "unique_skills_total": int(all_long["skill_uri"].nunique(dropna=True)),
                "unique_skills_allowed": int(enriched_allowed["skill_uri"].nunique(dropna=True)),
                "threshold_jobs": args.threshold_jobs,
                "threshold_modules": args.threshold_modules,
                "batch_size": args.batch_size,
                "device": args.device or "auto",
            }
        ]
    )

    data_quality = pd.DataFrame(
        [
            {
                "rows_total": len(enriched_all),
                "rows_allowed": len(enriched_allowed),
                "rows_invalid_uri": int((~enriched_all["is_valid_skill_uri"]).sum()),
                "rows_missing_label": int(enriched_all["skill_label"].isna().sum()),
                # count true NaNs (not empty strings)
                "rows_missing_type": int(enriched_all["skill_type"].isna().sum()),
                "rows_broader_skill_mapped": int((enriched_all["broader_skill_status"] == "mapped").sum()),
                "rows_broader_skill_missing": int((enriched_all["broader_skill_status"] == "no_parent").sum()),
                "broader_skill_multi_parent_rows": int((enriched_all["broader_skill_n_parents"] > 1).sum()),
                "rows_broader_missing_label": int(enriched_all["broader_skill_label_1"].isna().sum()),
                "rows_broader_missing_type": int(enriched_all["broader_skill_type_1"].isna().sum()),
            }
        ]
    )

    # ---- Minimal run meta (IN EXCEL ONLY)
    meta = {
        "run_timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "args": vars(args),
        "inputs": {
            "jobs_xlsx": {"path": args.jobs_xlsx, "sha256": sha256_file(args.jobs_xlsx), "mtime_utc": file_mtime_iso(args.jobs_xlsx)},
            "modules_xlsx": {"path": args.modules_xlsx, "sha256": sha256_file(args.modules_xlsx), "mtime_utc": file_mtime_iso(args.modules_xlsx)},
            "skills_en": {"path": args.skills_en, "sha256": sha256_file(args.skills_en), "mtime_utc": file_mtime_iso(args.skills_en)},
            "allowed_skills": {"path": args.allowed_skills, "sha256": sha256_file(args.allowed_skills), "mtime_utc": file_mtime_iso(args.allowed_skills)},
            "broader_relations": {"path": args.broader_relations, "sha256": sha256_file(args.broader_relations), "mtime_utc": file_mtime_iso(args.broader_relations)},
            "skill_groups": {"path": args.skill_groups, "sha256": sha256_file(args.skill_groups), "mtime_utc": file_mtime_iso(args.skill_groups)},
        },
    }

    sheets: Dict[str, pd.DataFrame] = {
        "jobs_input": jobs,
        "modules_input": modules,
        "skills_long_raw": all_long,
        "skills_long_enriched_all": enriched_all,
        "skills_long_enriched_allowed": enriched_allowed,
        "skills_en_lookup": skills_en,
        "allowed_skills_lookup": allowed_df,
        "broader_relations": broader_df,
        "skill_groups_skill_ns": skill_groups_skill_ns,
        "run_summary": run_summary,
        "data_quality": data_quality,
        "run_meta_flat": pd.json_normalize(meta, sep="."),
    }

    write_excel(args.out_xlsx, sheets)
    logging.info(f"Done. Wrote output Excel: {args.out_xlsx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
