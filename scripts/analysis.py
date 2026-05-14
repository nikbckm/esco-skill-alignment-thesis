#!/usr/bin/env python3
# analysis.py
# Console-only analysis for curriculum (modules) vs job market (jobs)
#
# Input: pipeline_dataset.xlsx with sheets:
# - skills_long_enriched_all
# - skills_long_enriched_allowed
# - run_summary (optional)

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import pandas as pd

SHEET_ALL = "skills_long_enriched_all"
SHEET_ALLOWED = "skills_long_enriched_allowed"
SHEET_SUMMARY = "run_summary"


# -------------------------
# Small utils
# -------------------------
def fmt_pct(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x * 100:.1f}%"


def safe_div(a: float, b: float) -> float:
    return (a / b) if b else float("nan")


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return float("nan")
    return len(a & b) / len(a | b)


def overlap_coeff(a: set, b: set) -> float:
    if not a or not b:
        return float("nan")
    return len(a & b) / min(len(a), len(b))


def cosine_sim_from_counters(c1: Counter, c2: Counter) -> float:
    keys = set(c1) | set(c2)
    if not keys:
        return float("nan")
    v1 = np.array([c1.get(k, 0) for k in keys], dtype=float)
    v2 = np.array([c2.get(k, 0) for k in keys], dtype=float)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return float("nan")
    return float(np.dot(v1, v2) / (n1 * n2))


def print_block(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def _clean_id_col(series: pd.Series) -> pd.Series:
    # keep as string; drop empty/nan-like
    s = series.astype("string")
    s = s.fillna(pd.NA)
    s = s.map(lambda x: x.strip() if isinstance(x, str) else x)
    return s


def _filter_nonempty_id(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    out[col] = _clean_id_col(out[col])
    out = out[out[col].notna()]
    out = out[out[col].astype(str).str.lower() != "nan"]
    out = out[out[col].astype(str).str.strip() != ""]
    return out


def _split_sources(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Robust source split: supports 'job'/'module' (your pipeline) and also 'jobs'/'modules' (older scripts).
    """
    src = df["source"].astype(str).str.strip().str.lower()
    jobs = df[src.isin(["job", "jobs"])].copy()
    mods = df[src.isin(["module", "modules"])].copy()
    return jobs, mods


# -------------------------
# Core analyses
# -------------------------
def analyze_level(
    df: pd.DataFrame,
    level_col: str,
    label_col: str | None,
    level_name: str,
) -> None:
    """
    Curriculum vs jobs comparisons at a given aggregation level.
    level_col: 'skill_uri' or 'broader_skill_uri_1'
    label_col: label column to print (optional)
    """
    print_block(f"ANALYSIS LEVEL: {level_name} ({level_col})")

    if level_col not in df.columns:
        print(f"Column '{level_col}' not found. Skipping.")
        return

    d = _filter_nonempty_id(df, level_col)
    jobs, mods = _split_sources(d)

    jobs_set = set(jobs[level_col].astype(str).unique())
    mods_set = set(mods[level_col].astype(str).unique())
    inter = jobs_set & mods_set

    jobs_freq = Counter(jobs[level_col].astype(str).tolist())
    mods_freq = Counter(mods[level_col].astype(str).tolist())

    coverage_job_unique = safe_div(len(inter), len(jobs_set))
    coverage_mod_unique = safe_div(len(inter), len(mods_set))
    jac = jaccard(jobs_set, mods_set)
    ovl = overlap_coeff(jobs_set, mods_set)
    cos = cosine_sim_from_counters(jobs_freq, mods_freq)

    print(f"Unique {level_name} in jobs    : {len(jobs_set):,}")
    print(f"Unique {level_name} in modules : {len(mods_set):,}")
    print(f"Overlap (intersection)         : {len(inter):,}")
    print(f"Job-unique coverage by curriculum (|J∩C|/|J|): {fmt_pct(coverage_job_unique)}")
    print(f"Curriculum-unique covered by jobs (|J∩C|/|C|): {fmt_pct(coverage_mod_unique)}")
    print(f"Jaccard similarity (sets)      : {jac:.3f}")
    print(f"Overlap coefficient (sets)     : {ovl:.3f}")
    print(f"Cosine similarity (freq)       : {cos:.3f}")

    missing = sorted(list(jobs_set - mods_set), key=lambda x: jobs_freq.get(x, 0), reverse=True)
    extra = sorted(list(mods_set - jobs_set), key=lambda x: mods_freq.get(x, 0), reverse=True)

    # label lookup (optional; best for skill_uri; broader labels may be missing)
    label_map: dict[str, str] = {}
    if label_col and label_col in d.columns:
        tmp = d[[level_col, label_col]].dropna()
        for k, v in tmp.itertuples(index=False):
            ks = str(k)
            vs = str(v).strip()
            if ks not in label_map and vs and vs.lower() != "nan":
                label_map[ks] = vs

    def show_list(title: str, ids: list[str], freq_counter: Counter, n: int = 25) -> None:
        print("\n" + title)
        print("-" * len(title))
        for i, sid in enumerate(ids[:n], start=1):
            lbl = label_map.get(sid, "")
            if lbl:
                print(f"{i:>2}. {lbl}  | count={freq_counter.get(sid, 0)}  | {sid}")
            else:
                print(f"{i:>2}. count={freq_counter.get(sid, 0)}  | {sid}")
        if len(ids) > n:
            print(f"... ({len(ids) - n:,} more)")

    show_list(f"Top missing {level_name} (job -> curriculum gaps)", missing, jobs_freq, n=25)
    show_list(f"Top extra {level_name} (curriculum-only)", extra, mods_freq, n=25)

    # Top-K coverage (unique) among most frequent job skills
    job_top = [k for k, _ in jobs_freq.most_common(200)]

    def topk_coverage(k: int) -> float:
        topk = set(job_top[:k])
        return safe_div(len(topk & mods_set), len(topk))

    print("\nTop-K job-skill coverage (unique) — 'Does curriculum cover the most frequent job skills?'")
    for k in [10, 25, 50, 100, 200]:
        print(f"  Top {k:>3}: {fmt_pct(topk_coverage(k))}")


def record_level_fit(df: pd.DataFrame, level_col: str, level_name: str) -> None:
    """
    Per job record: share of job's unique skills covered by curriculum set.
    """
    print_block(f"RECORD-LEVEL FIT: {level_name} ({level_col})")

    if level_col not in df.columns:
        print(f"Column '{level_col}' not found. Skipping.")
        return

    d = _filter_nonempty_id(df, level_col)
    jobs, mods = _split_sources(d)

    curriculum_set = set(mods[level_col].astype(str).unique())
    if jobs.empty:
        print("No job rows found. Skipping record-level fit.")
        return

    job_to_skills = jobs.groupby("record_id")[level_col].apply(lambda s: set(s.astype(str).tolist()))
    if job_to_skills.empty:
        print("No job records found. Skipping record-level fit.")
        return

    ratios = []
    sizes = []
    for _, skills in job_to_skills.items():
        if not skills:
            continue
        covered = len(skills & curriculum_set)
        ratios.append(covered / len(skills))
        sizes.append(len(skills))

    ratios = np.array(ratios, dtype=float)
    sizes = np.array(sizes, dtype=float)

    def q(a: np.ndarray, p: float) -> float:
        return float(np.percentile(a, p)) if len(a) else float("nan")

    print(f"Job records analysed: {len(ratios):,}")
    print(f"Avg skills per job (unique {level_name}) : {np.mean(sizes):.2f}")
    print(f"Median skills per job                    : {np.median(sizes):.2f}")

    print("\nCoverage distribution per job record: (covered job skills by curriculum)")
    print(f"Mean   : {fmt_pct(float(np.mean(ratios)))}")
    print(f"Median : {fmt_pct(float(np.median(ratios)))}")
    print(f"P10    : {fmt_pct(q(ratios, 10))}")
    print(f"P25    : {fmt_pct(q(ratios, 25))}")
    print(f"P75    : {fmt_pct(q(ratios, 75))}")
    print(f"P90    : {fmt_pct(q(ratios, 90))}")

    job_fit = pd.DataFrame(
        {
            "record_id": job_to_skills.index.astype(str),
            "job_unique_skills": [len(s) for s in job_to_skills.values],
            "coverage_ratio": ratios,
        }
    ).sort_values(["coverage_ratio", "job_unique_skills"], ascending=[True, False])

    print("\nBottom 10 job records by coverage:")
    print(job_fit.head(10).to_string(index=False))

    print("\nTop 10 job records by coverage:")
    print(
        job_fit.tail(10)
        .sort_values(["coverage_ratio", "job_unique_skills"], ascending=[False, False])
        .to_string(index=False)
    )


def breakdowns(df: pd.DataFrame) -> None:
    print_block("BREAKDOWNS (sanity & extraction artifacts)")

    print("Rows:", f"{len(df):,}")
    if "source" in df.columns:
        print("Sources:", df["source"].astype(str).str.lower().value_counts(dropna=False).to_dict())
    if {"source", "threshold"}.issubset(df.columns):
        print("\nThresholds (rows):")
        print(df.groupby([df["source"].astype(str).str.lower(), "threshold"]).size().to_string())

    if "skill_type" in df.columns:
        print("\nSkill type counts (rows):")
        print(df.groupby([df["source"].astype(str).str.lower(), "skill_type"]).size().to_string())

    if "extraction_method" in df.columns:
        print("\nExtraction method counts (rows):")
        print(df.groupby([df["source"].astype(str).str.lower(), "extraction_method"]).size().to_string())

    if "is_valid_skill_uri" in df.columns:
        print("\nValidity flag counts (rows):")
        print(df["is_valid_skill_uri"].value_counts(dropna=False).to_string())

    if "in_allowed" in df.columns:
        print("\nIn-allowed flag counts (rows):")
        print(df["in_allowed"].value_counts(dropna=False).to_string())


# -------------------------
# Main
# -------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Minimal console-only module vs job skill alignment analysis.")
    ap.add_argument("--file", default="out/pipeline_dataset.xlsx", help="Path to pipeline_dataset.xlsx")
    ap.add_argument(
        "--use",
        choices=["allowed", "all"],
        default="allowed",
        help="Analyse allowed sheet (recommended) or full unfiltered sheet",
    )
    args = ap.parse_args()

    xls = pd.ExcelFile(args.file)

    # run_summary (optional)
    run_sum = None
    if SHEET_SUMMARY in xls.sheet_names:
        run_sum = pd.read_excel(args.file, sheet_name=SHEET_SUMMARY)

    sheet = SHEET_ALLOWED if args.use == "allowed" else SHEET_ALL
    if sheet not in xls.sheet_names:
        raise SystemExit(f"Missing sheet '{sheet}'. Found: {list(xls.sheet_names)}")

    df = pd.read_excel(args.file, sheet_name=sheet)

    print_block("RUN SUMMARY (if available)")
    if run_sum is not None and len(run_sum):
        print(run_sum.to_string(index=False))
    else:
        print("No run_summary sheet found.")

    print_block(f"DATASET OVERVIEW: {sheet}")
    print("Columns:", list(df.columns))
    breakdowns(df)

    # Level 1: skill_uri (labels usually available)
    analyze_level(
        df=df,
        level_col="skill_uri",
        label_col="skill_label" if "skill_label" in df.columns else None,
        level_name="Skill URI (fine-grained)",
    )
    record_level_fit(df, "skill_uri", "Skill URI (fine-grained)")

    # Level 2: broader_skill_uri_1 (labels may or may not exist; don't require them)
    if "broader_skill_uri_1" in df.columns and df["broader_skill_uri_1"].notna().any():
        # Prefer broader label if present, else skip labels
        broader_label = "broader_skill_label_1" if "broader_skill_label_1" in df.columns else None
        analyze_level(
            df=df,
            level_col="broader_skill_uri_1",
            label_col=broader_label,
            level_name="Broader Skill URI (1-hop parent)",
        )
        record_level_fit(df, "broader_skill_uri_1", "Broader Skill URI (1-hop parent)")
    else:
        print_block("BROADER SKILL LEVEL")
        print("No broader_skill_uri_1 values found. Skipping broader-level analysis.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
