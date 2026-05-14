#!/usr/bin/env python3
"""
Build whitelist of ESCO skills linked to data science–related occupations
(ISCO 25 and 212).

Input (expected in data/ESCO_skills/):
    - occupations_en.csv
    - occupationSkillRelations_en.csv
    - skills_en.csv

Output:
    - ESCO_skills_allowed_data_domain.csv
"""

from pathlib import Path
import pandas as pd


# =========================
# Config
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]
ESCO_DIR = BASE_DIR / "data" / "ESCO_skills"

OCC_FILE = ESCO_DIR / "occupations_en.csv"
REL_FILE = ESCO_DIR / "occupationSkillRelations_en.csv"
SKILL_FILE = ESCO_DIR / "skills_en.csv"

OUTPUT_FILE = ESCO_DIR / "ESCO_skills_allowed_data_domain.csv"


# =========================
# Main
# =========================

def main():

    print("Loading ESCO files...")

    occupations = pd.read_csv(OCC_FILE)
    relations = pd.read_csv(REL_FILE)
    skills = pd.read_csv(SKILL_FILE)

    print(f"Occupations loaded: {len(occupations)}")
    print(f"Relations loaded: {len(relations)}")
    print(f"Skills loaded: {len(skills)}\n")

    print("Occupation columns:", occupations.columns.tolist())
    print("Relations columns:", relations.columns.tolist())
    print("Skills columns:", skills.columns.tolist(), "\n")

    # -------------------------
    # 1️⃣ Filter occupations by ISCO group
    # -------------------------

    occ_filtered = occupations[
        occupations["iscoGroup"].astype(str).str.startswith("25", na=False) |
        occupations["iscoGroup"].astype(str).str.startswith("212", na=False)
    ]

    allowed_occ_uris = set(occ_filtered["conceptUri"])

    print(f"Selected occupations (ISCO 25 + 212): {len(allowed_occ_uris)}")

    # -------------------------
    # 2️⃣ Filter occupation-skill relations
    # -------------------------

    rel_filtered = relations[
        relations["occupationUri"].isin(allowed_occ_uris)
    ]

    allowed_skill_uris = set(rel_filtered["skillUri"])

    print(f"Unique linked skills: {len(allowed_skill_uris)}")

    # -------------------------
    # 3️⃣ Attach skill labels
    # -------------------------

    skills_filtered = skills[
        skills["conceptUri"].isin(allowed_skill_uris)
    ][["conceptUri", "preferredLabel", "skillType"]].drop_duplicates()

    print(f"Final whitelist size: {len(skills_filtered)}")

    # -------------------------
    # 4️⃣ Save whitelist
    # -------------------------

    skills_filtered = skills_filtered.rename(columns={
        "conceptUri": "skill_uri",
        "preferredLabel": "skill_label",
        "skillType": "skill_type"
    })

    skills_filtered.to_csv(OUTPUT_FILE, index=False)

    print("\nWhitelist successfully saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
