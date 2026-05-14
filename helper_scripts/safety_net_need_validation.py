#!/usr/bin/env python3
"""
Quick ESCOX sanity check on a few sample module texts.

Purpose:
- verify whether ESCOX at threshold 0.55 picks up expected skills
- inspect local misses such as mathematics / Python / machine learning

Run:
python helper_scripts/sanity_check_escox_samples.py
"""

from esco_skill_extractor import SkillExtractor

THRESHOLD = 0.60
DEVICE = "cpu"   # or None

samples = [
    {
        "name": "Mathematics: Linear Algebra",
        "text": """Mathematics: Linear Algebra
▪ explain fundamental notions in the domain of linear equation systems.
▪ exemplify properties of vectors and vector spaces.
▪ summarize characteristics of linear and affine mappings.
▪ identify important relations in analytical geometry.
▪ utilize different methods for matrix decomposition."""
    },
    {
        "name": "Introduction to Academic Work for IT and Technology",
        "text": """Introduction to Academic Work for IT and Technology
▪ explain what science is and why science is needed (including in practice-based studies and
professional practice).
▪ name and apply theories, methods, and models in IT and technology.
▪ find, analyze, and classify academic literature and types of sources.
▪ prepare academic papers independently."""
    },
    {
        "name": "Project: Object Oriented and Functional Programming with Python",
        "text": """Project: Object Oriented and Functional Programming
with Python
▪ explain basic notions in object-oriented programming such as functions and classes.
▪ understand object-oriented programming concepts and their relation to software design and
engineering.
▪ describe advanced function concepts in Python.
▪ recognize important ideas from functional programming.
▪ recall important libraries for functional programming in Python."""
    },
]

extractor = SkillExtractor(skills_threshold=THRESHOLD, device=DEVICE)

print("=" * 100)
print(f"ESCOX SANITY CHECK | threshold={THRESHOLD} | device={DEVICE}")
print("=" * 100)

for sample in samples:
    skill_uris = extractor.get_skills([sample["text"]])[0]

    print("\n" + "-" * 100)
    print(f"SAMPLE: {sample['name']}")
    print("-" * 100)
    print("TEXT:")
    print(sample["text"])
    print("\nEXTRACTED SKILL URIS:")
    if not skill_uris:
        print("  [no skills extracted]")
    else:
        for uri in skill_uris:
            print(f"  {uri}")

print("\n" + "=" * 100)
print("Done.")
print("=" * 100)