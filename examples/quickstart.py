"""Quickstart: Load pre-computed embeddings and find cross-domain matches.

This script demonstrates the most common use case: loading pre-computed
embeddings and computing semantic similarity between classification codes
across different domains (e.g., finding which scientific fields are most
related to a given patent technology class).

No API keys required — uses only the pre-computed data shipped with this repo.
"""

import numpy as np
import pandas as pd

from entity_embeddings.load import (
    cosine_similarity,
    load_candidates,
    load_descriptions,
    load_embeddings,
)


def example_1_load_and_inspect():
    """Load embeddings and inspect what's available."""
    print("=" * 60)
    print("Example 1: Load and inspect embeddings")
    print("=" * 60)

    # Load IPC4 technology embeddings
    vectors, codes = load_embeddings("ipc4")
    print(f"IPC4: {vectors.shape[0]} codes, {vectors.shape[1]} dimensions")
    print(f"First 5 codes:\n{codes.head()}\n")

    # Load enriched descriptions to see what was embedded
    desc = load_descriptions("ipc4")
    row = desc[desc["code_id"] == "G06F"].iloc[0]
    print(f"G06F ({row['name']}):")
    print(f"  Description: {row['enriched_description'][:200]}...")
    print(f"  Key topics: {row['key_activities_or_topics'][:3]}")
    print()


def example_2_cross_domain_similarity():
    """Find which scientific fields are most related to a technology."""
    print("=" * 60)
    print("Example 2: Cross-domain similarity (IPC4 → Scientific fields)")
    print("=" * 60)

    # Compute full similarity matrix
    sim, ipc4_codes, concept_codes = cosine_similarity("ipc4", "concepts")

    # Find top scientific fields for G06F (Electric Digital Data Processing)
    ipc4_idx = ipc4_codes[ipc4_codes["code_id"] == "G06F"].index[0]
    similarities = sim[ipc4_idx]
    top_indices = np.argsort(similarities)[::-1][:10]

    print("Top 10 scientific fields for G06F (Electric Digital Data Processing):")
    for rank, idx in enumerate(top_indices, 1):
        name = concept_codes.iloc[idx]["name"]
        score = similarities[idx]
        print(f"  {rank:2d}. {name:<40s} (similarity: {score:.3f})")
    print()


def example_3_precomputed_candidates():
    """Use pre-computed top-K candidates (faster than full matrix)."""
    print("=" * 60)
    print("Example 3: Pre-computed candidates (IPC4 → HS products)")
    print("=" * 60)

    candidates = load_candidates("ipc4", "hs")
    print(f"Total candidate pairs: {len(candidates):,}")
    print(f"Unique IPC4 codes: {candidates['source_code'].nunique()}")

    # Top products for C12N (Micro-organisms or Enzymes)
    c12n = candidates[candidates["source_code"] == "C12N"].sort_values("rank")
    print("\nTop 10 HS products for C12N (Micro-organisms or Enzymes):")
    for _, row in c12n.head(10).iterrows():
        print(
            f"  {row['rank']:2.0f}. HS {row['candidate_code']} — "
            f"{row['candidate_name'][:50]:<50s} "
            f"(sim: {row['embedding_similarity']:.3f})"
        )
    print()


def example_4_product_technology_matching():
    """Find which technologies relate to a specific product."""
    print("=" * 60)
    print("Example 4: Reverse lookup — HS product → IPC4 technologies")
    print("=" * 60)

    # Use the IPC4→HS candidates and filter for a specific HS code
    candidates = load_candidates("ipc4", "hs")

    # Find all IPC4 codes that have HS 3004 (Medicaments) as a top candidate
    medicaments = candidates[candidates["candidate_code"] == "3004"].sort_values(
        "embedding_similarity", ascending=False
    )

    print("IPC4 technologies most related to HS 3004 (Medicaments):")
    ipc4_desc = load_descriptions("ipc4")
    for _, row in medicaments.head(10).iterrows():
        name = ipc4_desc[ipc4_desc["code_id"] == row["source_code"]]["name"].values[0]
        print(
            f"  {row['source_code']} — {name[:45]:<45s} (sim: {row['embedding_similarity']:.3f})"
        )
    print()


if __name__ == "__main__":
    example_1_load_and_inspect()
    example_2_cross_domain_similarity()
    example_3_precomputed_candidates()
    example_4_product_technology_matching()
