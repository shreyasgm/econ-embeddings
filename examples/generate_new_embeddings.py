"""Generate embeddings for a new classification system.

This script demonstrates the full pipeline: enrich descriptions via LLM,
then embed them. Use this when you have a classification system that isn't
already included in the pre-computed data.

Requires API keys:
    - OPENAI_API_KEY (for GPT enrichment)
    - GOOGLE_API_KEY (for Gemini embeddings)
"""

import numpy as np
import pandas as pd

from entity_embeddings.embed import embed_descriptions, top_k_similar
from entity_embeddings.enrich import enrich_codes
from entity_embeddings.load import load_embeddings


def example_enrich_and_embed():
    """Enrich and embed a small set of custom codes."""

    # Step 1: Prepare your data as a DataFrame.
    # Required columns: code_id, name, description, category
    df = pd.DataFrame(
        {
            "code_id": ["A01", "A02", "B01", "B02"],
            "name": [
                "Wheat farming",
                "Rice farming",
                "Iron ore mining",
                "Copper ore mining",
            ],
            "description": [
                "Growing wheat and other cereal grains",
                "Cultivating rice in paddies",
                "Extracting iron ore from open-pit and underground mines",
                "Mining copper ores and concentrates",
            ],
            "category": ["Agriculture", "Agriculture", "Mining", "Mining"],
        }
    )

    # Step 2: Enrich descriptions.
    # The domain_type tells the LLM what kind of codes these are,
    # so it can generate appropriate cross-domain descriptions.
    print("Enriching descriptions...")
    enriched = enrich_codes(
        df,
        domain_type="economic activity",
        model="gpt-4.1-mini",  # or any OpenAI model with structured output support
    )
    print(f"Enriched {len(enriched)} codes")
    print(f"Sample embedding text: {enriched.iloc[0]['embedding_text'][:200]}...")
    print()

    # Step 3: Embed the enriched descriptions.
    print("Generating embeddings...")
    vectors = embed_descriptions(enriched)
    print(f"Embedding shape: {vectors.shape}")
    print()

    # Step 4: Compare against pre-computed domains.
    # Find which IPC4 technologies are most related to each custom code.
    ipc4_vectors, ipc4_codes = load_embeddings("ipc4")

    codes_df = enriched[["code_id", "name"]].copy()
    matches = top_k_similar(
        source_vectors=vectors,
        target_vectors=ipc4_vectors,
        source_codes=codes_df,
        target_codes=ipc4_codes,
        k=5,
        self_match=True,  # different domains, so self-match is fine
    )

    print("Top 5 IPC4 matches per custom code:")
    for code_id in df["code_id"]:
        code_matches = matches[matches["source_code"] == code_id]
        name = df[df["code_id"] == code_id]["name"].values[0]
        print(f"\n  {code_id} — {name}:")
        for _, row in code_matches.iterrows():
            print(
                f"    {row['rank']}. {row['target_code']} — "
                f"{row['target_name'][:50]} (sim: {row['cosine_similarity']:.3f})"
            )


if __name__ == "__main__":
    example_enrich_and_embed()
