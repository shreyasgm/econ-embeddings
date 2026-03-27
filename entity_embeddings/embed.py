"""Generate semantic embeddings from enriched descriptions.

This module embeds enriched description text using Google's Gemini embedding
model and provides utilities for computing cross-domain similarity.

Usage:
    import pandas as pd
    from entity_embeddings.embed import embed_descriptions, top_k_similar

    # From enriched descriptions (output of enrich.py)
    enriched_df = pd.read_parquet("enriched_descriptions_ipc4.parquet")
    vectors = embed_descriptions(enriched_df)

    # Or embed arbitrary texts
    from entity_embeddings.embed import embed_texts
    vectors = embed_texts(["Electric digital data processing", "Organic chemistry"])
"""

import logging
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def embed_texts(
    texts: list[str],
    model: str = "gemini-embedding-2-preview",
    dimensions: int = 3072,
    task_type: str = "SEMANTIC_SIMILARITY",
    batch_size: int = 100,
    api_key: str | None = None,
) -> np.ndarray:
    """Embed a list of texts using Google Gemini.

    Args:
        texts: List of text strings to embed.
        model: Gemini embedding model name.
        dimensions: Output embedding dimensionality.
        task_type: Gemini task type hint.
        batch_size: Texts per API call.
        api_key: Google API key. If None, reads from GOOGLE_API_KEY env var.

    Returns:
        numpy array of shape (len(texts), dimensions), dtype float32.
    """
    from google import genai
    from google.genai import types

    key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError("No API key provided. Set GOOGLE_API_KEY or pass api_key=")

    client = genai.Client(api_key=key)

    n_batches = (len(texts) - 1) // batch_size + 1
    all_embeddings = []
    for i in tqdm(
        range(0, len(texts), batch_size),
        total=n_batches,
        desc="Embedding",
        unit="batch",
    ):
        batch = texts[i : i + batch_size]
        response = client.models.embed_content(
            model=model,
            contents=batch,
            config=types.EmbedContentConfig(
                output_dimensionality=dimensions,
                task_type=task_type,
            ),
        )
        for emb in response.embeddings:
            all_embeddings.append(emb.values)

    return np.array(all_embeddings, dtype=np.float32)


def embed_descriptions(
    enriched_df: pd.DataFrame,
    text_column: str = "embedding_text",
    **kwargs,
) -> np.ndarray:
    """Embed enriched descriptions from a DataFrame.

    Args:
        enriched_df: DataFrame with an embedding text column
            (output of entity_embeddings.enrich.enrich_codes).
        text_column: Column containing the text to embed.
        **kwargs: Passed to embed_texts (model, dimensions, api_key, etc.).

    Returns:
        numpy array of shape (len(df), dimensions), dtype float32.
        Row i corresponds to enriched_df.iloc[i].
    """
    texts = enriched_df[text_column].tolist()
    return embed_texts(texts, **kwargs)


def top_k_similar(
    source_vectors: np.ndarray,
    target_vectors: np.ndarray,
    source_codes: pd.DataFrame,
    target_codes: pd.DataFrame,
    k: int = 25,
    self_match: bool = False,
) -> pd.DataFrame:
    """Find top-K most similar target codes for each source code.

    Args:
        source_vectors: Shape (n_source, dim).
        target_vectors: Shape (n_target, dim).
        source_codes: DataFrame with 'code_id' and 'name' for source.
        target_codes: DataFrame with 'code_id' and 'name' for target.
        k: Number of top matches per source code.
        self_match: If True, allow a code to match itself
            (set False for within-domain matching).

    Returns:
        DataFrame with columns: source_code, target_code, target_name,
        cosine_similarity, rank.
    """
    # L2 normalize
    src_norm = source_vectors / np.linalg.norm(source_vectors, axis=1, keepdims=True)
    tgt_norm = target_vectors / np.linalg.norm(target_vectors, axis=1, keepdims=True)

    sim_matrix = src_norm @ tgt_norm.T

    if not self_match:
        np.fill_diagonal(sim_matrix, -np.inf)

    effective_k = min(k, sim_matrix.shape[1] - (0 if self_match else 1))

    rows = []
    for i in range(len(source_codes)):
        top_indices = np.argsort(sim_matrix[i])[::-1][:effective_k]
        for rank, j in enumerate(top_indices, 1):
            rows.append({
                "source_code": source_codes.iloc[i]["code_id"],
                "target_code": target_codes.iloc[j]["code_id"],
                "target_name": target_codes.iloc[j]["name"],
                "cosine_similarity": float(sim_matrix[i, j]),
                "rank": rank,
            })

    return pd.DataFrame(rows)
