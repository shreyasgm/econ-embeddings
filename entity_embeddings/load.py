"""Convenience functions for loading pre-computed embeddings and descriptions."""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Available domains
DOMAINS = ["concepts", "ipc4", "hs", "naics"]

# Domain metadata
DOMAIN_INFO = {
    "concepts": {
        "description": "OpenAlex Level-1 scientific concepts (284 fields)",
        "code_column": "code_id",
        "example_codes": ["artificial intelligence", "quantum mechanics", "economics"],
    },
    "ipc4": {
        "description": "IPC 4-digit technology classes (650 patent categories)",
        "code_column": "code_id",
        "example_codes": ["G06F", "H01L", "C07K"],
    },
    "hs": {
        "description": "HS 4-digit product codes (1,229 traded products)",
        "code_column": "code_id",
        "example_codes": ["8471", "2709", "3004"],
    },
    "naics": {
        "description": "NAICS 6-digit industry codes (~1,065 industries)",
        "code_column": "code_id",
        "example_codes": ["541511", "325411", "334413"],
    },
}


def load_embeddings(domain: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Load pre-computed embedding vectors and their code mappings.

    Args:
        domain: One of 'concepts', 'ipc4', 'hs', 'naics'.

    Returns:
        Tuple of (vectors, codes) where:
            - vectors: numpy array of shape (n_codes, 3072), float32
            - codes: DataFrame with columns ['code_id', 'name'],
                     row i corresponds to vectors[i]
    """
    if domain not in DOMAINS:
        raise ValueError(f"Unknown domain '{domain}'. Choose from: {DOMAINS}")

    vectors = np.load(DATA_DIR / f"embeddings_{domain}.npy")
    codes = pd.read_parquet(DATA_DIR / f"embeddings_{domain}_codes.parquet")

    assert len(vectors) == len(codes), (
        f"Vector/code mismatch: {len(vectors)} vectors, {len(codes)} codes"
    )
    return vectors, codes


def load_descriptions(domain: str) -> pd.DataFrame:
    """Load enriched descriptions for a domain.

    Args:
        domain: One of 'concepts', 'ipc4', 'hs', 'naics'.

    Returns:
        DataFrame with columns:
            - code_id: The classification code identifier
            - name: Human-readable name
            - enriched_description: LLM-generated cross-domain description
            - key_activities_or_topics: List of 5-10 specific activities
            - distinguishing_features: List of 2-4 distinguishing features
            - reasoning: LLM reasoning trace
            - embedding_text: Text that was actually embedded ("{name}. {description}")
            - category: Grouping category within the domain
    """
    if domain not in DOMAINS:
        raise ValueError(f"Unknown domain '{domain}'. Choose from: {DOMAINS}")

    return pd.read_parquet(DATA_DIR / f"enriched_descriptions_{domain}.parquet")


def load_candidates(source: str, target: str) -> pd.DataFrame:
    """Load pre-computed top-K candidate matches between two domains.

    Pre-computed candidates are available for IPC4 as the source domain
    matched against all four target domains.

    Args:
        source: Source domain (currently only 'ipc4' available).
        target: Target domain ('concepts', 'hs', 'naics', or 'ipc4').

    Returns:
        DataFrame with columns:
            - ipc4: Source IPC4 code
            - candidate_code: Target domain code
            - candidate_name: Target domain name
            - embedding_similarity: Cosine similarity score
            - rank: Rank within this source code's candidates (1 = most similar)
    """
    pair_name = f"{source}_{target}"
    path = DATA_DIR / f"embedding_candidates_{pair_name}.parquet"
    if not path.exists():
        available = [p.stem.replace("embedding_candidates_", "") for p in DATA_DIR.glob("embedding_candidates_*.parquet")]
        raise FileNotFoundError(
            f"No candidates for pair '{pair_name}'. Available: {available}"
        )
    return pd.read_parquet(path)


def cosine_similarity(
    domain_a: str,
    domain_b: str,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Compute full cosine similarity matrix between two domains.

    Args:
        domain_a: First domain.
        domain_b: Second domain.

    Returns:
        Tuple of (similarity_matrix, codes_a, codes_b) where:
            - similarity_matrix: shape (n_a, n_b), values in [-1, 1]
            - codes_a: DataFrame for rows
            - codes_b: DataFrame for columns
    """
    vecs_a, codes_a = load_embeddings(domain_a)
    vecs_b, codes_b = load_embeddings(domain_b)

    # L2 normalize
    norm_a = vecs_a / np.linalg.norm(vecs_a, axis=1, keepdims=True)
    norm_b = vecs_b / np.linalg.norm(vecs_b, axis=1, keepdims=True)

    sim = norm_a @ norm_b.T
    return sim, codes_a, codes_b
