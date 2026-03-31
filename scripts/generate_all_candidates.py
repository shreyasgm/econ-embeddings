"""Generate pre-computed top-K candidate matches for all domain pairs.

Uses existing embeddings (no API calls) to compute cosine similarity
and extract top-25 matches for every (source, target) domain combination.

Usage:
    uv run python scripts/generate_all_candidates.py
"""

from itertools import product

from entity_embeddings.embed import top_k_similar
from entity_embeddings.load import DATA_DIR, DOMAINS, load_embeddings

K = 25


def generate_candidates(source: str, target: str) -> None:
    """Generate and save top-K candidates for one (source, target) pair."""
    src_vectors, src_codes = load_embeddings(source)
    tgt_vectors, tgt_codes = load_embeddings(target)

    self_match = source != target

    df = top_k_similar(
        source_vectors=src_vectors,
        target_vectors=tgt_vectors,
        source_codes=src_codes,
        target_codes=tgt_codes,
        k=K,
        self_match=self_match,
    )

    # Rename columns to match the candidate file convention
    df = df.rename(
        columns={
            "target_code": "candidate_code",
            "target_name": "candidate_name",
            "cosine_similarity": "embedding_similarity",
        }
    )

    out_path = DATA_DIR / f"embedding_candidates_{source}_{target}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  {source} → {target}: {len(df):,} pairs → {out_path.name}")


def main():
    print(
        f"Generating top-{K} candidates for all {len(DOMAINS)}×{len(DOMAINS)} = {len(DOMAINS) ** 2} domain pairs\n"
    )

    for source, target in product(DOMAINS, repeat=2):
        generate_candidates(source, target)

    # Verify
    files = sorted(DATA_DIR.glob("embedding_candidates_*.parquet"))
    print(f"\nDone. {len(files)} candidate files in {DATA_DIR}/")


if __name__ == "__main__":
    main()
