# Econ-Embeddings: Semantic Representations of Economic Classification Systems

Pre-computed 3,072-dimensional semantic embeddings for economic classification codes, plus a pipeline for generating embeddings for new classification systems.

**Domains included:**

| Domain | Codes | Description |
|--------|------:|-------------|
| **HS products** | 1,229 | Harmonized System 4-digit trade product codes |
| **IPC4 technologies** | 650 | International Patent Classification 4-digit subclasses |
| **NAICS industries** | ~1,065 | North American Industry Classification System 6-digit codes |
| **OpenAlex concepts** | 284 | Level-1 scientific field concepts |

## How the embeddings were generated

The embeddings are produced by a two-step pipeline:

1. **LLM enrichment.** An LLM (GPT-5.4 / GPT-5.4-mini) generates enriched, cross-domain descriptions for each classification code. The prompt instructs the model to describe each code using vocabulary that bridges across domains — mentioning the underlying science, real-world products, industrial applications, and economic activities. Sibling codes from the same category are provided as context so the model can articulate what distinguishes each code from its neighbors.

2. **Semantic embedding.** The enriched descriptions are embedded using Google Gemini (`gemini-embedding-2-preview`, 3,072 dimensions, `SEMANTIC_SIMILARITY` task type). The text that gets embedded is `"{code_name}. {enriched_description}"` — just the name and core description, without the key activities or distinguishing features lists, to avoid template boilerplate inflating the cosine similarity floor.

The enrichment step is the key innovation: raw classification code names and descriptions are often too terse or jargon-specific for meaningful cross-domain comparison. The LLM enrichment bridges this gap by generating descriptions in a shared vocabulary.

## Quick start

### Installation

```bash
git clone https://github.com/shreyasgm/econ-embeddings.git
cd econ-embeddings
pip install -r requirements.txt
```

### Load pre-computed embeddings (no API keys needed)

```python
from entity_embeddings.load import load_embeddings, load_descriptions, cosine_similarity

# Load embedding vectors and code mappings
vectors, codes = load_embeddings("ipc4")
# vectors: numpy array, shape (650, 3072)
# codes: DataFrame with columns ['code_id', 'name']

# Load enriched descriptions
desc = load_descriptions("hs")

# Compute cross-domain similarity
sim, ipc4_codes, hs_codes = cosine_similarity("ipc4", "hs")
# sim: shape (650, 1229), cosine similarity between every IPC4-HS pair
```

### Find top matches

```python
import numpy as np
from entity_embeddings.load import load_embeddings

# Which scientific fields are closest to patent class G06F?
vectors_ipc4, codes_ipc4 = load_embeddings("ipc4")
vectors_concepts, codes_concepts = load_embeddings("concepts")

# L2 normalize and compute similarity
norm_ipc4 = vectors_ipc4 / np.linalg.norm(vectors_ipc4, axis=1, keepdims=True)
norm_concepts = vectors_concepts / np.linalg.norm(vectors_concepts, axis=1, keepdims=True)
sim = norm_ipc4 @ norm_concepts.T

# Get G06F's row
idx = codes_ipc4[codes_ipc4["code_id"] == "G06F"].index[0]
top_10 = np.argsort(sim[idx])[::-1][:10]
for rank, j in enumerate(top_10, 1):
    print(f"{rank}. {codes_concepts.iloc[j]['name']} (sim: {sim[idx, j]:.3f})")
```

### Use pre-computed candidate matches

Top-K candidate matches (IPC4 → each domain) are pre-computed for convenience:

```python
from entity_embeddings.load import load_candidates

# Top-25 HS products per IPC4 code
candidates = load_candidates("ipc4", "hs")
# DataFrame with: ipc4, candidate_code, candidate_name, embedding_similarity, rank

# Filter to a specific technology
c12n_products = candidates[candidates["ipc4"] == "C12N"].sort_values("rank")
```

### Generate embeddings for a new classification system

If you have a classification system not included in the pre-computed data, you can run the full pipeline. This requires API keys for OpenAI (enrichment) and Google Gemini (embedding).

```bash
export OPENAI_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
```

```python
import pandas as pd
from entity_embeddings.enrich import enrich_codes
from entity_embeddings.embed import embed_descriptions

# Prepare your data
df = pd.DataFrame({
    "code_id": ["541511", "541512"],
    "name": ["Custom Computer Programming", "Computer Systems Design"],
    "description": ["Writing, testing, supporting custom software", "Planning and designing computer systems"],
    "category": ["Professional Services", "Professional Services"],
})

# Step 1: Enrich descriptions
enriched = enrich_codes(df, domain_type="industry (NAICS classification)")

# Step 2: Embed
vectors = embed_descriptions(enriched)

# Now use vectors for similarity, clustering, etc.
```

See `examples/` for complete runnable scripts.

## Pre-computed data

All pre-computed data lives in `data/`:

| File pattern | Description |
|---|---|
| `enriched_descriptions_{domain}.parquet` | LLM-enriched descriptions with metadata |
| `embeddings_{domain}.npy` | Embedding vectors, shape (n_codes, 3072) |
| `embeddings_{domain}_codes.parquet` | Code-to-name mapping (row i ↔ vector i) |
| `embedding_candidates_ipc4_{domain}.parquet` | Pre-computed top-K matches from IPC4 |

### Enriched description columns

| Column | Description |
|--------|-------------|
| `code_id` | Classification code identifier |
| `name` | Human-readable name |
| `enriched_description` | LLM-generated cross-domain description (2-3 sentences) |
| `key_activities_or_topics` | List of 5-10 specific activities under this code |
| `distinguishing_features` | List of 2-4 features distinguishing from sibling codes |
| `reasoning` | LLM reasoning trace |
| `embedding_text` | Exact text that was embedded |
| `category` | Grouping category within the domain |

## Repository structure

```
entity_embeddings/
├── data/                          # Pre-computed outputs (~44 MB)
│   ├── enriched_descriptions_*.parquet
│   ├── embeddings_*.npy
│   ├── embeddings_*_codes.parquet
│   └── embedding_candidates_*.parquet
├── entity_embeddings/             # Python package
│   ├── __init__.py
│   ├── load.py                    # Load pre-computed data
│   ├── enrich.py                  # LLM enrichment pipeline
│   └── embed.py                   # Embedding generation
├── examples/
│   ├── quickstart.py              # Uses pre-computed data only
│   └── generate_new_embeddings.py # Full pipeline with API calls
├── requirements.txt
└── README.md
```

## Citation

If you use anything from this repository, please cite it as:

```bibtex
@software{gadginmatha2026econembeddings,
  author       = {Gadgin Matha, Shreyas},
  title        = {Econ-Embeddings: Semantic Representations of Economic Classification Systems},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/shreyasgm/econ-embeddings}
}
```

## License

MIT
