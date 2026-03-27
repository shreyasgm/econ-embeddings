"""Enrich classification code descriptions via LLM for embedding.

This module takes a DataFrame of classification codes with basic descriptions
and produces enriched, cross-domain descriptions suitable for semantic embedding.
The enrichment uses an LLM to generate descriptions that bridge across domains
(science, technology, production, industry), making the resulting embeddings
useful for cross-domain relatedness tasks.

Usage:
    import pandas as pd
    from entity_embeddings.enrich import enrich_codes

    # Prepare your data as a DataFrame
    df = pd.DataFrame({
        "code_id": ["541511", "541512", "541519"],
        "name": ["Custom Computer Programming", "Computer Systems Design", "Other CS Services"],
        "description": ["Writing, testing, and supporting custom software", ...],
        "category": ["Professional Services", "Professional Services", "Professional Services"],
    })

    # Enrich (requires OPENAI_API_KEY environment variable)
    enriched = enrich_codes(df, domain_type="industry (NAICS classification)")
"""

import asyncio
import logging
import os

import pandas as pd
from openai import AsyncOpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# --- Pydantic model for structured output ---

class EnrichmentResponse(BaseModel):
    """Structured output from the enrichment LLM call."""

    reasoning: str
    enriched_description: str
    key_activities_or_topics: list[str]
    distinguishing_features: list[str]


# --- Prompts ---

ENRICHMENT_SYSTEM_PROMPT = """You are creating precise, disambiguated descriptions of classification codes.
These descriptions will be used to measure cross-domain relatedness between
classification systems (e.g., matching technology classes to scientific fields,
products, and industries).

You must respond with a JSON object containing:
- "reasoning": Your step-by-step thinking about what this code covers (1-2 sentences)
- "enriched_description": A 2-3 sentence description satisfying the requirements below
- "key_activities_or_topics": A list of 5-10 specific activities, technologies,
  materials, or topics that fall under this code
- "distinguishing_features": A list of 2-4 features that distinguish this code
  from its most easily confused siblings"""


def _build_enrichment_prompt(
    code_id: str,
    name: str,
    description: str | None,
    domain_type: str,
    category_name: str,
    sibling_names: list[str],
) -> str:
    """Build the per-code enrichment user prompt."""
    desc_text = description if description else "None available"
    siblings_text = "\n".join(f"  - {s}" for s in sibling_names[:40])

    return f"""Code: {code_id} — {name}
Existing description: {desc_text}

Sibling codes in the same category ({category_name}):
{siblings_text}

Your task: Create a description of this {domain_type} code that would allow
someone from an ADJACENT domain to understand what it covers. For example,
if this is a technology class, a scientist, product engineer, or economist
should be able to read the description and identify which scientific fields,
products, or industries relate to it.

Requirements:
(a) Clearly state what this code COVERS — the specific activities, topics, or products
(b) Clearly state what this code does NOT cover, referencing specific sibling codes
(c) Use concrete examples, materials, processes, or applications
(d) Use vocabulary that bridges across domains — mention the underlying science,
    the real-world products, the industrial applications, and the economic activities
    associated with this code, not just the classification-system-specific terminology"""


# --- Core enrichment logic ---


async def _enrich_one(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    system_prompt: str,
    user_prompt: str,
    model: str,
) -> EnrichmentResponse:
    """Make a single async enrichment call."""
    async with semaphore:
        completion = await client.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=EnrichmentResponse,
            temperature=0,
        )
        return completion.choices[0].message.parsed


def enrich_codes(
    df: pd.DataFrame,
    domain_type: str,
    model: str = "gpt-4.1-mini",
    concurrency: int = 20,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Enrich classification code descriptions via LLM.

    Args:
        df: DataFrame with required columns:
            - code_id: Unique identifier for each code
            - name: Human-readable name
            - description: Existing description (can be None/NaN)
            - category: Grouping category (used for sibling context)
        domain_type: Human-readable domain label for the prompt, e.g.
            "technology class", "product (HS trade classification)",
            "industry (NAICS classification)", "scientific field"
        model: OpenAI model to use. Default: gpt-4.1-mini.
        concurrency: Max concurrent API requests. Default: 20.
        api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.

    Returns:
        DataFrame with columns: code_id, name, enriched_description,
        key_activities_or_topics, distinguishing_features, reasoning,
        embedding_text, category
    """
    required_cols = {"code_id", "name", "description", "category"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("No API key provided. Set OPENAI_API_KEY or pass api_key=")

    # Build sibling lookup
    siblings = df.groupby("category")["code_id"].apply(list).to_dict()

    # Build tasks
    tasks = []
    for _, row in df.iterrows():
        sibling_codes = siblings.get(row["category"], [])
        sibling_rows = df[
            df["code_id"].isin(sibling_codes) & (df["code_id"] != row["code_id"])
        ]
        sibling_names = [
            f"{r['code_id']} — {r['name']}" for _, r in sibling_rows.iterrows()
        ]

        user_prompt = _build_enrichment_prompt(
            code_id=row["code_id"],
            name=row["name"],
            description=row["description"] if pd.notna(row["description"]) else None,
            domain_type=domain_type,
            category_name=row["category"],
            sibling_names=sibling_names,
        )
        tasks.append({"row": row, "user_prompt": user_prompt})

    # Run async
    async def _run():
        client = AsyncOpenAI(api_key=key, max_retries=3)
        semaphore = asyncio.Semaphore(concurrency)
        results = []

        from tqdm.asyncio import tqdm as atqdm

        coros = [
            _enrich_one(client, semaphore, ENRICHMENT_SYSTEM_PROMPT, t["user_prompt"], model)
            for t in tasks
        ]
        responses = await atqdm.gather(*coros, desc="Enriching", unit="code")
        return responses

    responses = asyncio.run(_run())

    # Build output
    output_rows = []
    for task, response in zip(tasks, responses):
        row = task["row"]
        embedding_text = f"{row['name']}. {response.enriched_description}"
        output_rows.append({
            "code_id": row["code_id"],
            "name": row["name"],
            "enriched_description": response.enriched_description,
            "key_activities_or_topics": list(response.key_activities_or_topics),
            "distinguishing_features": list(response.distinguishing_features),
            "reasoning": response.reasoning,
            "embedding_text": embedding_text,
            "category": row["category"],
        })

    return pd.DataFrame(output_rows)
