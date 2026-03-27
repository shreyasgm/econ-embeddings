**Econ-Embeddings: Semantic Representations of Economic Classification Systems**

I've published pre-computed 3,072-dimensional semantic embeddings for HS products (1,229), IPC4 patent classes (650), NAICS industries (~1,065), and OpenAlex scientific fields (284).

Rather than naively embedding raw classification names — which are terse, jargon-heavy, and produce a similarity space where everything looks vaguely alike — I first used an LLM to generate enriched, cross-domain descriptions for every code. The enrichment prompt asks the model to describe each code using vocabulary that bridges across domains (underlying science, real-world products, industrial applications), while distinguishing it from sibling codes. These enriched descriptions were then embedded with Google's Gemini embedding model.

The result is a shared semantic space where you can compute meaningful similarity between any pair of codes, within or across classification systems. The repo also includes a pipeline for generating embeddings for other classification systems (SITC, CPC, ISIC, etc.).

**Repo:** [github.com/shreyasgm/econ-embeddings](https://github.com/shreyasgm/econ-embeddings) — loading the embeddings requires only `numpy` and `pandas`, no API keys.
