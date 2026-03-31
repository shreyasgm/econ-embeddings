"""Econ-Embeddings: Semantic Representations of Economic Classification Systems.

Pre-computed 3072-dimensional embeddings for HS products, IPC4 technologies,
NAICS industries, and OpenAlex scientific concepts, plus a pipeline
for generating new embeddings for other classification systems.
"""

__version__ = "0.1.0"

from entity_embeddings.load import concord
