"""Vendored pure-function modules from the server judgment chain (experiment 01/04).

Provenance (synced 2026-11-04, commit 20efbdb; files copied verbatim — keep them
byte-identical to the sources so future server syncs diff cleanly):

- fact_extraction_v2.py, semantic_review_v2.py, semantic_review_v21.py,
  semantic_review_v22.py, longmemeval.py
    <- experiments/01_evidence_grounded_memory_evolution/implementation/evidence_memory/
- normalization.py, pair_retrieval.py, relation_judge.py
    <- experiments/04_fact_attribute_pair_retrieval/src/

Only modules with no platform locks (no fcntl) and no LLM client imports are
vendored: the server StateStore (store.py) stays experiment-side and is NOT
part of this bridge. LLM transport is injected by coregraph.evidence_bridge.
"""
