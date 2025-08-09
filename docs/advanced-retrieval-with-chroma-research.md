# Advanced Retrieval with Chroma — Research and Recommendations

## Overview
This document distills actionable techniques from “RAG I: Advanced Techniques with Chroma” by Sulaiman Shamasna (based on the OpenAI course “Advanced Retrieval for AI with Chroma”) and maps them to concrete improvements for the AI News Summarizer. The focus: retrieval quality, robustness, and user experience.

References:
- RAG I: Advanced Techniques with Chroma (Medium)
- OpenAI short course: Advanced Retrieval for AI with Chroma

## Methodology
- Reviewed the article’s techniques: Query Expansion (generated answers, multi-query), Cross-Encoder re-ranking, Query Adapters, chunking/tokenization considerations, UMAP-based diagnostics.
- Compared against current architecture (OpenAI embeddings + ChromaDB + simple similarity threshold).
- Identified integration points with pipeline, searcher, and CLI/Streamlit UI.

## Key Findings
- Pure vector similarity is necessary but insufficient. General queries often underperform; distractors can degrade RAG output quality.
- Query-time augmentation and post-retrieval re-ranking substantially improve precision and recall.
- Chunking/token-awareness and metadata constraints materially affect retrieval quality.

## Recommendations (Prioritized)
1) Adaptive Thresholds + Query Classification (Short-term, high ROI)
- Problem: General queries like “health” or “virus” produce 0.45–0.55 scores—useful but filtered by 0.7 threshold.
- Recommendation: Implement adaptive thresholds by query type (specific vs general vs exploratory) as a light-weight first step. Expose CLI flags: --broad/--strict/--threshold.
- Why now: Minimal code, immediate UX improvement, aligns with article’s insight on query intent.

2) Multi-Query Expansion (Medium-term)
- Technique: Use LLM to generate 3–5 paraphrases/related questions; issue parallel vector searches; merge and deduplicate results.
- Ranking: Use similarity for initial cut; optionally cap per-query contribution (e.g., top-2 each) to avoid over-domination.
- Benefit: Expands coverage for broad/under-specified queries; reduces cold-start misses.
- Integration: Add a flag (e.g., --expand N) and pipeline path that aggregates results across sub-queries.

3) Hypothetical Answer Expansion (HyDE) (Medium-term)
- Technique: Generate a brief “imagined” answer for the user query; append to query before embedding.
- Benefit: Moves the query vector nearer to answer-like regions; notably boosts factual lookup in news summaries.
- Guardrails: Limit length; cache hyde text per query to control costs; disable for very short queries.

4) Cross-Encoder Re-ranking (Medium/Long-term)
- Technique: After initial retrieval (k≈20), score (query, document) pairs with a cross-encoder (e.g., bge-reranker, Cohere, or OpenAI re-rank API when available).
- Benefit: Reorders long tail, suppresses distractors; improves top-3 precision.
- Considerations: Extra latency and cost; make it opt-in (--rerank) and cache results.

5) Token- and Field-Aware Chunking (Medium-term)
- Current: We embed a single concatenation: title | summary | topics | body_excerpt.
- Recommendation: Introduce small chunks (e.g., 300–500 tokens) with overlaps for long bodies; keep title/summary in every chunk’s metadata; store a “parent” article_id for regrouping in UI.
- Benefit: Better recall for details; limits truncation; works well with re-ranking.

6) Metadata-Driven Filtering (Short/Mid-term)
- Technique: Use Chroma where filters (date ranges, source domains, topic tags) to constrain search space before embedding similarity.
- Benefit: Reduces distractors; improves precision and speed for scoped queries (e.g., last 7 days).
- Implementation: Add optional filters in CLI/UI (e.g., --since YYYY-MM-DD, --domain bbc.com, --topic covid-19).

7) Result Diversification (MMR) (Medium-term)
- Technique: Maximal Marginal Relevance on top-k results to balance relevance and diversity.
- Benefit: Avoids near-duplicate top results; improves coverage across subtopics.
- Parameter: λ in [0.2, 0.7] to control diversity vs relevance.

8) Diagnostics and Quality Monitoring (Short-term)
- Technique: Log score distributions per query type; track recall@k and precision@k for a small labeled set; optionally visualize with UMAP offline.
- Benefit: Evidence-based threshold and pipeline tuning; faster debugging of distractors.

9) Query Adapters (Long-term, experimental)
- Technique: Learn a linear adapter matrix from labeled (query, doc, label) pairs to warp query embeddings.
- Benefit: Application-specific retrieval improvements without retraining embedding model.
- Caveat: Requires dataset; start with synthetic labels; revisit once real user feedback accumulates.

## Tailored Improvement Plan for This Repo
A. Short-term (this sprint)
- Implement adaptive thresholds (config + CLI flags) and default to adaptive mode.
- Add metadata filters: date range (extracted_at), domain(host), and topic tag filter.
- Add export stability tests; ensure export skips embedding calls (done).

B. Medium-term (next sprint)
- Add multi-query expansion (--expand 3) and HyDE (--hyde) behind flags.
- Add simple MMR re-ranking on the initial top-k (no external model needed).
- Prototype cross-encoder re-ranking with a lightweight model and cache.

C. Long-term
- Split articles into token-aware chunks; add parent-child linkage.
- Stand up evaluation harness: curated queries, gold labels, metrics dashboard.
- Explore adapter training once we have usage data.

## Cost-Benefit Outline
- Adaptive thresholds, filters, MMR: Low cost, immediate value.
- Multi-query and HyDE: Moderate cost (LLM calls), high benefit on broad queries; cache aggressively.
- Cross-encoder re-ranking: Medium-high cost; large precision gains for top-3.
- Query adapters: High complexity; defer until we have feedback data.

## Risks and Mitigations
- Latency: Use flags, caching, and k-stage retrieval (k1 large → rerank → k2 small).
- Cost: Limit tokens, cache expansions, batch operations.
- Complexity creep: Keep base path simple; add advanced features as optional layers.

## Implementation Considerations
- Keep a simple, robust baseline path (current vector search) for reliability.
- Introduce advanced modes incrementally via CLI/UI flags and config toggles.
- Centralize query processing pipeline (expansion → retrieval → rerank → filter → format).
- Document fallbacks and timeouts to maintain responsiveness.

## Next Steps
1. Add adaptive threshold + CLI flags.
2. Add where-filters for date/domain/topics.
3. Implement MMR diversification.
4. Prototype multi-query expansion (flag-gated) and cache.
5. Add basic metrics logging and a small evaluation set in tests/.
# Advanced Retrieval with Chroma — Research and Recommendations

## Overview
This document distills actionable techniques from “RAG I: Advanced Techniques with Chroma” by Sulaiman Shamasna (based on the OpenAI course “Advanced Retrieval for AI with Chroma”) and maps them to concrete improvements for the AI News Summarizer. The focus: retrieval quality, robustness, and user experience.

References:
- RAG I: Advanced Techniques with Chroma (Medium)
- OpenAI short course: Advanced Retrieval for AI with Chroma

## Methodology
- Reviewed the article’s techniques: Query Expansion (generated answers, multi-query), Cross-Encoder re-ranking, Query Adapters, chunking/tokenization considerations, UMAP-based diagnostics.
- Compared against current architecture (OpenAI embeddings + ChromaDB + simple similarity threshold).
- Identified integration points with pipeline, searcher, and CLI/Streamlit UI.

## Key Findings
- Pure vector similarity is necessary but insufficient. General queries often underperform; distractors can degrade RAG output quality.
- Query-time augmentation and post-retrieval re-ranking substantially improve precision and recall.
- Chunking/token-awareness and metadata constraints materially affect retrieval quality.

## Recommendations (Prioritized)

1) Adaptive Thresholds + Query Classification (Short-term, high ROI)
- Problem: General queries like “health” or “virus” produce 0.45–0.55 scores—useful but filtered by 0.7 threshold.
- Recommendation: Implement adaptive thresholds by query type (specific vs general vs exploratory) as a light-weight first step. Expose CLI flags: --broad/--strict/--threshold.
- Why now: Minimal code, immediate UX improvement, aligns with article’s insight on query intent.

2) Multi-Query Expansion (Medium-term)
- Technique: Use LLM to generate 3–5 paraphrases/related questions; issue parallel vector searches; merge and deduplicate results.
- Ranking: Use similarity for initial cut; optionally cap per-query contribution (e.g., top-2 each) to avoid over-domination.
- Benefit: Expands coverage for broad/under-specified queries; reduces cold-start misses.
- Integration: Add a flag (e.g., --expand N) and pipeline path that aggregates results across sub-queries.

3) Hypothetical Answer Expansion (HyDE) (Medium-term)
- Technique: Generate a brief “imagined” answer for the user query; append to query before embedding.
- Benefit: Moves the query vector nearer to answer-like regions; notably boosts factual lookup in news summaries.
- Guardrails: Limit length; cache hyde text per query to control costs; disable for very short queries.

4) Cross-Encoder Re-ranking (Medium/Long-term)
- Technique: After initial retrieval (k≈20), score (query, document) pairs with a cross-encoder (e.g., bge-reranker, Cohere, or OpenAI re-rank API when available).
- Benefit: Reorders long tail, suppresses distractors; improves top-3 precision.
- Considerations: Extra latency and cost; make it opt-in (--rerank) and cache results.

5) Token- and Field-Aware Chunking (Medium-term)
- Current: We embed a single concatenation: title | summary | topics | body_excerpt.
- Recommendation: Introduce small chunks (e.g., 300–500 tokens) with overlaps for long bodies; keep title/summary in every chunk’s metadata; store a “parent” article_id for regrouping in UI.
- Benefit: Better recall for details; limits truncation; works well with re-ranking.

6) Metadata-Driven Filtering (Short/Mid-term)
- Technique: Use Chroma where filters (date ranges, source domains, topic tags) to constrain search space before embedding similarity.
- Benefit: Reduces distractors; improves precision and speed for scoped queries (e.g., last 7 days).
- Implementation: Add optional filters in CLI/UI (e.g., --since YYYY-MM-DD, --domain bbc.com, --topic covid-19).

7) Result Diversification (MMR) (Medium-term)
- Technique: Maximal Marginal Relevance on top-k results to balance relevance and diversity.
- Benefit: Avoids near-duplicate top results; improves coverage across subtopics.
- Parameter: λ in [0.2, 0.7] to control diversity vs relevance.

8) Diagnostics and Quality Monitoring (Short-term)
- Technique: Log score distributions per query type; track recall@k and precision@k for a small labeled set; optionally visualize with UMAP offline.
- Benefit: Evidence-based threshold and pipeline tuning; faster debugging of distractors.

9) Query Adapters (Long-term, experimental)
- Technique: Learn a linear adapter matrix from labeled (query, doc, label) pairs to warp query embeddings.
- Benefit: Application-specific retrieval improvements without retraining embedding model.
- Caveat: Requires dataset; start with synthetic labels; revisit once real user feedback accumulates.

## Tailored Improvement Plan for This Repo

A. Short-term (this sprint)
- Implement adaptive thresholds (config + CLI flags) and default to adaptive mode.
- Add metadata filters: date range (extracted_at), domain(host), and topic tag filter.
- Add export stability tests; ensure export skips embedding calls (done).

B. Medium-term (next sprint)
- Add multi-query expansion (--expand 3) and HyDE (--hyde) behind flags.
- Add simple MMR re-ranking on the initial top-k (no external model needed).
- Prototype cross-encoder re-ranking with a lightweight model and cache.

C. Long-term
- Split articles into token-aware chunks; add parent-child linkage.
- Stand up evaluation harness: curated queries, gold labels, metrics dashboard.
- Explore adapter training once we have usage data.

## Cost-Benefit Outline
- Adaptive thresholds, filters, MMR: Low cost, immediate value.
- Multi-query and HyDE: Moderate cost (LLM calls), high benefit on broad queries; cache aggressively.
- Cross-encoder re-ranking: Medium-high cost; large precision gains for top-3.
- Query adapters: High complexity; defer until we have feedback data.

## Risks and Mitigations
- Latency: Use flags, caching, and k-stage retrieval (k1 large → rerank → k2 small).
- Cost: Limit tokens, cache expansions, batch operations.
- Complexity creep: Keep base path simple; add advanced features as optional layers.

## Implementation Considerations
- Keep a simple, robust baseline path (current vector search) for reliability.
- Introduce advanced modes incrementally via CLI/UI flags and config toggles.
- Centralize query processing pipeline (expansion → retrieval → rerank → filter → format).
- Document fallbacks and timeouts to maintain responsiveness.

## Next Steps
1. Add adaptive threshold + CLI flags.
2. Add where-filters for date/domain/topics.
3. Implement MMR diversification.
4. Prototype multi-query expansion (flag-gated) and cache.
5. Add basic metrics logging and a small evaluation set in tests/.

"}
