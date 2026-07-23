# Derived Embedding Analysis Contracts

This package defines storage contracts only. It does not compute PCA, run
k-means, call OpenRouter, select a corpus, or decide whether work should be
retried.

`pca_analyses` identifies one PCA result by a sorted unique embedding-id set
and sanitized `pca_config`. Its lifecycle is `pending`, `complete`, or
`failed`; completed coordinates align exactly with the stored embedding ids.

`kmeans_analyses` identifies one clustering result by the same kind of
embedding set, a sanitized `clustering_config`, and an explicit feature source:
raw embedding vectors or one completed PCA entity. PCA-derived clustering uses
the PCA entity's exact embedding set; component selection belongs in
`clustering_config`.

Cluster summaries are nested inside the clustering entity and keyed by a
sanitized summary-configuration hash. Each cluster has independent state and
attempt history so a future analysis script can skip completed summaries and
retry only missing or failed summaries. The stored response metadata preserves
returned usage and cost (explicitly `null` when not reported) but excludes
credentials, full prompts, duplicated reasoning text, raw responses, and model
reasoning traces.

MongoDB persistence lives in `db_ops.pca_analyses` and
`db_ops.kmeans_analyses`. Run/skip/retry policy and analytical choices belong
to the standalone mechanism-analysis work governed by HD-0002.
