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
retry only missing or failed summaries. Schema version 2 stores every exact
prompt and its SHA-256 digest over the exact UTF-8 bytes. Only a completed
provider result with the same summary-configuration hash and byte-identical
verified prompt may be reused; reuse adds no provider attempt or duplicated
usage. Failed and successful attempts retain their individual usage and cost
(explicitly `null` when not reported).

Version-1 records remain readable and upgrade to version 2 before mutation.
Because their exact prompts were not retained, migrated prompt fields are
`null` and `exact_prompt_verified` is false, so they cannot satisfy exact
reuse. Credentials, request headers, raw provider responses, and model
reasoning traces remain excluded from every version.

Local-JSON persistence lives in `db_ops.pca_analyses` and
`db_ops.kmeans_analyses`. Run/skip/retry policy and analytical choices belong
to an explicit orchestration layer, not these storage contracts.
