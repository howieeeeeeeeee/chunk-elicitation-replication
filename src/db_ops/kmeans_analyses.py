"""Local-JSON persistence and query helpers for k-means analysis entities."""

from copy import deepcopy

from derived_analysis.common import attempt_history_extends
from derived_analysis.kmeans import (
    IMMUTABLE_KMEANS_FIELDS,
    KMEANS_SCHEMA_VERSION,
    LEGACY_KMEANS_SCHEMA_VERSION,
    add_summary_run,
    record_cluster_summary_reused,
    record_cluster_summary_completed,
    record_cluster_summary_failed,
    record_cluster_summary_started,
    record_clustering_completed,
    record_clustering_failed,
    record_clustering_started,
    rendered_prompt_hash,
    upgrade_kmeans_analysis,
    validate_kmeans_analysis,
)
from db_ops.pca_analyses import find_pca_analysis


KMEANS_ANALYSES_COLLECTION = "kmeans_analyses"


def setup_kmeans_analysis_indexes(db) -> list[str]:
    return []


def _summary_by_hash(entity: dict) -> dict:
    return {
        summary["summary_config_hash"]: summary for summary in entity["summaries"]
    }


def _assert_replacement_allowed(existing: dict, replacement: dict) -> None:
    if (
        existing.get("schema_version") == LEGACY_KMEANS_SCHEMA_VERSION
        and replacement.get("schema_version") == KMEANS_SCHEMA_VERSION
    ):
        if replacement != upgrade_kmeans_analysis(existing):
            raise ValueError(
                "Legacy K-means entities must be upgraded before other changes."
            )
        return
    changed = [
        field
        for field in IMMUTABLE_KMEANS_FIELDS
        if existing.get(field) != replacement.get(field)
    ]
    if changed:
        raise ValueError("K-means identity fields are immutable.")
    if not attempt_history_extends(
        existing["clustering"]["attempts"],
        replacement["clustering"]["attempts"],
    ):
        raise ValueError("Clustering attempt history is append-only.")
    if (
        existing["clustering"]["status"] == "complete"
        and existing["clustering"] != replacement["clustering"]
    ):
        raise ValueError("A complete clustering stage is immutable.")

    replacement_summaries = _summary_by_hash(replacement)
    for summary_hash, existing_summary in _summary_by_hash(existing).items():
        replacement_summary = replacement_summaries.get(summary_hash)
        if replacement_summary is None:
            raise ValueError("Stored summary runs cannot be removed.")
        for field in ("summary_config", "summary_config_hash", "created_at"):
            if existing_summary[field] != replacement_summary[field]:
                raise ValueError("Summary-run provenance is immutable.")
        replacement_clusters = {
            cluster["cluster_id"]: cluster
            for cluster in replacement_summary["clusters"]
        }
        for cluster in existing_summary["clusters"]:
            replacement_cluster = replacement_clusters.get(cluster["cluster_id"])
            if replacement_cluster is None:
                raise ValueError("Stored summary clusters cannot be removed.")
            if not attempt_history_extends(
                cluster["attempts"], replacement_cluster["attempts"]
            ):
                raise ValueError("Cluster summary attempt history is append-only.")
            if cluster["status"] == "complete" and (
                replacement_cluster != cluster
            ):
                raise ValueError("A complete cluster summary is immutable.")


def upsert_kmeans_analysis(db, entity: dict) -> dict:
    validate_kmeans_analysis(entity)
    collection = db[KMEANS_ANALYSES_COLLECTION]
    existing = collection.find_one({"_id": entity["_id"]})
    if existing is not None:
        _assert_replacement_allowed(existing, entity)
    changed = existing != entity
    result = collection.bulk_upsert([deepcopy(entity)])
    return {
        "kmeans_analysis_id": entity["_id"],
        "matched_count": result["matched_count"],
        "modified_count": int(existing is not None and changed),
        "upserted_id": entity["_id"] if result["upserted_count"] else None,
    }


def update_kmeans_analyses(db, entities) -> dict:
    results = [upsert_kmeans_analysis(db, entity) for entity in entities]
    return {
        "entity_count": len(results),
        "matched_count": sum(item["matched_count"] for item in results),
        "modified_count": sum(item["modified_count"] for item in results),
        "upserted_count": sum(item["upserted_id"] is not None for item in results),
        "results": results,
    }


def find_kmeans_analysis(db, analysis_id: str):
    return db[KMEANS_ANALYSES_COLLECTION].find_one({"_id": analysis_id})


def find_completed_clustering(db, analysis_id: str):
    return db[KMEANS_ANALYSES_COLLECTION].find_one(
        {"_id": analysis_id, "clustering.status": "complete"}
    )


def find_completed_cluster_summary(
    db, analysis_id: str, summary_config_hash: str, cluster_id: int
):
    entity = find_completed_clustering(db, analysis_id)
    if entity is None:
        return None
    for summary_run in entity["summaries"]:
        if summary_run["summary_config_hash"] != summary_config_hash:
            continue
        for cluster in summary_run["clusters"]:
            if cluster["cluster_id"] == cluster_id and cluster["status"] == "complete":
                return deepcopy(cluster)
    return None


def find_completed_exact_cluster_summary(
    db,
    summary_config_hash: str,
    prompt: str,
    *,
    exclude: tuple[str, int] | None = None,
):
    prompt_digest = rendered_prompt_hash(prompt)
    candidates = []
    # LocalJsonCollection does not implement MongoDB's nested-array matching.
    # Iterate the small derived collection and enforce exact byte equality here.
    for entity in db[KMEANS_ANALYSES_COLLECTION].find({}):
        if entity.get("schema_version") != KMEANS_SCHEMA_VERSION:
            continue
        validate_kmeans_analysis(entity)
        for summary_run in entity.get("summaries", []):
            if summary_run.get("summary_config_hash") != summary_config_hash:
                continue
            for cluster in summary_run.get("clusters", []):
                identity = (entity["_id"], cluster.get("cluster_id"))
                if exclude == identity:
                    continue
                if (
                    cluster.get("status") == "complete"
                    and cluster.get("reuse") is None
                    and cluster.get("exact_prompt_verified") is True
                    and cluster.get("prompt_hash") == prompt_digest
                    and cluster.get("prompt") == prompt
                    and isinstance(cluster.get("output"), dict)
                ):
                    candidates.append(
                        {
                            "analysis_id": entity["_id"],
                            "summary_config_hash": summary_config_hash,
                            "cluster_id": cluster["cluster_id"],
                            "cluster": deepcopy(cluster),
                        }
                    )
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            item["analysis_id"],
            item["cluster_id"],
        )
    )
    return candidates[0]


def query_kmeans_analyses(
    db,
    *,
    embedding_set_hash: str | None = None,
    clustering_config_hash: str | None = None,
    feature_kind: str | None = None,
    clustering_status: str | None = None,
):
    query = {}
    if embedding_set_hash is not None:
        query["embedding_set_hash"] = embedding_set_hash
    if clustering_config_hash is not None:
        query["clustering_config_hash"] = clustering_config_hash
    if feature_kind is not None:
        if feature_kind not in {"embeddings", "pca"}:
            raise ValueError("K-means feature-kind query is invalid.")
        query["feature_source.kind"] = feature_kind
    if clustering_status is not None:
        if clustering_status not in {"pending", "complete", "failed"}:
            raise ValueError("K-means clustering-status query is invalid.")
        query["clustering.status"] = clustering_status
    return db[KMEANS_ANALYSES_COLLECTION].find(query)


def _load(db, analysis_id: str) -> dict:
    entity = find_kmeans_analysis(db, analysis_id)
    if entity is None:
        raise ValueError("K-means analysis does not exist.")
    if entity.get("schema_version") == LEGACY_KMEANS_SCHEMA_VERSION:
        upgraded = upgrade_kmeans_analysis(entity)
        upsert_kmeans_analysis(db, upgraded)
        return upgraded
    return entity


def start_clustering_attempt(db, analysis_id: str, *, timestamp=None) -> dict:
    updated = record_clustering_started(_load(db, analysis_id), timestamp=timestamp)
    upsert_kmeans_analysis(db, updated)
    return updated


def complete_clustering_attempt(
    db, analysis_id: str, output: dict, *, timestamp=None
) -> dict:
    entity = _load(db, analysis_id)
    pca_analysis = None
    if entity["feature_source"]["kind"] == "pca":
        pca_analysis = find_pca_analysis(
            db, entity["feature_source"]["pca_analysis_id"]
        )
    updated = record_clustering_completed(
        entity,
        output,
        pca_analysis=pca_analysis,
        timestamp=timestamp,
    )
    upsert_kmeans_analysis(db, updated)
    return updated


def fail_clustering_attempt(
    db, analysis_id: str, error: Exception, *, timestamp=None
) -> dict:
    updated = record_clustering_failed(
        _load(db, analysis_id), error, timestamp=timestamp
    )
    upsert_kmeans_analysis(db, updated)
    return updated


def register_summary_run(
    db, analysis_id: str, summary_config: dict, *, timestamp=None
) -> dict:
    updated = add_summary_run(
        _load(db, analysis_id), summary_config, timestamp=timestamp
    )
    upsert_kmeans_analysis(db, updated)
    return updated


def start_cluster_summary_attempt(
    db,
    analysis_id: str,
    summary_config_hash: str,
    cluster_id: int,
    *,
    input_hash: str,
    prompt: str,
    timestamp=None,
) -> dict:
    updated = record_cluster_summary_started(
        _load(db, analysis_id),
        summary_config_hash,
        cluster_id,
        input_hash=input_hash,
        prompt=prompt,
        timestamp=timestamp,
    )
    upsert_kmeans_analysis(db, updated)
    return updated


def reuse_completed_cluster_summary(
    db,
    analysis_id: str,
    summary_config_hash: str,
    cluster_id: int,
    *,
    input_hash: str,
    prompt: str,
    timestamp=None,
):
    source = find_completed_exact_cluster_summary(
        db,
        summary_config_hash,
        prompt,
        exclude=(analysis_id, cluster_id),
    )
    if source is None:
        return None
    updated = record_cluster_summary_reused(
        _load(db, analysis_id),
        summary_config_hash,
        cluster_id,
        input_hash=input_hash,
        prompt=prompt,
        source_analysis_id=source["analysis_id"],
        source_summary_config_hash=source["summary_config_hash"],
        source_cluster_id=source["cluster_id"],
        source_cluster=source["cluster"],
        timestamp=timestamp,
    )
    upsert_kmeans_analysis(db, updated)
    return updated


def complete_cluster_summary_attempt(
    db,
    analysis_id: str,
    summary_config_hash: str,
    cluster_id: int,
    response: dict,
    *,
    timestamp=None,
) -> dict:
    updated = record_cluster_summary_completed(
        _load(db, analysis_id),
        summary_config_hash,
        cluster_id,
        response,
        timestamp=timestamp,
    )
    upsert_kmeans_analysis(db, updated)
    return updated


def fail_cluster_summary_attempt(
    db,
    analysis_id: str,
    summary_config_hash: str,
    cluster_id: int,
    error: Exception,
    *,
    timestamp=None,
) -> dict:
    updated = record_cluster_summary_failed(
        _load(db, analysis_id),
        summary_config_hash,
        cluster_id,
        error,
        timestamp=timestamp,
    )
    upsert_kmeans_analysis(db, updated)
    return updated
