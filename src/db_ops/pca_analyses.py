"""Local-JSON persistence and query helpers for PCA analysis entities."""

from copy import deepcopy

from derived_analysis.common import attempt_history_extends
from derived_analysis.pca import (
    IMMUTABLE_PCA_FIELDS,
    record_pca_completed,
    record_pca_failed,
    record_pca_started,
    validate_pca_analysis,
)


PCA_ANALYSES_COLLECTION = "pca_analyses"


def setup_pca_analysis_indexes(db) -> list[str]:
    return []


def _assert_replacement_allowed(existing: dict, replacement: dict) -> None:
    changed = [
        field
        for field in IMMUTABLE_PCA_FIELDS
        if existing.get(field) != replacement.get(field)
    ]
    if changed:
        raise ValueError("PCA identity fields are immutable.")
    if not attempt_history_extends(existing["attempts"], replacement["attempts"]):
        raise ValueError("PCA attempt history is append-only.")
    if existing["status"] == "complete" and existing != replacement:
        raise ValueError("A complete PCA analysis is immutable.")


def upsert_pca_analysis(db, entity: dict) -> dict:
    validate_pca_analysis(entity)
    collection = db[PCA_ANALYSES_COLLECTION]
    existing = collection.find_one({"_id": entity["_id"]})
    if existing is not None:
        _assert_replacement_allowed(existing, entity)
    changed = existing != entity
    result = collection.bulk_upsert([deepcopy(entity)])
    return {
        "pca_analysis_id": entity["_id"],
        "matched_count": result["matched_count"],
        "modified_count": int(existing is not None and changed),
        "upserted_id": entity["_id"] if result["upserted_count"] else None,
    }


def update_pca_analyses(db, entities) -> dict:
    results = [upsert_pca_analysis(db, entity) for entity in entities]
    return {
        "entity_count": len(results),
        "matched_count": sum(item["matched_count"] for item in results),
        "modified_count": sum(item["modified_count"] for item in results),
        "upserted_count": sum(item["upserted_id"] is not None for item in results),
        "results": results,
    }


def find_pca_analysis(db, analysis_id: str):
    return db[PCA_ANALYSES_COLLECTION].find_one({"_id": analysis_id})


def find_completed_pca_analysis(db, analysis_id: str):
    return db[PCA_ANALYSES_COLLECTION].find_one(
        {"_id": analysis_id, "status": "complete"}
    )


def query_pca_analyses(
    db,
    *,
    embedding_set_hash: str | None = None,
    pca_config_hash: str | None = None,
    status: str | None = None,
):
    query = {}
    if embedding_set_hash is not None:
        query["embedding_set_hash"] = embedding_set_hash
    if pca_config_hash is not None:
        query["pca_config_hash"] = pca_config_hash
    if status is not None:
        if status not in {"pending", "complete", "failed"}:
            raise ValueError("PCA status query is invalid.")
        query["status"] = status
    return db[PCA_ANALYSES_COLLECTION].find(query)


def start_pca_attempt(db, analysis_id: str, *, timestamp=None) -> dict:
    entity = find_pca_analysis(db, analysis_id)
    if entity is None:
        raise ValueError("PCA analysis does not exist.")
    updated = record_pca_started(entity, timestamp=timestamp)
    upsert_pca_analysis(db, updated)
    return updated


def complete_pca_attempt(
    db, analysis_id: str, output: dict, *, timestamp=None
) -> dict:
    entity = find_pca_analysis(db, analysis_id)
    if entity is None:
        raise ValueError("PCA analysis does not exist.")
    updated = record_pca_completed(entity, output, timestamp=timestamp)
    upsert_pca_analysis(db, updated)
    return updated


def fail_pca_attempt(
    db, analysis_id: str, error: Exception, *, timestamp=None
) -> dict:
    entity = find_pca_analysis(db, analysis_id)
    if entity is None:
        raise ValueError("PCA analysis does not exist.")
    updated = record_pca_failed(entity, error, timestamp=timestamp)
    upsert_pca_analysis(db, updated)
    return updated
