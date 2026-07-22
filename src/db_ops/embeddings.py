"""Local-JSON persistence and query primitives for reasoning embeddings."""

from copy import deepcopy


EMBEDDINGS_COLLECTION = "embeddings"


def setup_embedding_indexes(db) -> list[str]:
    return []


def _assert_immutable_fields(existing: dict, replacement: dict) -> None:
    from embedding.entities import IMMUTABLE_EMBEDDING_FIELDS

    changed = [
        field
        for field in IMMUTABLE_EMBEDDING_FIELDS
        if existing.get(field) != replacement.get(field)
    ]
    if changed:
        raise ValueError("Embedding identity or input fields are immutable.")


def upsert_embedding(db, entity: dict) -> dict:
    from embedding.entities import validate_embedding_entity

    validate_embedding_entity(entity)
    collection = db[EMBEDDINGS_COLLECTION]
    existing = collection.find_one({"_id": entity["_id"]})
    if existing is not None:
        _assert_immutable_fields(existing, entity)
    changed = existing != entity
    result = collection.bulk_upsert([deepcopy(entity)])
    return {
        "embedding_id": entity["_id"],
        "matched_count": result["matched_count"],
        "modified_count": int(existing is not None and changed),
        "upserted_id": entity["_id"] if result["upserted_count"] else None,
    }


def update_embeddings(db, entities) -> dict:
    summaries = [upsert_embedding(db, entity) for entity in entities]
    return {
        "entity_count": len(summaries),
        "matched_count": sum(item["matched_count"] for item in summaries),
        "modified_count": sum(item["modified_count"] for item in summaries),
        "upserted_count": sum(item["upserted_id"] is not None for item in summaries),
        "results": summaries,
    }


def find_embedding(db, embedding_id: str):
    return db[EMBEDDINGS_COLLECTION].find_one({"_id": embedding_id})


def find_successful_embedding(db, embedding_id: str):
    return db[EMBEDDINGS_COLLECTION].find_one(
        {"_id": embedding_id, "success": True}
    )


def query_embeddings(
    db,
    *,
    simulation_id: str | None = None,
    simulation_session_id: str | None = None,
    embedding_config_hash: str | None = None,
    success: bool | None = None,
):
    query = {}
    if simulation_id is not None:
        query["simulation_id"] = simulation_id
    if simulation_session_id is not None:
        query["simulation_session_id"] = simulation_session_id
    if embedding_config_hash is not None:
        query["embedding_config_hash"] = embedding_config_hash
    if success is not None:
        if not isinstance(success, bool):
            raise ValueError("success query must be boolean or None.")
        query["success"] = success
    return db[EMBEDDINGS_COLLECTION].find(query)
