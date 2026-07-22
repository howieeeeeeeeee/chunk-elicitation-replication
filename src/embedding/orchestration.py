"""Sequential, resumable reasoning-embedding orchestration and dry-run plans."""

import logging

from db_ops.embeddings import (
    find_embedding,
    find_successful_embedding,
    upsert_embedding,
)
from embedding.client import EmbeddingRequestError, request_reasoning_embedding
from embedding.config import embedding_entity_id, normalize_embedding_config
from embedding.entities import (
    build_embedding_entity,
    record_embedding_failure,
    record_embedding_success,
)


logger = logging.getLogger(__name__)


class EmbeddingSourceError(EmbeddingRequestError):
    """A structured failure caused by a missing or malformed source value."""


class EmbeddingEligibilityError(ValueError):
    """Raised when source records are not eligible for embedding."""


def _source_error(code: str, message: str) -> EmbeddingSourceError:
    return EmbeddingSourceError(code, message)


def _require_ready_simulation(simulation: dict, simulation_id: str) -> None:
    if simulation.get("completed") is not True:
        raise EmbeddingEligibilityError(
            f"Simulation id={simulation_id} is not completed."
        )
    if simulation.get("archived") is not False:
        raise EmbeddingEligibilityError(f"Simulation id={simulation_id} is archived.")


def _require_successful_session(session: dict, simulation_session_id: str) -> None:
    if (
        session.get("agent_response_success") is not True
        or session.get("schema_check_pass") is not True
    ):
        raise EmbeddingEligibilityError(
            f"Simulation session id={simulation_session_id} is not successful."
        )


def _result(status: str, embedding_id: str, *, attempted: bool, reason: str) -> dict:
    return {
        "status": status,
        "embedding_id": embedding_id,
        "attempted": attempted,
        "reason": reason,
    }


def _base_entity(
    existing,
    simulation_id: str,
    simulation_session_id: str,
    decision_index: int,
    input_text: str | None,
    embedding_config: dict,
):
    if existing is not None:
        if existing.get("input_text") != input_text:
            raise ValueError("Embedding input is immutable for an existing identity.")
        return existing
    return build_embedding_entity(
        simulation_id,
        simulation_session_id,
        decision_index,
        input_text,
        embedding_config,
    )


def _save_source_failure(
    db,
    existing,
    simulation_id: str,
    simulation_session_id: str,
    decision_index: int,
    input_text: str | None,
    embedding_config: dict,
    error: EmbeddingSourceError,
    embedding_id: str,
    attempted_ids: set[str],
    active_logger: logging.Logger,
) -> dict:
    try:
        entity = _base_entity(
            existing,
            simulation_id,
            simulation_session_id,
            decision_index,
            input_text,
            embedding_config,
        )
        attempted_ids.add(embedding_id)
        upsert_embedding(db, record_embedding_failure(entity, error))
    except Exception:
        active_logger.warning(
            "Unable to persist source failure for embedding_id=%s", embedding_id
        )
        return _result(
            "failed", embedding_id, attempted=False, reason="persistence_failure"
        )
    active_logger.warning(
        "Saved source failure for embedding_id=%s code=%s", embedding_id, error.code
    )
    return _result("failed", embedding_id, attempted=True, reason=error.code)


def embed_decision_reasoning(
    db,
    simulation_session_id: str,
    decision_index: int,
    embedding_config: dict,
    *,
    request_fn=request_reasoning_embedding,
    attempted_ids: set[str] | None = None,
    active_logger: logging.Logger | None = None,
) -> dict:
    """Embed one decision at most once in this invocation and persist its attempt."""
    active_logger = active_logger or logger
    attempted_ids = attempted_ids if attempted_ids is not None else set()
    normalized_config = normalize_embedding_config(embedding_config)
    embedding_id = embedding_entity_id(
        simulation_session_id, decision_index, normalized_config
    )

    if embedding_id in attempted_ids:
        active_logger.info(
            "Skipping already-attempted embedding_id=%s in this invocation",
            embedding_id,
        )
        return _result(
            "skipped", embedding_id, attempted=False, reason="already_attempted"
        )
    if find_successful_embedding(db, embedding_id) is not None:
        active_logger.info("Skipping successful embedding_id=%s", embedding_id)
        return _result(
            "skipped", embedding_id, attempted=False, reason="existing_success"
        )

    existing = find_embedding(db, embedding_id)
    session = db["simulation_sessions"].find_one({"_id": simulation_session_id})
    if session is None:
        active_logger.warning(
            "Missing simulation session for embedding_id=%s", embedding_id
        )
        return _result(
            "missing", embedding_id, attempted=False, reason="missing_session"
        )
    _require_successful_session(session, simulation_session_id)
    simulation_id = session.get("simulation_id")
    if not isinstance(simulation_id, str) or not simulation_id:
        active_logger.warning("Missing simulation id for embedding_id=%s", embedding_id)
        return _result(
            "malformed", embedding_id, attempted=False, reason="missing_simulation_id"
        )

    decisions = session.get("decisions")
    if not isinstance(decisions, list) or decision_index >= len(decisions):
        return _save_source_failure(
            db,
            existing,
            simulation_id,
            simulation_session_id,
            decision_index,
            None,
            normalized_config,
            _source_error(
                "missing_decision", "Simulation session decision is unavailable."
            ),
            embedding_id,
            attempted_ids,
            active_logger,
        )

    decision = decisions[decision_index]
    input_text = (
        decision[1]
        if isinstance(decision, (list, tuple))
        and len(decision) > 1
        and isinstance(decision[1], str)
        and decision[1].strip()
        else None
    )
    if not isinstance(decision, (list, tuple)) or len(decision) != 2 or input_text is None:
        return _save_source_failure(
            db,
            existing,
            simulation_id,
            simulation_session_id,
            decision_index,
            input_text,
            normalized_config,
            _source_error(
                "malformed_reasoning",
                "Simulation decision does not contain a valid reasoning string.",
            ),
            embedding_id,
            attempted_ids,
            active_logger,
        )

    try:
        entity = _base_entity(
            existing,
            simulation_id,
            simulation_session_id,
            decision_index,
            input_text,
            normalized_config,
        )
    except ValueError:
        active_logger.warning("Immutable input mismatch for embedding_id=%s", embedding_id)
        return _result(
            "failed", embedding_id, attempted=False, reason="immutable_input_mismatch"
        )

    attempted_ids.add(embedding_id)
    active_logger.info("Starting embedding_id=%s", embedding_id)
    try:
        response = request_fn(input_text, normalized_config)
        saved = record_embedding_success(entity, response)
        upsert_embedding(db, saved)
    except Exception as error:
        try:
            saved = record_embedding_failure(entity, error)
            upsert_embedding(db, saved)
        except Exception:
            active_logger.warning(
                "Unable to persist failed attempt for embedding_id=%s", embedding_id
            )
            return _result(
                "failed", embedding_id, attempted=True, reason="persistence_failure"
            )
        active_logger.warning("Embedding attempt failed for embedding_id=%s", embedding_id)
        return _result("failed", embedding_id, attempted=True, reason="request_failure")

    active_logger.info("Embedding succeeded for embedding_id=%s", embedding_id)
    return _result("succeeded", embedding_id, attempted=True, reason="success")


def _empty_summary(scope: str, identifier: str) -> dict:
    return {
        "scope": scope,
        "id": identifier,
        "processed": 0,
        "succeeded": 0,
        "failed": 0,
        "skipped": 0,
        "missing": 0,
        "malformed": 0,
    }


def _count_result(summary: dict, result: dict) -> None:
    summary["processed"] += 1
    status = result["status"]
    if status in {"succeeded", "failed", "skipped", "missing", "malformed"}:
        summary[status] += 1


def embed_simulation_session(
    db,
    simulation_session_id: str,
    embedding_config: dict,
    *,
    request_fn=request_reasoning_embedding,
    attempted_ids: set[str] | None = None,
    active_logger: logging.Logger | None = None,
) -> dict:
    """Process every decision in a simulation session sequentially."""
    active_logger = active_logger or logger
    attempted_ids = attempted_ids if attempted_ids is not None else set()
    summary = _empty_summary("simulation_session", simulation_session_id)
    session = db["simulation_sessions"].find_one({"_id": simulation_session_id})
    if session is None:
        summary["missing"] = 1
        active_logger.warning("Missing simulation session id=%s", simulation_session_id)
        return summary
    _require_successful_session(session, simulation_session_id)
    decisions = session.get("decisions")
    if not isinstance(decisions, list):
        summary["malformed"] = 1
        active_logger.warning("Malformed decisions for session id=%s", simulation_session_id)
        return summary

    active_logger.info(
        "Embedding session id=%s decisions=%d", simulation_session_id, len(decisions)
    )
    for decision_index in range(len(decisions)):
        try:
            result = embed_decision_reasoning(
                db,
                simulation_session_id,
                decision_index,
                embedding_config,
                request_fn=request_fn,
                attempted_ids=attempted_ids,
                active_logger=active_logger,
            )
        except Exception:
            active_logger.warning(
                "Unexpected decision orchestration failure session=%s index=%d",
                simulation_session_id,
                decision_index,
            )
            result = {
                "status": "failed",
                "embedding_id": None,
                "attempted": False,
                "reason": "orchestration_failure",
            }
        _count_result(summary, result)
    active_logger.info(
        "Embedding session summary id=%s processed=%d succeeded=%d failed=%d skipped=%d",
        simulation_session_id,
        summary["processed"],
        summary["succeeded"],
        summary["failed"],
        summary["skipped"],
    )
    return summary


def embed_simulation(
    db,
    simulation_id: str,
    embedding_config: dict,
    *,
    request_fn=request_reasoning_embedding,
    attempted_ids: set[str] | None = None,
    active_logger: logging.Logger | None = None,
) -> dict:
    """Gate a simulation and process only its referenced sessions sequentially."""
    active_logger = active_logger or logger
    attempted_ids = attempted_ids if attempted_ids is not None else set()
    summary = _empty_summary("simulation", simulation_id)
    summary.update({"sessions": 0, "eligible": False, "session_summaries": []})
    simulation = db["simulations"].find_one({"_id": simulation_id})
    if simulation is None:
        summary["missing"] = 1
        active_logger.warning("Missing simulation id=%s", simulation_id)
        return summary
    _require_ready_simulation(simulation, simulation_id)
    if simulation.get("instruction_config", {}).get("explain_reasoning") is not True:
        summary["skipped"] = 1
        active_logger.warning("Reasoning is not enabled for simulation id=%s", simulation_id)
        return summary

    session_ids = simulation.get("simulation_sessions")
    if not isinstance(session_ids, list):
        summary["malformed"] = 1
        active_logger.warning("Malformed session ids for simulation id=%s", simulation_id)
        return summary
    summary["eligible"] = True
    summary["sessions"] = len(session_ids)
    active_logger.info(
        "Embedding simulation id=%s sessions=%d", simulation_id, len(session_ids)
    )
    for session_id in session_ids:
        if not isinstance(session_id, str) or not session_id:
            summary["malformed"] += 1
            continue
        session_summary = embed_simulation_session(
            db,
            session_id,
            embedding_config,
            request_fn=request_fn,
            attempted_ids=attempted_ids,
            active_logger=active_logger,
        )
        summary["session_summaries"].append(session_summary)
        for field in ("processed", "succeeded", "failed", "skipped", "missing", "malformed"):
            summary[field] += session_summary[field]
    active_logger.info(
        "Embedding simulation summary id=%s processed=%d succeeded=%d failed=%d skipped=%d",
        simulation_id,
        summary["processed"],
        summary["succeeded"],
        summary["failed"],
        summary["skipped"],
    )
    return summary


def summarize_embedding_plan(
    db,
    simulation_ids: list[str],
    embedding_config: dict,
    *,
    active_logger: logging.Logger | None = None,
) -> dict:
    """Read eligibility and existing-state counts without calls or writes."""
    active_logger = active_logger or logger
    normalized_config = normalize_embedding_config(embedding_config)
    summary = {
        "selected_simulations": len(simulation_ids),
        "eligible_simulations": 0,
        "ineligible_simulations": 0,
        "missing_simulations": 0,
        "sessions": 0,
        "missing_sessions": 0,
        "decisions": 0,
        "new_embeddings": 0,
        "existing_successes": 0,
        "existing_failures": 0,
        "malformed": 0,
        "planned_attempts": 0,
    }
    for simulation_id in simulation_ids:
        simulation = db["simulations"].find_one({"_id": simulation_id})
        if simulation is None:
            summary["missing_simulations"] += 1
            continue
        _require_ready_simulation(simulation, simulation_id)
        if simulation.get("instruction_config", {}).get("explain_reasoning") is not True:
            summary["ineligible_simulations"] += 1
            continue
        session_ids = simulation.get("simulation_sessions")
        if not isinstance(session_ids, list):
            summary["malformed"] += 1
            continue
        summary["eligible_simulations"] += 1
        summary["sessions"] += len(session_ids)
        for session_id in session_ids:
            if not isinstance(session_id, str) or not session_id:
                summary["malformed"] += 1
                continue
            session = db["simulation_sessions"].find_one({"_id": session_id})
            if session is None:
                summary["missing_sessions"] += 1
                continue
            _require_successful_session(session, session_id)
            decisions = session.get("decisions")
            if not isinstance(decisions, list):
                summary["malformed"] += 1
                continue
            summary["decisions"] += len(decisions)
            for decision_index, decision in enumerate(decisions):
                embedding_id = embedding_entity_id(
                    session_id, decision_index, normalized_config
                )
                existing = find_embedding(db, embedding_id)
                if existing is not None and existing.get("success") is True:
                    summary["existing_successes"] += 1
                elif existing is not None:
                    summary["existing_failures"] += 1
                    summary["planned_attempts"] += 1
                elif (
                    isinstance(decision, (list, tuple))
                    and len(decision) == 2
                    and isinstance(decision[1], str)
                    and decision[1].strip()
                ):
                    summary["new_embeddings"] += 1
                    summary["planned_attempts"] += 1
                else:
                    summary["malformed"] += 1
                    summary["planned_attempts"] += 1
    active_logger.info(
        "Embedding plan selected=%d eligible=%d sessions=%d decisions=%d planned_attempts=%d",
        summary["selected_simulations"],
        summary["eligible_simulations"],
        summary["sessions"],
        summary["decisions"],
        summary["planned_attempts"],
    )
    return summary
