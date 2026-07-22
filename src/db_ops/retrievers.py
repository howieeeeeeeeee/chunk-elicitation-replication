from db_ops.embeddings import (
    find_embedding,
    find_successful_embedding,
    query_embeddings,
)


def get_embedding(_db, embedding_id):
    return find_embedding(_db, embedding_id)


def get_successful_embedding(_db, embedding_id):
    return find_successful_embedding(_db, embedding_id)


def get_embeddings_by_simulation(
    _db, simulation_id, embedding_config_hash=None, success=None
):
    return query_embeddings(
        _db,
        simulation_id=simulation_id,
        embedding_config_hash=embedding_config_hash,
        success=success,
    )


def get_embeddings_by_session(
    _db, simulation_session_id, embedding_config_hash=None, success=None
):
    return query_embeddings(
        _db,
        simulation_session_id=simulation_session_id,
        embedding_config_hash=embedding_config_hash,
        success=success,
    )


def get_embeddings_by_config(_db, embedding_config_hash, success=None):
    return query_embeddings(
        _db,
        embedding_config_hash=embedding_config_hash,
        success=success,
    )


def get_simulation(_db, simulation_id):
    return _db.simulations.find_one({"_id": simulation_id})


def get_simulation_results(_db, simulation_id):
    simulation = _db.simulations.find_one({"_id": simulation_id})
    decisions = []
    if not simulation:
        return decisions
    for session in _db.simulation_sessions.find(
        {"_id": {"$in": simulation.get("simulation_sessions", [])}}
    ):
        decisions.extend(session.get("decisions", []))
    return decisions


def get_benchmark_results(_db, game_type):
    benchmarks = _db.benchmarks.find({"game_type": game_type})
    decisions = []
    for benchmark in benchmarks:
        decisions.extend(benchmark.get("decisions", []))
    return decisions


def get_findings(_db, finding_id):
    return _db.findings.find_one({"_id": finding_id})


def get_all_simulation_results(_db, simulation_ids):
    simulation_ids = list(simulation_ids) if isinstance(simulation_ids, tuple) else simulation_ids
    simulations = list(
        _db.simulations.find(
            {"_id": {"$in": simulation_ids}}, {"_id": 1, "simulation_sessions": 1}
        )
    )
    sim_to_sessions = {
        sim["_id"]: sim.get("simulation_sessions", []) for sim in simulations
    }
    all_session_ids = [
        session_id for sessions in sim_to_sessions.values() for session_id in sessions
    ]
    sessions = list(
        _db.simulation_sessions.find(
            {"_id": {"$in": all_session_ids}}, {"_id": 1, "decisions": 1}
        )
    )
    session_to_decisions = {
        session["_id"]: session.get("decisions", []) for session in sessions
    }
    results = {}
    for sim_id, session_ids in sim_to_sessions.items():
        decisions = []
        for session_id in session_ids:
            decisions.extend(session_to_decisions.get(session_id, []))
        results[sim_id] = decisions
    return results
