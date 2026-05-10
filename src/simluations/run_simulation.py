from agents.econ_agent import EconomicAgent  # Updated import
from db_ops.config import DATA_ROOT, get_database
from db_ops.update import update_simulation_sessions, update_simulations
from games.schemas import GAME_DECISION_ARRAY_SCHEMA
from datetime import datetime
from pathlib import Path
from .process_simulation_data import run_schema_check
import uuid
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED


# Set up the logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set the logging level for this module

# Create a handler (if needed) and set its logging level
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# Create a formatter and set it for the handler
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

# Add the handler to the logger
if not logger.handlers:
    logger.addHandler(handler)

MAX_FAILED_SESSION_ALLOW_ITERATIVE = 50
MAX_FAILED_SESSION_ALLOW_BATCH = 20


def _normalize_extra_flag(extra_flag):
    if extra_flag is None:
        return []
    if isinstance(extra_flag, list):
        return list(extra_flag)
    if isinstance(extra_flag, tuple):
        return list(extra_flag)
    return [extra_flag]


def initialize_simulation(
    simulation_name,
    phase_name,
    simulation_config,
    instruction_config,
    llm_config,
    extraFlag=None,
):

    decision_schema = GAME_DECISION_ARRAY_SCHEMA[simulation_config["game_type"]].schema
    if "decision" in decision_schema:
        simulation_config["decision_length"] = len(decision_schema["decision"])
    else:
        simulation_config["decision_length"] = len(decision_schema)
    extra_flag = _normalize_extra_flag(extraFlag)
    return {
        "_id": str(uuid.uuid4()),
        "name": simulation_name,
        "simulation_config": simulation_config,
        "instruction_config": instruction_config,
        "llm_config": llm_config,
        "extraFlag": extra_flag,
        "simulation_sessions": [],
        "n_valid_simulation_results": 0,
        "failed_sessions": [],
        "completed": False,
        "phase_name": phase_name,
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
        "archived": False,
        "notes": [],
    }


def create_simulation_session(simulation_id):
    return {
        "_id": str(uuid.uuid4()),
        "simulation_id": simulation_id,
        "schema_check_pass": None,
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
    }


def handle_successful_response(simulation, simulation_session, result, decisions):
    simulation_session["decisions"] = decisions
    simulation_session["schema_check_pass"] = True
    simulation["n_valid_simulation_results"] += len(decisions)
    simulation["simulation_sessions"].append(simulation_session["_id"])
    logger.info(
        "Valid simulation result added. Total valid results: %d",
        simulation["n_valid_simulation_results"],
    )


def handle_failed_response(simulation, simulation_session, msg):
    simulation_session["schema_check_pass"] = False
    simulation_session["schema_error"] = msg
    simulation["failed_sessions"].append(simulation_session["_id"])
    logger.warning("Schema check failed: %s", msg)


def update_simulation_status(db, simulation, simulation_session):
    simulation["updated_at"] = datetime.now()
    simulation_session["updated_at"] = datetime.now()
    update_simulation_sessions(db=db, data=[simulation_session])
    update_simulations(db=db, data=[simulation])
    logger.info("Simulation session updated: %s", simulation_session["_id"])
    logger.info("Simulation updated: %s", simulation["_id"])


def check_termination_conditions(
    simulation,
    target_simulation_n,
    schema_check_failed_first_response,
    max_failed_sessions=MAX_FAILED_SESSION_ALLOW_ITERATIVE,
):
    if simulation["n_valid_simulation_results"] >= target_simulation_n:
        simulation["completed"] = True
        logger.info("Simulation completed!")
        simulation["notes"].append("Simulation completed!")
        simulation["complete_at"] = datetime.now()
        return True
    if len(simulation["failed_sessions"]) > max_failed_sessions:
        logger.warning("Simulation failed: Too many failed sessions.")
        simulation["notes"].append("Simulation failed due to too many failed sessions.")
        return True
    if schema_check_failed_first_response:
        logger.warning("Simulation failed: Wrong schema for the first responses")
        simulation["notes"].append(
            "Simulation failed due to wrong schema for the first responses"
        )
        return True
    return False


def _run_split_mode(
    agent,
    simulation,
    simulation_config,
    instruction_config,
    target_simulation_n,
    db,
):
    batch_simulation_n = simulation_config["batch_simulation_n"]
    split_n = instruction_config["split_n"]
    total_chunks = batch_simulation_n // split_n

    terminated_by_failure = False

    while (
        simulation["n_valid_simulation_results"] < target_simulation_n
        and not terminated_by_failure
    ):
        agent.reset_split_thread()

        for chunk_index in range(1, total_chunks + 1):
            simulation_session = create_simulation_session(simulation["_id"])
            success, result = agent.get_decisions_split_chunk()
            simulation_session.update(
                {
                    "agent_response_success": success,
                    "result": result,
                    "split_chunk_index": chunk_index,
                    "split_total_chunks": total_chunks,
                    "updated_at": datetime.now(),
                }
            )

            if not success:
                simulation["failed_sessions"].append(simulation_session["_id"])
                simulation["notes"].append(
                    f"Split mode: API failure in chunk {chunk_index}/{total_chunks}. Error: {result.get('msg')}"
                )
                logger.warning(
                    "Split mode API failure in chunk %d: %s",
                    chunk_index,
                    result.get("msg"),
                )
                update_simulation_status(db, simulation, simulation_session)
                if len(simulation["failed_sessions"]) > MAX_FAILED_SESSION_ALLOW_BATCH:
                    simulation["notes"].append(
                        "Split mode: too many failed sessions; simulation terminated."
                    )
                    terminated_by_failure = True
                break

            schema_check, msg = run_schema_check(
                result["result"], simulation_config, instruction_config
            )
            if not schema_check:
                handle_failed_response(simulation, simulation_session, msg)
                simulation["notes"].append(
                    f"Split mode: schema check failed in chunk {chunk_index}/{total_chunks}. Error: {msg}"
                )
                update_simulation_status(db, simulation, simulation_session)
                if len(simulation["failed_sessions"]) > MAX_FAILED_SESSION_ALLOW_BATCH:
                    simulation["notes"].append(
                        "Split mode: too many failed sessions; simulation terminated."
                    )
                    terminated_by_failure = True
                break

            decisions = result["result"]["all_responses"]
            handle_successful_response(
                simulation, simulation_session, result, decisions
            )
            update_simulation_status(db, simulation, simulation_session)

            if simulation["n_valid_simulation_results"] >= target_simulation_n:
                simulation["completed"] = True
                simulation["notes"].append("Simulation completed!")
                simulation["complete_at"] = datetime.now()
                logger.info("Split mode: simulation completed early at chunk %d", chunk_index)
                return

        if not terminated_by_failure:
            if simulation["n_valid_simulation_results"] >= target_simulation_n:
                simulation["completed"] = True
                simulation["notes"].append("Simulation completed!")
                simulation["complete_at"] = datetime.now()
                return


def run_simulation(
    simulation_name,
    phase_name,
    simulation_config,
    instruction_config,
    llm_config,
    db,
    extraFlag=None,
):
    try:
        simulation = initialize_simulation(
            simulation_name,
            phase_name,
            simulation_config,
            instruction_config,
            llm_config,
            extraFlag=extraFlag,
        )
        update_simulations(db=db, data=[simulation])
        logger.info(
            "Simulation created and inserted into the database: %s", simulation["_id"]
        )

        # Initialize the agent once outside the loop (used for non-concurrent paths)
        agent = EconomicAgent(simulation_config, instruction_config, llm_config)

        target_simulation_n = simulation_config["target_simulation_n"]
        logger.info(
            "Target number of valid simulation results: %d", target_simulation_n
        )

        schema_check_failed_first_response = False

        # Split-mode BatchSimulate: chunked thread, each chunk is its own session.
        if (
            simulation_config.get("simulation_mode") == "BatchSimulate"
            and instruction_config.get("explain_reasoning_mode") == "split"
        ):
            _run_split_mode(
                agent=agent,
                simulation=simulation,
                simulation_config=simulation_config,
                instruction_config=instruction_config,
                target_simulation_n=target_simulation_n,
                db=db,
            )
            update_simulations(db=db, data=[simulation])
            return

        # IterativeSimulate concurrency: allow multiple in-flight API calls.
        # Only the main thread mutates `simulation` / writes to DB to avoid lost updates.
        if simulation_config.get("simulation_mode") == "IterativeSimulate":
            workers = int(simulation_config.get("iterative_workers", 5))
            if workers < 1:
                workers = 1

            logger.info("IterativeSimulate concurrency enabled. workers=%d", workers)

            thread_local = threading.local()

            def _get_thread_agent() -> EconomicAgent:
                # One agent per thread; do not share agent across threads.
                if not hasattr(thread_local, "agent"):
                    thread_local.agent = EconomicAgent(
                        simulation_config, instruction_config, llm_config
                    )
                return thread_local.agent

            def _attempt_once():
                simulation_session = create_simulation_session(simulation["_id"])
                try:
                    a = _get_thread_agent()
                    success, result = a.get_decisions()
                except Exception as e:
                    success, result = False, {"result": {}, "msg": str(e)}

                simulation_session.update(
                    {
                        "agent_response_success": success,
                        "result": result,
                        "updated_at": datetime.now(),
                    }
                )

                if not success:
                    return simulation_session, False, result, None, None

                schema_check, msg = run_schema_check(
                    result["result"], simulation_config, instruction_config
                )
                if schema_check:
                    decisions = [result["result"]["response"]]
                    return simulation_session, True, result, decisions, None
                return simulation_session, True, result, None, msg

            with ThreadPoolExecutor(max_workers=workers) as executor:
                in_flight = {executor.submit(_attempt_once) for _ in range(workers)}

                terminate = False
                while in_flight and not terminate:
                    done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                    for fut in done:
                        try:
                            simulation_session, success, result, decisions, msg = (
                                fut.result()
                            )
                        except Exception as e:
                            # Shouldn't happen (we catch in worker), but keep safe.
                            simulation_session = create_simulation_session(
                                simulation["_id"]
                            )
                            success, result, decisions, msg = (
                                False,
                                {"msg": str(e)},
                                None,
                                None,
                            )
                            simulation_session.update(
                                {
                                    "agent_response_success": False,
                                    "result": result,
                                    "updated_at": datetime.now(),
                                }
                            )

                        if not success:
                            simulation["failed_sessions"].append(
                                simulation_session["_id"]
                            )
                            logger.warning(
                                "Simulation session failed: %s",
                                result.get("msg", "unknown error"),
                            )
                        else:
                            if decisions is not None:
                                handle_successful_response(
                                    simulation, simulation_session, result, decisions
                                )
                            else:
                                handle_failed_response(
                                    simulation, simulation_session, msg
                                )

                        update_simulation_status(db, simulation, simulation_session)

                        if check_termination_conditions(
                            simulation,
                            target_simulation_n,
                            schema_check_failed_first_response,
                        ):
                            update_simulations(db=db, data=[simulation])
                            terminate = True
                            break

                        # Keep the pipeline full until we terminate.
                        in_flight.add(executor.submit(_attempt_once))

                # Best-effort cancel of remaining futures; in-flight HTTP cannot be forcibly stopped.
                if terminate:
                    for fut in in_flight:
                        fut.cancel()
            return

        while True:
            simulation_session = create_simulation_session(simulation["_id"])
            # Use the new agent.get_decisions() method
            success, result = agent.get_decisions()

            simulation_session.update(
                {
                    "agent_response_success": success,
                    "result": result,
                    "updated_at": datetime.now(),
                }
            )

            if not success:
                simulation["failed_sessions"].append(simulation_session["_id"])
                logger.warning("Simulation session failed: %s", result["msg"])
            else:
                schema_check, msg = run_schema_check(
                    result["result"], simulation_config, instruction_config
                )
                if schema_check:
                    decisions = (
                        [result["result"]["response"]]
                        if simulation_config["simulation_mode"] == "IterativeSimulate"
                        else result["result"]["all_responses"]
                    )
                    handle_successful_response(
                        simulation, simulation_session, result, decisions
                    )

                    # Update the agent's simulation_config for appending mode
                    if (
                        simulation_config["simulation_mode"] == "BatchSimulate"
                        and simulation_config["batch_mode"] == "Appending"
                    ):
                        simulation_config["previous_responses"].append(decisions)
                        # Update the agent's configuration
                        agent.simulation_config = simulation_config
                else:
                    handle_failed_response(simulation, simulation_session, msg)
                    if (
                        simulation_config["simulation_mode"] == "BatchSimulate"
                        and simulation_config["batch_mode"] == "Appending"
                    ):
                        schema_check_failed_first_response = True

            update_simulation_status(db, simulation, simulation_session)

            if check_termination_conditions(
                simulation,
                target_simulation_n,
                schema_check_failed_first_response,
                max_failed_sessions=MAX_FAILED_SESSION_ALLOW_BATCH,
            ):
                update_simulations(db=db, data=[simulation])
                break

    except Exception as e:
        logger.error("Error during simulation run: %s", e)


def run_simulation_to_json(
    *,
    experiment_name,
    simulation_name,
    phase_name,
    simulation_config,
    instruction_config,
    llm_config,
    data_root: str | Path | None = None,
    extraFlag=None,
):
    """Run one simulation and persist it under ``data/<experiment_name>/``.

    This is the replication-friendly entrypoint. It keeps the original
    ``run_simulation`` implementation intact, but constructs a local JSON
    database instead of requiring MongoDB.
    """
    db = get_database(
        target="local_json",
        data_root=Path(data_root) if data_root else DATA_ROOT,
        experiment_name=experiment_name,
    )
    return run_simulation(
        simulation_name=simulation_name,
        phase_name=phase_name,
        simulation_config=simulation_config,
        instruction_config=instruction_config,
        llm_config=llm_config,
        db=db,
        extraFlag=extraFlag,
    )
