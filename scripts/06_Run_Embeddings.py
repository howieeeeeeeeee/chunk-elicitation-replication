"""Run selected reasoning embeddings into experiment-local JSON."""

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv


REPLICATION_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPLICATION_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
load_dotenv(REPLICATION_ROOT / ".env")

from db_ops.config import get_database
from db_ops.embeddings import setup_embedding_indexes
from embedding.config import EmbeddingConfigurationError, normalize_embedding_config
from embedding.orchestration import (
    EmbeddingEligibilityError,
    embed_simulation,
    summarize_embedding_plan,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)

SIMULATION_IDS = []
EMBEDDING_MODEL = "REPLACE_WITH_OPENROUTER_EMBEDDING_MODEL"
MODEL_PLACEHOLDER = "REPLACE_WITH_OPENROUTER_EMBEDDING_MODEL"


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPLICATION_ROOT / "data",
        help="Top-level replication data directory.",
    )
    parser.add_argument(
        "--experiment",
        choices=["exp1", "exp2", "exp3"],
        default="exp1",
    )
    parser.add_argument("--simulation-id", action="append", default=[])
    parser.add_argument("--model", default=EMBEDDING_MODEL)
    parser.add_argument("--dimensions", type=positive_integer)
    parser.add_argument("--input-type")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def resolve_simulation_ids(explicit_ids: list[str]) -> list[str]:
    selected = explicit_ids if explicit_ids else SIMULATION_IDS
    return list(dict.fromkeys(item.strip() for item in selected if item.strip()))


def build_embedding_config(args) -> dict:
    config = {"model": args.model, "encoding_format": "float"}
    if args.dimensions is not None:
        config["dimensions"] = args.dimensions
    if args.input_type is not None:
        config["input_type"] = args.input_type
    return normalize_embedding_config(config)


def _log_plan(summary: dict, experiment: str) -> None:
    logger.info(
        "Plan experiment=%s selected=%d eligible=%d ineligible=%d missing=%d "
        "sessions=%d decisions=%d new=%d successes=%d failures=%d "
        "malformed=%d attempts=%d",
        experiment,
        summary["selected_simulations"],
        summary["eligible_simulations"],
        summary["ineligible_simulations"],
        summary["missing_simulations"],
        summary["sessions"],
        summary["decisions"],
        summary["new_embeddings"],
        summary["existing_successes"],
        summary["existing_failures"],
        summary["malformed"],
        summary["planned_attempts"],
    )


def main(argv=None, *, database_factory=get_database, input_func=input) -> int:
    args = argument_parser().parse_args(argv)
    simulation_ids = resolve_simulation_ids(args.simulation_id)
    if not simulation_ids:
        logger.warning("No simulation ids selected; no work will run.")
        return 0 if args.dry_run else 2
    if args.model.strip() == MODEL_PLACEHOLDER:
        logger.warning("Embedding model placeholder is unresolved; no work will run.")
        return 2
    try:
        embedding_config = build_embedding_config(args)
    except EmbeddingConfigurationError as error:
        logger.warning("Invalid embedding configuration: %s", error)
        return 2

    db = database_factory(
        data_root=args.data_root,
        experiment_name=args.experiment,
    )
    try:
        summary = summarize_embedding_plan(
            db, simulation_ids, embedding_config, active_logger=logger
        )
    except EmbeddingEligibilityError as error:
        logger.warning("Embedding source is not eligible: %s", error)
        return 2
    _log_plan(summary, args.experiment)
    if args.dry_run:
        logger.info("Dry run complete; no files, API calls, or writes occurred.")
        return 0
    if summary["planned_attempts"] == 0:
        logger.info("No embedding attempts are currently eligible.")
        return 0
    if not args.yes:
        try:
            response = input_func(
                f"Proceed with {summary['planned_attempts']} embedding attempts? [y/N]: "
            )
        except EOFError:
            response = ""
        if response.strip().lower() != "y":
            logger.info("Embedding run cancelled before writes or API calls.")
            return 3

    setup_embedding_indexes(db)
    attempted_ids = set()
    totals = {"processed": 0, "succeeded": 0, "failed": 0, "skipped": 0}
    for simulation_id in simulation_ids:
        try:
            result = embed_simulation(
                db,
                simulation_id,
                embedding_config,
                attempted_ids=attempted_ids,
                active_logger=logger,
            )
        except EmbeddingEligibilityError as error:
            logger.warning("Embedding source became ineligible: %s", error)
            return 2
        for field in totals:
            totals[field] += result[field]
    logger.info(
        "Embedding run complete processed=%d succeeded=%d failed=%d skipped=%d",
        totals["processed"],
        totals["succeeded"],
        totals["failed"],
        totals["skipped"],
    )
    return 1 if totals["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
