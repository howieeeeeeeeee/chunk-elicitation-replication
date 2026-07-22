import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


REPLICATION_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPLICATION_ROOT / "scripts/06_Run_Embeddings.py"
SPEC = importlib.util.spec_from_file_location("run_embeddings", SCRIPT_PATH)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


PLAN = {
    "selected_simulations": 1,
    "eligible_simulations": 1,
    "ineligible_simulations": 0,
    "missing_simulations": 0,
    "sessions": 1,
    "missing_sessions": 0,
    "decisions": 1,
    "new_embeddings": 1,
    "existing_successes": 0,
    "existing_failures": 0,
    "malformed": 0,
    "planned_attempts": 1,
}


class ReplicationEmbeddingRunnerTests(unittest.TestCase):
    def test_checked_in_default_dry_run_never_opens_database(self):
        database_factory = Mock()
        self.assertEqual(
            0,
            runner.main(["--dry-run"], database_factory=database_factory),
        )
        database_factory.assert_not_called()

    def test_explicit_dry_run_reads_without_creating_embedding_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary) / "data"
            exp_dir = data_root / "exp1"
            exp_dir.mkdir(parents=True)
            (exp_dir / "simulations.json").write_text(
                json.dumps(
                    [
                        {
                            "_id": "simulation-1",
                            "instruction_config": {"explain_reasoning": True},
                            "simulation_sessions": ["session-1"],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (exp_dir / "simulation_sessions.json").write_text(
                json.dumps(
                    [
                        {
                            "_id": "session-1",
                            "simulation_id": "simulation-1",
                            "decisions": [[[1], "private reasoning"]],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            status = runner.main(
                [
                    "--data-root",
                    str(data_root),
                    "--experiment",
                    "exp1",
                    "--simulation-id",
                    "simulation-1",
                    "--model",
                    "embedding/model",
                    "--dry-run",
                ]
            )
            self.assertEqual(0, status)
            self.assertFalse((exp_dir / "embeddings.json").exists())

    @patch.object(runner, "summarize_embedding_plan")
    @patch.object(runner, "embed_simulation")
    @patch.object(runner, "setup_embedding_indexes")
    def test_confirmation_and_yes_gate_applied_calls(
        self, setup_indexes, embed_simulation, summarize
    ):
        summarize.return_value = PLAN
        database = object()
        common = [
            "--simulation-id",
            "simulation-1",
            "--model",
            "embedding/model",
        ]
        cancelled = runner.main(
            common,
            database_factory=lambda **_: database,
            input_func=lambda _: "n",
        )
        self.assertEqual(3, cancelled)
        setup_indexes.assert_not_called()
        embed_simulation.assert_not_called()

        embed_simulation.return_value = {
            "processed": 1,
            "succeeded": 1,
            "failed": 0,
            "skipped": 0,
        }
        applied = runner.main(
            [*common, "--yes"],
            database_factory=lambda **_: database,
        )
        self.assertEqual(0, applied)
        setup_indexes.assert_called_once_with(database)
        embed_simulation.assert_called_once()


if __name__ == "__main__":
    unittest.main()
