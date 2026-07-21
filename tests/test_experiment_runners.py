from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from _common import argument_parser, main_for_experiment
from plan import (
    build_plan,
    make_jobs,
    record_signature,
)
from experiment_specs import EXPERIMENT_1, EXPERIMENT_2, EXPERIMENT_3


def record_for(cell, *, completed=True):
    return {
        "_id": f"record-{cell['section']}-{cell['simulation_name']}-{cell['game']}",
        "name": cell["simulation_name"],
        "phase_name": cell["phase_name"],
        "extraFlag": list(cell["extra_flag"]),
        "simulation_config": copy.deepcopy(cell["config"]["simulation_config"]),
        "instruction_config": copy.deepcopy(cell["config"]["instruction_config"]),
        "llm_config": copy.deepcopy(cell["config"]["llm_config"]),
        "completed": completed,
        "archived": False,
    }


class ReplicationRunnerTests(unittest.TestCase):
    def test_expected_manifest_cardinality(self):
        for experiment, expected in (
            (EXPERIMENT_1, 672),
            (EXPERIMENT_2, 1320),
            (EXPERIMENT_3, 165),
        ):
            with self.subTest(experiment=experiment["key"]):
                jobs, invalid = make_jobs(experiment)
                self.assertEqual(expected, len(jobs))
                self.assertEqual(0, invalid)
                self.assertEqual(expected, len({job["signature"] for job in jobs}))

    def test_historical_absent_defaults_match_formal_signature(self):
        cell = make_jobs(EXPERIMENT_1)[0][0]
        record = record_for(cell)
        record.pop("extraFlag")
        record["llm_config"].pop("frequency_penalty")
        record["simulation_config"]["iterative_workers"] = 5
        self.assertEqual(cell["signature"], record_signature(record))

    def test_completed_incomplete_duplicate_and_legacy_classification(self):
        jobs, _ = make_jobs(EXPERIMENT_3)
        completed = record_for(jobs[0])
        duplicate = record_for(jobs[0])
        duplicate["_id"] = "duplicate-completed"
        incomplete = record_for(jobs[1], completed=False)
        legacy = record_for(jobs[2])
        legacy["llm_config"]["model"] = "google/gemini-3-pro-preview"

        plan = build_plan(
            EXPERIMENT_3,
            existing_records=[completed, duplicate, incomplete, legacy],
        )
        self.assertEqual(1, plan["counts"]["completed"])
        self.assertEqual(1, plan["counts"]["incomplete"])
        self.assertEqual(163, plan["counts"]["missing"])
        self.assertEqual(163, plan["counts"]["runnable"])
        self.assertEqual(1, plan["counts"]["duplicate_completed_signatures"])
        self.assertEqual(1, plan["counts"]["legacy_records"])

    def test_replication_cli_has_json_options_and_no_mongodb_options(self):
        parser = argument_parser(EXPERIMENT_1)
        option_strings = {
            option
            for action in parser._actions
            for option in action.option_strings
        }
        self.assertIn("--data-root", option_strings)
        self.assertNotIn("--db-target", option_strings)
        self.assertNotIn("--db-name", option_strings)

    def test_dry_run_does_not_create_json_directories_or_execute(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir) / "not-created"
            argv = [
                "02_Run_Experiment_1.py",
                "--dry-run",
                "--data-root",
                str(data_root),
            ]
            with (
                patch("_common.run_jobs") as run_jobs,
                patch.object(sys, "argv", argv),
            ):
                self.assertEqual(0, main_for_experiment(EXPERIMENT_1))
            run_jobs.assert_not_called()
            self.assertFalse(data_root.exists())


if __name__ == "__main__":
    unittest.main()
