import copy
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import _common as common


def load_runner(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


experiment_1 = load_runner("02_Run_Experiment_1.py", "formal_experiment_1")
experiment_2 = load_runner("03_Run_Experiment_2.py", "formal_experiment_2")
experiment_3 = load_runner("04_Run_Experiment_3.py", "formal_experiment_3")


def get_by_dot_path(document, path):
    current = document
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def matches(record, query):
    for key, expected in query.items():
        if key == "$or":
            if not any(matches(record, branch) for branch in expected):
                return False
            continue
        actual = get_by_dot_path(record, key)
        if isinstance(expected, dict) and "$exists" in expected:
            if (actual is not None) is not expected["$exists"]:
                return False
        elif actual != expected:
            return False
    return True


class ReadOnlyCollection:
    def __init__(self, records=()):
        self.records = list(records)
        self.find_calls = []

    def find(self, query):
        self.find_calls.append(copy.deepcopy(query))
        return [
            copy.deepcopy(record)
            for record in self.records
            if matches(record, query)
        ]


class ReadOnlyDatabase:
    def __init__(self, records=()):
        self.simulations = ReadOnlyCollection(records)


class FormalRunnerTests(unittest.TestCase):
    def test_public_runners_share_complete_normalized_signature(self):
        self.assertNotIn(
            "simulation_config.iterative_workers",
            common.FORMAL_SIGNATURE_FIELDS,
        )
        for runner in (experiment_1, experiment_2, experiment_3):
            self.assertEqual(
                list(common.FORMAL_SIGNATURE_FIELDS),
                runner.LIST_OF_PARAMETERS,
            )

    def test_completed_matching_normalizes_defaults_and_empty_flag(self):
        config = copy.deepcopy(experiment_2.BASELINE_CONFIG)
        config["simulation_config"]["game_type"] = experiment_2.BEHAVIOR_GAMES[0]
        config["llm_config"]["model"] = experiment_2.MODELS["with_thinking"][0]
        common.apply_overrides(
            config,
            experiment_2.PARAMETER_COMBINATIONS["with_thinking"][0],
        )
        record = {
            "_id": "formal-cell",
            "phase_name": experiment_2.PHASE_NAME,
            "extraFlag": None,
            "archived": False,
            "completed": True,
            **copy.deepcopy(config),
        }
        record["simulation_config"]["iterative_workers"] = 5
        del record["instruction_config"]["split_n"]
        del record["instruction_config"]["context"]
        database = ReadOnlyDatabase([record])

        self.assertTrue(
            common.simulation_exists(
                db=database,
                config=config,
                phase_name=experiment_2.PHASE_NAME,
                list_of_parameters=experiment_2.LIST_OF_PARAMETERS,
                extra_flag=[],
            )
        )

        changed = copy.deepcopy(config)
        changed["llm_config"]["temperature"] = 2
        self.assertFalse(
            common.simulation_exists(
                db=database,
                config=changed,
                phase_name=experiment_2.PHASE_NAME,
                list_of_parameters=experiment_2.LIST_OF_PARAMETERS,
                extra_flag=[],
            )
        )

    def test_plan_building_is_read_only(self):
        database = ReadOnlyDatabase()
        with patch.object(common, "get_database", return_value=database):
            jobs, counts = common.build_jobs(
                games=[experiment_1.GAMES[0]],
                models={"with_thinking": experiment_1.MODELS["with_thinking"]},
                parameter_combinations={
                    "with_thinking": experiment_1.PARAMETER_COMBINATIONS[
                        "with_thinking"
                    ][:1]
                },
                baseline_config=experiment_1.BASELINE_CONFIG,
                phase_name=experiment_1.PHASE_NAME,
                list_of_parameters=experiment_1.LIST_OF_PARAMETERS,
                experiment_name=experiment_1.EXPERIMENT_NAME,
                data_root=ROOT / "data",
                extra_flag=[],
                treatment_overrides=experiment_1.TREATMENT_OVERRIDES[:1],
            )

        self.assertEqual(1, counts["to_run"])
        self.assertEqual(1, len(jobs))
        self.assertIn("signature", jobs[0])
        self.assertGreater(len(database.simulations.find_calls), 0)


if __name__ == "__main__":
    unittest.main()
