import json
import sys
import tempfile
import unittest
from pathlib import Path


REPLICATION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPLICATION_ROOT / "src"))

from db_ops.config import get_combined_database, get_database
from db_ops.embeddings import (
    find_successful_embedding,
    query_embeddings,
    upsert_embedding,
)
from embedding.entities import build_embedding_entity, record_embedding_failure
from embedding.orchestration import embed_simulation


CONFIG = {"model": "embedding/model", "dimensions": 3}


def response():
    return {
        "response_id": "response-1",
        "resolved_model": "resolved/model",
        "object": "embedding",
        "vector": [0.1, 0.2, 0.3],
        "vector_dimension": 3,
        "usage": {"prompt_tokens": 4, "total_tokens": 4, "cost": 0.01},
    }


class LocalJsonEmbeddingTests(unittest.TestCase):
    def test_database_construction_is_read_only_and_embeddings_are_atomic(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary) / "data"
            database = get_database(data_root=data_root, experiment_name="exp1")
            self.assertFalse((data_root / "exp1").exists())

            entity = build_embedding_entity(
                "simulation-1", "session-1", 0, "reasoning", CONFIG
            )
            result = upsert_embedding(database, entity)
            self.assertEqual(entity["_id"], result["upserted_id"])
            embedding_path = data_root / "exp1/embeddings.json"
            self.assertTrue(embedding_path.exists())
            self.assertEqual([entity], json.loads(embedding_path.read_text()))
            self.assertEqual([], list(data_root.rglob("*.tmp")))

    def test_upsert_retrieval_immutability_and_combined_database(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary) / "data"
            first_db = get_database(data_root=data_root, experiment_name="exp1")
            second_db = get_database(data_root=data_root, experiment_name="exp2")
            first = build_embedding_entity(
                "simulation-1", "session-1", 0, "reasoning one", CONFIG
            )
            second = build_embedding_entity(
                "simulation-2", "session-2", 0, "reasoning two", CONFIG
            )
            upsert_embedding(first_db, first)
            upsert_embedding(second_db, second)

            failed = record_embedding_failure(first, RuntimeError("failed"))
            update = upsert_embedding(first_db, failed)
            self.assertEqual(1, update["matched_count"])
            self.assertEqual(1, update["modified_count"])
            self.assertIsNone(find_successful_embedding(first_db, first["_id"]))
            self.assertEqual(
                1,
                len(
                    query_embeddings(
                        first_db,
                        simulation_id="simulation-1",
                        embedding_config_hash=first["embedding_config_hash"],
                        success=False,
                    )
                ),
            )

            changed = build_embedding_entity(
                "simulation-1", "session-1", 0, "changed reasoning", CONFIG
            )
            with self.assertRaisesRegex(ValueError, "immutable"):
                upsert_embedding(first_db, changed)

            combined = get_combined_database(data_root=data_root)
            self.assertEqual(2, len(combined.embeddings.find({})))

    def test_orchestration_writes_only_experiment_embedding_collection(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary) / "data"
            database = get_database(data_root=data_root, experiment_name="exp1")
            database.simulations.bulk_upsert(
                [
                    {
                        "_id": "simulation-1",
                        "instruction_config": {"explain_reasoning": True},
                        "simulation_sessions": ["session-1"],
                    }
                ]
            )
            database.simulation_sessions.bulk_upsert(
                [
                    {
                        "_id": "session-1",
                        "simulation_id": "simulation-1",
                        "decisions": [[[1], "private reasoning"]],
                    }
                ]
            )
            simulation_before = (data_root / "exp1/simulations.json").read_bytes()
            session_before = (data_root / "exp1/simulation_sessions.json").read_bytes()
            summary = embed_simulation(
                database,
                "simulation-1",
                CONFIG,
                request_fn=lambda *_: response(),
            )
            self.assertEqual(1, summary["succeeded"])
            self.assertEqual(
                simulation_before, (data_root / "exp1/simulations.json").read_bytes()
            )
            self.assertEqual(
                session_before,
                (data_root / "exp1/simulation_sessions.json").read_bytes(),
            )
            saved = json.loads((data_root / "exp1/embeddings.json").read_text())
            self.assertEqual(1, len(saved))
            self.assertTrue(saved[0]["success"])


if __name__ == "__main__":
    unittest.main()
