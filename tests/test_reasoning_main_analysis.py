import sys
import unittest
from pathlib import Path
from unittest.mock import patch


REPLICATION_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(REPLICATION_ROOT / "src"),
    str(REPLICATION_ROOT / "scripts"),
]

from main_analysis.environment import get_analysis_database
from main_analysis.reasoning import SIMULATION_SPECS, _load_corpus


class ReasoningMainAnalysisTests(unittest.TestCase):
    def test_corpus_uses_one_batch_query_per_selected_collection(self):
        database = get_analysis_database()
        collections = {
            name: database[name]
            for name in ("simulations", "simulation_sessions", "embeddings")
        }
        with (
            patch.object(
                collections["simulations"],
                "find",
                wraps=collections["simulations"].find,
            ) as simulations_find,
            patch.object(
                collections["simulation_sessions"],
                "find",
                wraps=collections["simulation_sessions"].find,
            ) as sessions_find,
            patch.object(
                collections["embeddings"],
                "find",
                wraps=collections["embeddings"].find,
            ) as embeddings_find,
        ):
            frame, matrix = _load_corpus(database)

        self.assertEqual((1200, 4096), matrix.shape)
        self.assertEqual(1200, len(frame))
        self.assertEqual(12, len(SIMULATION_SPECS))
        for query in (
            simulations_find.call_args.args[0],
            sessions_find.call_args.args[0],
            embeddings_find.call_args.args[0],
        ):
            self.assertEqual({"_id"}, set(query))
            self.assertEqual({"$in"}, set(query["_id"]))
        self.assertEqual(1, simulations_find.call_count)
        self.assertEqual(1, sessions_find.call_count)
        self.assertEqual(1, embeddings_find.call_count)


if __name__ == "__main__":
    unittest.main()
