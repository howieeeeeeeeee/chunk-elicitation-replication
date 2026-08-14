import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPLICATION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPLICATION_ROOT / "src"))

from db_ops.local_json_db import LocalJsonCollection, matches_filter


class LocalJsonCollectionPerformanceTests(unittest.TestCase):
    def test_reuses_parsed_records_and_refreshes_external_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "records.json"
            path.write_text(
                json.dumps([{"_id": "first", "value": 1}]),
                encoding="utf-8",
            )
            collection = LocalJsonCollection(path)

            with patch(
                "db_ops.local_json_db.json.load",
                wraps=json.load,
            ) as load:
                self.assertEqual(1, collection.find_one({"_id": "first"})["value"])
                self.assertEqual(1, collection.find_one({"_id": "first"})["value"])
                self.assertEqual(1, load.call_count)

                path.write_text(
                    json.dumps([{"_id": "second", "value": 200}]),
                    encoding="utf-8",
                )
                self.assertIsNone(collection.find_one({"_id": "first"}))
                self.assertEqual(
                    200,
                    collection.find_one({"_id": "second"})["value"],
                )
                self.assertEqual(2, load.call_count)

    def test_write_operations_refresh_cache_across_collection_handles(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "records.json"
            writer = LocalJsonCollection(path)
            reader = LocalJsonCollection(path)

            writer.replace_all([{"_id": "one", "value": 1}])
            returned = reader.find_one({"_id": "one"})
            returned["value"] = -1
            self.assertEqual(1, reader.find_one({"_id": "one"})["value"])

            writer.bulk_upsert(
                [
                    {"_id": "one", "value": 2},
                    {"_id": "two", "value": 3},
                ]
            )
            self.assertEqual(2, reader.find_one({"_id": "one"})["value"])
            self.assertEqual(3, reader.find_one({"_id": "two"})["value"])

            writer.replace_all([{"_id": "three", "value": 4}])
            self.assertIsNone(reader.find_one({"_id": "one"}))
            self.assertEqual(4, reader.find_one({"_id": "three"})["value"])

    def test_exact_id_index_preserves_projections_and_fallback_queries(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "records.json"
            collection = LocalJsonCollection(path)
            collection.replace_all(
                [
                    {
                        "_id": "first",
                        "kind": "a",
                        "nested": {"value": 1},
                    },
                    {
                        "_id": "second",
                        "kind": "b",
                        "nested": {"value": 2},
                    },
                ]
            )

            with patch(
                "db_ops.local_json_db.matches_filter",
                wraps=matches_filter,
            ) as matcher:
                self.assertEqual(
                    {"nested.value": 2},
                    collection.find_one(
                        {"_id": "second"},
                        {"nested.value": 1},
                    ),
                )
                self.assertIsNone(collection.find_one({"_id": "missing"}))
                self.assertEqual(0, matcher.call_count)

                self.assertEqual(
                    {"_id": "second"},
                    collection.find_one({"kind": "b"}, {"_id": 1}),
                )
                self.assertEqual(
                    "first",
                    collection.find_one({"_id": {"$in": ["first"]}})["_id"],
                )
                self.assertGreater(matcher.call_count, 0)


if __name__ == "__main__":
    unittest.main()
