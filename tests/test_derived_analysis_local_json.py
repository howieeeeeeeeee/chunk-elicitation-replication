import copy
import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path


REPLICATION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPLICATION_ROOT / "src"))

from db_ops.config import get_combined_database, get_database
from db_ops.kmeans_analyses import (
    complete_cluster_summary_attempt,
    complete_clustering_attempt,
    fail_cluster_summary_attempt,
    find_completed_cluster_summary,
    find_completed_exact_cluster_summary,
    find_completed_clustering,
    register_summary_run,
    reuse_completed_cluster_summary,
    start_cluster_summary_attempt,
    start_clustering_attempt,
    upsert_kmeans_analysis,
)
from db_ops.pca_analyses import (
    complete_pca_attempt,
    find_completed_pca_analysis,
    query_pca_analyses,
    start_pca_attempt,
    upsert_pca_analysis,
)
from derived_analysis.kmeans import (
    build_kmeans_analysis,
    rendered_prompt_hash,
    summary_config_hash,
)
from derived_analysis.pca import build_pca_analysis


FIXED = datetime(2026, 7, 23, 3, 0, tzinfo=timezone.utc)
LATER = datetime(2026, 7, 23, 3, 1, tzinfo=timezone.utc)
IDS = ["embedding-b", "embedding-a"]
PCA_CONFIG = {"n_components": 2}
CLUSTER_CONFIG = {"n_clusters": 2, "pca_component_indices": [0, 1]}
SUMMARY_CONFIG = {
    "model": "openai/model",
    "prompt_version": "summary-v1",
    "reasoning": {"effort": "low"},
}


def pca_output():
    return {
        "coordinates": [
            {"embedding_id": "embedding-a", "values": [0.1, 0.2]},
            {"embedding_id": "embedding-b", "values": [0.3, 0.4]},
        ],
        "n_samples": 2,
        "n_input_dimensions": 3,
        "n_components": 2,
        "diagnostics": {"explained_variance_ratio": [0.8, 0.1]},
    }


def clustering_output():
    return {
        "assignments": [
            {"embedding_id": "embedding-a", "cluster_id": 0},
            {"embedding_id": "embedding-b", "cluster_id": 1},
        ],
        "centroids": [
            {"cluster_id": 0, "values": [0.1, 0.2]},
            {"cluster_id": 1, "values": [0.3, 0.4]},
        ],
        "n_clusters": 2,
        "n_features": 2,
        "diagnostics": {"inertia": 0.0},
    }


def summary_response():
    return {
        "summary": "Cluster zero summary",
        "response_id": "response-0",
        "resolved_model": "openai/resolved",
        "finish_reason": "stop",
        "native_finish_reason": "stop",
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 3,
            "total_tokens": 8,
            "cost": 0.01,
        },
    }


class SummaryFailure(RuntimeError):
    usage = {
        "prompt_tokens": 5,
        "completion_tokens": 0,
        "total_tokens": 5,
        "cost": 0.004,
    }


class DerivedAnalysisLocalJsonTests(unittest.TestCase):
    def test_shared_derived_database_is_read_only_until_first_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary) / "data"
            exp1 = get_database(data_root=data_root, experiment_name="exp1")
            exp2 = get_database(data_root=data_root, experiment_name="exp2")
            combined = get_combined_database(data_root=data_root)
            self.assertFalse((data_root / "derived").exists())

            pca = build_pca_analysis(IDS, PCA_CONFIG, timestamp=FIXED)
            upsert_pca_analysis(exp1, pca)
            path = data_root / "derived/pca_analyses.json"
            self.assertTrue(path.exists())
            self.assertEqual([pca], json.loads(path.read_text()))
            self.assertEqual(pca, exp2.pca_analyses.find_one({"_id": pca["_id"]}))
            self.assertEqual(
                pca, combined.pca_analyses.find_one({"_id": pca["_id"]})
            )
            self.assertFalse((data_root / "exp1/pca_analyses.json").exists())
            self.assertFalse((data_root / "exp2/pca_analyses.json").exists())
            self.assertEqual([], list(data_root.rglob("*.tmp")))

    def test_pca_kmeans_and_summary_transitions_share_one_derived_store(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary) / "data"
            database = get_database(data_root=data_root, experiment_name="exp1")
            pca = build_pca_analysis(IDS, PCA_CONFIG, timestamp=FIXED)
            upsert_pca_analysis(database, pca)
            start_pca_attempt(database, pca["_id"], timestamp=FIXED)
            pca = complete_pca_attempt(
                database, pca["_id"], pca_output(), timestamp=LATER
            )
            self.assertEqual(
                pca, find_completed_pca_analysis(database, pca["_id"])
            )
            self.assertEqual(
                1,
                len(
                    query_pca_analyses(
                        database,
                        pca_config_hash=pca["pca_config_hash"],
                        status="complete",
                    )
                ),
            )

            kmeans = build_kmeans_analysis(
                IDS,
                {"kind": "pca", "pca_analysis_id": pca["_id"]},
                CLUSTER_CONFIG,
                timestamp=FIXED,
            )
            upsert_kmeans_analysis(database, kmeans)
            start_clustering_attempt(database, kmeans["_id"], timestamp=FIXED)
            kmeans = complete_clustering_attempt(
                database,
                kmeans["_id"],
                clustering_output(),
                timestamp=LATER,
            )
            self.assertEqual(
                kmeans, find_completed_clustering(database, kmeans["_id"])
            )

            kmeans = register_summary_run(
                database, kmeans["_id"], SUMMARY_CONFIG, timestamp=LATER
            )
            summary_hash = summary_config_hash(SUMMARY_CONFIG)
            start_cluster_summary_attempt(
                database,
                kmeans["_id"],
                summary_hash,
                0,
                input_hash="input-hash",
                prompt="Exact UTF-8 prompt: 合作",
                timestamp=LATER,
            )
            complete_cluster_summary_attempt(
                database,
                kmeans["_id"],
                summary_hash,
                0,
                summary_response(),
                timestamp=LATER,
            )
            saved = find_completed_cluster_summary(
                database, kmeans["_id"], summary_hash, 0
            )
            self.assertEqual("complete", saved["status"])
            self.assertEqual("Exact UTF-8 prompt: 合作", saved["prompt"])
            self.assertEqual(
                rendered_prompt_hash("Exact UTF-8 prompt: 合作"),
                saved["prompt_hash"],
            )
            self.assertTrue(saved["exact_prompt_verified"])
            self.assertEqual(0.01, saved["output"]["usage"]["cost"])
            self.assertEqual(
                1,
                len(
                    json.loads(
                        (
                            data_root / "derived/kmeans_analyses.json"
                        ).read_text()
                    )
                ),
            )
            self.assertEqual([], list(data_root.rglob("*.tmp")))

    def test_exact_reuse_requires_matching_config_and_prompt(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary) / "data"
            database = get_database(data_root=data_root, experiment_name="exp1")
            kmeans = build_kmeans_analysis(
                IDS,
                {"kind": "embeddings"},
                CLUSTER_CONFIG,
                timestamp=FIXED,
            )
            upsert_kmeans_analysis(database, kmeans)
            start_clustering_attempt(database, kmeans["_id"], timestamp=FIXED)
            complete_clustering_attempt(
                database, kmeans["_id"], clustering_output(), timestamp=LATER
            )
            kmeans = register_summary_run(
                database, kmeans["_id"], SUMMARY_CONFIG, timestamp=LATER
            )
            summary_hash = summary_config_hash(SUMMARY_CONFIG)
            start_cluster_summary_attempt(
                database,
                kmeans["_id"],
                summary_hash,
                0,
                input_hash="input-hash",
                prompt="Reusable exact prompt",
                timestamp=LATER,
            )
            complete_cluster_summary_attempt(
                database,
                kmeans["_id"],
                summary_hash,
                0,
                summary_response(),
                timestamp=LATER,
            )
            source = find_completed_exact_cluster_summary(
                database,
                summary_hash,
                "Reusable exact prompt",
            )
            self.assertEqual(0, source["cluster_id"])
            self.assertIsNone(
                find_completed_exact_cluster_summary(
                    database,
                    summary_hash,
                    "Changed exact prompt",
                )
            )
            self.assertIsNone(
                find_completed_exact_cluster_summary(
                    database,
                    "different-summary-config-hash",
                    "Reusable exact prompt",
                )
            )

            reused = reuse_completed_cluster_summary(
                database,
                kmeans["_id"],
                summary_hash,
                1,
                input_hash="reused-input-hash",
                prompt="Reusable exact prompt",
                timestamp=LATER,
            )
            reused_cluster = reused["summaries"][0]["clusters"][1]
            self.assertEqual("complete", reused_cluster["status"])
            self.assertEqual(0, reused_cluster["attempt_count"])
            self.assertEqual([], reused_cluster["attempts"])
            self.assertEqual(0, reused_cluster["reuse"]["source_cluster_id"])
            self.assertNotIn("usage", reused_cluster["output"])

    def test_failed_attempt_history_and_legacy_migration_are_persisted(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary) / "data"
            database = get_database(data_root=data_root, experiment_name="exp1")
            kmeans = build_kmeans_analysis(
                IDS,
                {"kind": "embeddings"},
                CLUSTER_CONFIG,
                timestamp=FIXED,
            )
            upsert_kmeans_analysis(database, kmeans)
            start_clustering_attempt(database, kmeans["_id"], timestamp=FIXED)
            complete_clustering_attempt(
                database, kmeans["_id"], clustering_output(), timestamp=LATER
            )
            register_summary_run(
                database, kmeans["_id"], SUMMARY_CONFIG, timestamp=LATER
            )
            summary_hash = summary_config_hash(SUMMARY_CONFIG)
            start_cluster_summary_attempt(
                database,
                kmeans["_id"],
                summary_hash,
                0,
                input_hash="first-input",
                prompt="First exact prompt",
                timestamp=LATER,
            )
            fail_cluster_summary_attempt(
                database,
                kmeans["_id"],
                summary_hash,
                0,
                SummaryFailure("private failure"),
                timestamp=LATER,
            )
            start_cluster_summary_attempt(
                database,
                kmeans["_id"],
                summary_hash,
                0,
                input_hash="second-input",
                prompt="Second exact prompt",
                timestamp=LATER,
            )
            completed = complete_cluster_summary_attempt(
                database,
                kmeans["_id"],
                summary_hash,
                0,
                summary_response(),
                timestamp=LATER,
            )
            cluster = completed["summaries"][0]["clusters"][0]
            self.assertEqual(2, cluster["attempt_count"])
            self.assertEqual("First exact prompt", cluster["attempts"][0]["prompt"])
            self.assertEqual(0.004, cluster["attempts"][0]["usage"]["cost"])
            self.assertEqual("Second exact prompt", cluster["attempts"][1]["prompt"])

            legacy = copy.deepcopy(completed)
            legacy["schema_version"] = 1
            for summary_run in legacy["summaries"]:
                for summary_cluster in summary_run["clusters"]:
                    summary_cluster.pop("prompt")
                    summary_cluster.pop("exact_prompt_verified")
                    summary_cluster.pop("reuse")
                    for attempt in summary_cluster["attempts"]:
                        attempt.pop("input_hash")
                        attempt.pop("prompt")
                        attempt.pop("prompt_hash")
                        attempt.pop("exact_prompt_verified")
            database.kmeans_analyses.replace_all([legacy])

            migrated = register_summary_run(
                database,
                kmeans["_id"],
                SUMMARY_CONFIG,
                timestamp=LATER,
            )
            migrated_cluster = migrated["summaries"][0]["clusters"][0]
            self.assertEqual(2, migrated["schema_version"])
            self.assertIsNone(migrated_cluster["prompt"])
            self.assertFalse(migrated_cluster["exact_prompt_verified"])
            self.assertEqual(2, migrated_cluster["attempt_count"])
            self.assertEqual(
                0.004,
                migrated_cluster["attempts"][0]["usage"]["cost"],
            )
            self.assertIsNone(
                find_completed_exact_cluster_summary(
                    database,
                    summary_hash,
                    "Second exact prompt",
                )
            )
            self.assertEqual(
                [migrated],
                json.loads(
                    (data_root / "derived/kmeans_analyses.json").read_text()
                ),
            )


if __name__ == "__main__":
    unittest.main()
