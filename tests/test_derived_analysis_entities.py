import copy
import json
import math
import sys
import unittest
from datetime import datetime, timezone


sys.path.insert(0, "src")

from derived_analysis.common import (
    DerivedAnalysisConfigurationError,
    normalize_embedding_ids,
)
from derived_analysis.kmeans import (
    add_summary_run,
    build_kmeans_analysis,
    kmeans_analysis_id,
    record_cluster_summary_completed,
    record_cluster_summary_failed,
    record_cluster_summary_started,
    record_clustering_completed,
    record_clustering_started,
    rendered_prompt_hash,
    summary_config_hash,
    upgrade_kmeans_analysis,
    validate_kmeans_analysis,
)
from derived_analysis.pca import (
    build_pca_analysis,
    pca_analysis_id,
    record_pca_completed,
    record_pca_failed,
    record_pca_started,
    validate_pca_analysis,
)


CREATED = datetime(2026, 7, 23, 1, 0, tzinfo=timezone.utc)
STARTED = datetime(2026, 7, 23, 1, 1, tzinfo=timezone.utc)
FINISHED = datetime(2026, 7, 23, 1, 2, tzinfo=timezone.utc)
RETRIED = datetime(2026, 7, 23, 1, 3, tzinfo=timezone.utc)
RETRY_FINISHED = datetime(2026, 7, 23, 1, 4, tzinfo=timezone.utc)
EMBEDDING_IDS = ["embedding-c", "embedding-a", "embedding-b"]
SORTED_IDS = sorted(EMBEDDING_IDS)
PCA_CONFIG = {"n_components": 2, "solver": "full", "standardize": True}
CLUSTERING_CONFIG = {"n_clusters": 2, "random_state": 7}
SUMMARY_CONFIG = {
    "model": "openai/model",
    "prompt_version": "cluster-summary-v1",
    "reasoning": {"effort": "low"},
    "provider": {"order": ["openai"]},
    "max_tokens": 500,
}


def pca_output():
    return {
        "coordinates": [
            {"embedding_id": "embedding-a", "values": [0.1, 0.2]},
            {"embedding_id": "embedding-b", "values": [0.3, 0.4]},
            {"embedding_id": "embedding-c", "values": [0.5, 0.6]},
        ],
        "n_samples": 3,
        "n_input_dimensions": 8,
        "n_components": 2,
        "diagnostics": {
            "explained_variance_ratio": [0.7, 0.2],
            "solver": "full",
        },
    }


def clustering_output():
    return {
        "assignments": [
            {"embedding_id": "embedding-a", "cluster_id": 0},
            {"embedding_id": "embedding-b", "cluster_id": 1},
            {"embedding_id": "embedding-c", "cluster_id": 0},
        ],
        "centroids": [
            {"cluster_id": 0, "values": [0.2, 0.4]},
            {"cluster_id": 1, "values": [0.3, 0.4]},
        ],
        "n_clusters": 2,
        "n_features": 2,
        "diagnostics": {"inertia": 1.25, "n_iter": 4},
    }


def summary_response(cluster_id=0):
    return {
        "summary": f"Summary for cluster {cluster_id}",
        "response_id": f"response-{cluster_id}",
        "resolved_model": "openai/resolved-model",
        "finish_reason": "stop",
        "native_finish_reason": "stop",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "total_tokens": 14,
            "cost": 0.02,
            "completion_tokens_details": {"reasoning_tokens": 2},
            "prompt_tokens_details": {"cached_tokens": 1},
            "cost_details": {"upstream_inference_cost": 0.01},
        },
    }


def completed_pca(embedding_ids=EMBEDDING_IDS):
    entity = build_pca_analysis(embedding_ids, PCA_CONFIG, timestamp=CREATED)
    entity = record_pca_started(entity, timestamp=STARTED)
    output = pca_output()
    if sorted(embedding_ids) != SORTED_IDS:
        output["coordinates"] = [
            {"embedding_id": embedding_id, "values": [index / 10, index / 5]}
            for index, embedding_id in enumerate(sorted(embedding_ids), start=1)
        ]
        output["n_samples"] = len(embedding_ids)
    return record_pca_completed(entity, output, timestamp=FINISHED)


def completed_clustering(feature_source=None, pca_entity=None):
    source = feature_source or {"kind": "embeddings"}
    entity = build_kmeans_analysis(
        EMBEDDING_IDS,
        source,
        CLUSTERING_CONFIG,
        timestamp=CREATED,
    )
    entity = record_clustering_started(entity, timestamp=STARTED)
    return record_clustering_completed(
        entity,
        clustering_output(),
        pca_analysis=pca_entity,
        timestamp=FINISHED,
    )


class SummaryFailure(RuntimeError):
    code = "provider_failure"
    status_code = 429
    response_id = "failed-response"
    resolved_model = "openai/resolved-model"
    finish_reason = "error"
    native_finish_reason = "rate_limit"
    usage = {
        "prompt_tokens": 10,
        "completion_tokens": 0,
        "total_tokens": 10,
        "cost": 0.005,
    }


class DerivedAnalysisEntityTests(unittest.TestCase):
    def test_embedding_set_and_config_identity_are_canonical_and_secret_safe(self):
        first = build_pca_analysis(EMBEDDING_IDS, PCA_CONFIG, timestamp=CREATED)
        second = build_pca_analysis(
            list(reversed(EMBEDDING_IDS)),
            {"standardize": True, "solver": "full", "n_components": 2},
            timestamp=CREATED,
        )
        self.assertEqual(SORTED_IDS, normalize_embedding_ids(EMBEDDING_IDS))
        self.assertEqual(first, second)
        self.assertEqual(
            first["_id"], pca_analysis_id(SORTED_IDS, first["pca_config"])
        )
        with self.assertRaises(DerivedAnalysisConfigurationError):
            build_pca_analysis(["a", "a"], PCA_CONFIG)
        with self.assertRaises(DerivedAnalysisConfigurationError) as raised:
            build_pca_analysis(
                ["a"], {"api_key": "private-value", "n_components": 1}
            )
        self.assertNotIn("private-value", str(raised.exception))
        with self.assertRaises(DerivedAnalysisConfigurationError):
            build_pca_analysis(["a"], {"token": "private-value"})
        with self.assertRaises(DerivedAnalysisConfigurationError):
            build_pca_analysis(["a"], {"value": math.nan})
        with self.assertRaisesRegex(ValueError, "prompts or source text"):
            summary_config_hash(
                {**SUMMARY_CONFIG, "generation": {"prompt": "do not store me"}}
            )

    def test_pca_failure_retry_completion_and_output_validation(self):
        entity = build_pca_analysis(EMBEDDING_IDS, PCA_CONFIG, timestamp=CREATED)
        started = record_pca_started(entity, timestamp=STARTED)
        failed = record_pca_failed(
            started,
            RuntimeError("secret raw reasoning"),
            timestamp=FINISHED,
        )
        self.assertEqual("failed", failed["status"])
        self.assertNotIn("secret", json.dumps(failed))
        self.assertIsNone(failed["output"])

        retried = record_pca_started(failed, timestamp=RETRIED)
        completed = record_pca_completed(
            retried, pca_output(), timestamp=RETRY_FINISHED
        )
        self.assertEqual("complete", completed["status"])
        self.assertEqual(2, completed["attempt_count"])
        self.assertEqual(SORTED_IDS, completed["embedding_ids"])
        self.assertEqual(2, completed["output"]["n_components"])
        validate_pca_analysis(completed)
        with self.assertRaisesRegex(ValueError, "complete"):
            record_pca_started(completed)

        malformed = pca_output()
        malformed["coordinates"][0]["embedding_id"] = "wrong"
        with self.assertRaisesRegex(ValueError, "order"):
            record_pca_completed(retried, malformed)

    def test_raw_and_pca_feature_sources_produce_distinct_identities(self):
        pca_entity = completed_pca()
        raw_id = kmeans_analysis_id(
            EMBEDDING_IDS, {"kind": "embeddings"}, CLUSTERING_CONFIG
        )
        pca_source = {"kind": "pca", "pca_analysis_id": pca_entity["_id"]}
        pca_id = kmeans_analysis_id(
            EMBEDDING_IDS, pca_source, CLUSTERING_CONFIG
        )
        component_variant = kmeans_analysis_id(
            EMBEDDING_IDS,
            pca_source,
            {**CLUSTERING_CONFIG, "pca_component_indices": [0]},
        )
        self.assertNotEqual(raw_id, pca_id)
        self.assertNotEqual(pca_id, component_variant)

        pending = build_kmeans_analysis(
            EMBEDDING_IDS, pca_source, CLUSTERING_CONFIG, timestamp=CREATED
        )
        pending = record_clustering_started(pending, timestamp=STARTED)
        with self.assertRaisesRegex(ValueError, "requires a PCA"):
            record_clustering_completed(pending, clustering_output())

        mismatched_pca = completed_pca(["embedding-a", "embedding-b"])
        with self.assertRaisesRegex(ValueError, "id does not match"):
            record_clustering_completed(
                pending,
                clustering_output(),
                pca_analysis=mismatched_pca,
            )

        completed = record_clustering_completed(
            pending,
            clustering_output(),
            pca_analysis=pca_entity,
            timestamp=FINISHED,
        )
        self.assertEqual("complete", completed["clustering"]["status"])
        validate_kmeans_analysis(completed)

    def test_summary_runs_resume_failed_clusters_and_preserve_usage(self):
        entity = completed_clustering()
        entity = add_summary_run(entity, SUMMARY_CONFIG, timestamp=FINISHED)
        summary_hash = summary_config_hash(SUMMARY_CONFIG)
        self.assertEqual(1, len(entity["summaries"]))

        entity = record_cluster_summary_started(
            entity,
            summary_hash,
            0,
            input_hash="input-hash-0",
            prompt="Exact prompt zero",
            timestamp=RETRIED,
        )
        entity = record_cluster_summary_completed(
            entity,
            summary_hash,
            0,
            summary_response(0),
            timestamp=RETRY_FINISHED,
        )
        with self.assertRaisesRegex(ValueError, "complete"):
            record_cluster_summary_started(
                entity,
                summary_hash,
                0,
                input_hash="input-hash-0",
                prompt="Exact prompt zero",
            )

        entity = record_cluster_summary_started(
            entity,
            summary_hash,
            1,
            input_hash="input-hash-1",
            prompt="Exact prompt one",
        )
        failed = record_cluster_summary_failed(
            entity, summary_hash, 1, SummaryFailure("private content")
        )
        summary_run = failed["summaries"][0]
        self.assertEqual("failed", summary_run["status"])
        cluster_zero, cluster_one = summary_run["clusters"]
        self.assertEqual("complete", cluster_zero["status"])
        self.assertEqual(0.02, cluster_zero["output"]["usage"]["cost"])
        self.assertEqual(0.005, cluster_one["attempts"][0]["usage"]["cost"])
        self.assertNotIn("private content", json.dumps(failed))

        retried = record_cluster_summary_started(
            failed,
            summary_hash,
            1,
            input_hash="input-hash-1",
            prompt="Exact prompt one",
        )
        completed = record_cluster_summary_completed(
            retried, summary_hash, 1, summary_response(1)
        )
        self.assertEqual("complete", completed["summaries"][0]["status"])
        self.assertEqual(2, completed["summaries"][0]["clusters"][1]["attempt_count"])
        self.assertEqual(
            rendered_prompt_hash("Exact prompt one"),
            completed["summaries"][0]["clusters"][1]["prompt_hash"],
        )
        self.assertEqual(
            0.005,
            completed["summaries"][0]["clusters"][1]["attempts"][0]["usage"][
                "cost"
            ],
        )

        alternative = {
            **SUMMARY_CONFIG,
            "model": "anthropic/other-model",
        }
        with_second_config = add_summary_run(completed, alternative)
        self.assertEqual(2, len(with_second_config["summaries"]))
        validate_kmeans_analysis(with_second_config)

    def test_missing_summary_cost_is_recorded_as_null(self):
        entity = completed_clustering()
        entity = add_summary_run(entity, SUMMARY_CONFIG, timestamp=FINISHED)
        summary_hash = summary_config_hash(SUMMARY_CONFIG)
        entity = record_cluster_summary_started(
            entity,
            summary_hash,
            0,
            input_hash="input-hash-0",
            prompt="Exact prompt zero",
            timestamp=RETRIED,
        )
        response = summary_response(0)
        del response["usage"]["cost"]
        completed = record_cluster_summary_completed(
            entity,
            summary_hash,
            0,
            response,
            timestamp=RETRY_FINISHED,
        )
        self.assertIsNone(
            completed["summaries"][0]["clusters"][0]["output"]["usage"]["cost"]
        )

    def test_changed_prompt_is_distinct_and_failed_history_is_preserved(self):
        entity = add_summary_run(
            completed_clustering(), SUMMARY_CONFIG, timestamp=FINISHED
        )
        summary_hash = summary_config_hash(SUMMARY_CONFIG)
        first = record_cluster_summary_started(
            entity,
            summary_hash,
            0,
            input_hash="input-hash-0",
            prompt="First exact prompt",
            timestamp=RETRIED,
        )
        failed = record_cluster_summary_failed(
            first,
            summary_hash,
            0,
            SummaryFailure("private content"),
            timestamp=RETRY_FINISHED,
        )
        retried = record_cluster_summary_started(
            failed,
            summary_hash,
            0,
            input_hash="input-hash-0-revised",
            prompt="Second exact prompt",
        )
        cluster = retried["summaries"][0]["clusters"][0]
        self.assertEqual(2, cluster["attempt_count"])
        self.assertEqual("First exact prompt", cluster["attempts"][0]["prompt"])
        self.assertEqual("Second exact prompt", cluster["attempts"][1]["prompt"])
        self.assertNotEqual(
            cluster["attempts"][0]["prompt_hash"],
            cluster["attempts"][1]["prompt_hash"],
        )
        self.assertEqual(
            0.005,
            cluster["attempts"][0]["usage"]["cost"],
        )

    def test_legacy_summary_is_readable_but_not_exact_prompt_verified(self):
        entity = add_summary_run(
            completed_clustering(), SUMMARY_CONFIG, timestamp=FINISHED
        )
        summary_hash = summary_config_hash(SUMMARY_CONFIG)
        entity = record_cluster_summary_started(
            entity,
            summary_hash,
            0,
            input_hash="legacy-input",
            prompt="Prompt absent from legacy storage",
            timestamp=RETRIED,
        )
        entity = record_cluster_summary_completed(
            entity,
            summary_hash,
            0,
            summary_response(0),
            timestamp=RETRY_FINISHED,
        )
        legacy = copy.deepcopy(entity)
        legacy["schema_version"] = 1
        for summary_run in legacy["summaries"]:
            for cluster in summary_run["clusters"]:
                cluster.pop("prompt")
                cluster.pop("exact_prompt_verified")
                cluster.pop("reuse")
                for attempt in cluster["attempts"]:
                    attempt.pop("input_hash")
                    attempt.pop("prompt")
                    attempt.pop("prompt_hash")
                    attempt.pop("exact_prompt_verified")

        validate_kmeans_analysis(legacy)
        upgraded = upgrade_kmeans_analysis(legacy)
        upgraded_cluster = upgraded["summaries"][0]["clusters"][0]
        self.assertEqual(2, upgraded["schema_version"])
        self.assertIsNone(upgraded_cluster["prompt"])
        self.assertFalse(upgraded_cluster["exact_prompt_verified"])
        self.assertEqual(1, upgraded_cluster["attempt_count"])
        self.assertEqual(
            0.02,
            upgraded_cluster["attempts"][0]["usage"]["cost"],
        )

    def test_validation_rejects_prompt_hash_not_matching_exact_bytes(self):
        entity = add_summary_run(
            completed_clustering(), SUMMARY_CONFIG, timestamp=FINISHED
        )
        summary_hash = summary_config_hash(SUMMARY_CONFIG)
        entity = record_cluster_summary_started(
            entity,
            summary_hash,
            0,
            input_hash="input-hash",
            prompt="Exact UTF-8 prompt: 合作",
            timestamp=RETRIED,
        )
        tampered = copy.deepcopy(entity)
        tampered["summaries"][0]["clusters"][0]["prompt_hash"] = "wrong"
        with self.assertRaisesRegex(ValueError, "exact UTF-8 bytes"):
            validate_kmeans_analysis(tampered)

    def test_validation_rejects_tampered_derived_entities(self):
        pca_entity = completed_pca()
        tampered_pca = copy.deepcopy(pca_entity)
        tampered_pca["pca_config_hash"] = "wrong"
        with self.assertRaises(ValueError):
            validate_pca_analysis(tampered_pca)

        kmeans_entity = completed_clustering()
        tampered_kmeans = copy.deepcopy(kmeans_entity)
        tampered_kmeans["clustering"]["output"]["centroids"][0]["values"] = [
            float("inf"),
            0,
        ]
        with self.assertRaises(ValueError):
            validate_kmeans_analysis(tampered_kmeans)


if __name__ == "__main__":
    unittest.main()
