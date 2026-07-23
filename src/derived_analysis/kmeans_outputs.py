"""Typed output validation for stored k-means and summary results."""

import math
from copy import deepcopy

from derived_analysis.common import (
    normalize_embedding_ids,
    normalize_json_value,
    validate_nonnegative_int,
)


def _finite_vector(values, field: str) -> list[float | int]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{field} must be a nonempty list.")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in values
    ):
        raise ValueError(f"{field} must contain finite numbers.")
    return deepcopy(values)


def normalize_clustering_output(output: dict, embedding_ids) -> dict:
    required = {
        "assignments",
        "centroids",
        "n_clusters",
        "n_features",
        "diagnostics",
    }
    if not isinstance(output, dict) or set(output) != required:
        raise ValueError("Clustering output has an unexpected schema.")
    normalized_ids = normalize_embedding_ids(embedding_ids)
    assignments = output["assignments"]
    if not isinstance(assignments, list) or len(assignments) != len(normalized_ids):
        raise ValueError("Clustering assignments must align with embedding_ids.")
    normalized_assignments = []
    assigned_clusters = set()
    for expected_id, assignment in zip(normalized_ids, assignments):
        if not isinstance(assignment, dict) or set(assignment) != {
            "embedding_id",
            "cluster_id",
        }:
            raise ValueError("Clustering assignment has an unexpected schema.")
        if assignment["embedding_id"] != expected_id:
            raise ValueError("Assignment order must match embedding_ids.")
        validate_nonnegative_int(assignment["cluster_id"], "cluster_id")
        assigned_clusters.add(assignment["cluster_id"])
        normalized_assignments.append(deepcopy(assignment))

    centroids = output["centroids"]
    if not isinstance(centroids, list) or not centroids:
        raise ValueError("Clustering centroids must be nonempty.")
    normalized_centroids = []
    centroid_ids = []
    feature_count = None
    for centroid in centroids:
        if not isinstance(centroid, dict) or set(centroid) != {
            "cluster_id",
            "values",
        }:
            raise ValueError("Clustering centroid has an unexpected schema.")
        validate_nonnegative_int(centroid["cluster_id"], "centroid cluster_id")
        values = _finite_vector(centroid["values"], "centroid values")
        if feature_count is None:
            feature_count = len(values)
        elif len(values) != feature_count:
            raise ValueError("Centroid dimensions must be consistent.")
        centroid_ids.append(centroid["cluster_id"])
        normalized_centroids.append(
            {"cluster_id": centroid["cluster_id"], "values": values}
        )
    if centroid_ids != sorted(set(centroid_ids)):
        raise ValueError("Centroids must have unique sorted cluster ids.")
    if set(centroid_ids) != assigned_clusters:
        raise ValueError("Centroids must match assigned cluster ids.")

    for field in ("n_clusters", "n_features"):
        validate_nonnegative_int(output[field], f"clustering output {field}")
        if output[field] < 1:
            raise ValueError(f"clustering output {field} must be positive.")
    if output["n_clusters"] != len(centroid_ids):
        raise ValueError("Clustering n_clusters is inconsistent.")
    if output["n_features"] != feature_count:
        raise ValueError("Clustering n_features is inconsistent.")
    diagnostics = normalize_json_value(
        output["diagnostics"], "clustering diagnostics"
    )
    if not isinstance(diagnostics, dict):
        raise ValueError("Clustering diagnostics must be a dictionary.")
    return {
        "assignments": normalized_assignments,
        "centroids": normalized_centroids,
        "n_clusters": output["n_clusters"],
        "n_features": output["n_features"],
        "diagnostics": diagnostics,
    }


def normalize_summary_usage(usage: dict) -> dict:
    normalized = normalize_json_value(usage, "summary usage")
    if not isinstance(normalized, dict):
        raise ValueError("Summary usage must be a dictionary.")
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        validate_nonnegative_int(normalized.get(field), f"summary usage {field}")
    cost = normalized.setdefault("cost", None)
    if cost is not None and (
        isinstance(cost, bool)
        or not isinstance(cost, (int, float))
        or not math.isfinite(cost)
        or cost < 0
    ):
        raise ValueError("Summary usage cost must be a nonnegative number or null.")
    return normalized
