"""Selected reasoning-embedding outputs for the canonical paper."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from db_ops.kmeans_analyses import find_kmeans_analysis
from db_ops.pca_analyses import find_pca_analysis
from derived_analysis.kmeans import (
    build_kmeans_analysis,
    rendered_prompt_hash,
    summary_config_hash,
    validate_kmeans_analysis,
)
from derived_analysis.pca import build_pca_analysis, validate_pca_analysis
from embedding.config import embedding_entity_id, normalize_embedding_config
from embedding.entities import validate_embedding_entity
from games.instructions import GAME_DESCRIPTION

from .environment import FIGS_DIR, MAIN_ANALYSIS_OUTPUT_DIR, TABLES_DIR


EMBEDDING_CONFIG = {
    "model": "qwen/qwen3-embedding-8b",
    "encoding_format": "float",
    "provider": {
        "only": ["deepinfra"],
        "allow_fallbacks": False,
        "data_collection": "deny",
    },
}

PCA_CONFIG = {
    "implementation": "sklearn.decomposition.PCA",
    "fit_scope": "within_game",
    "preprocessing": {
        "row_l2_normalize": True,
        "mean_center": True,
        "feature_standardize": False,
    },
    "n_components": {
        "strategy": "minimum_cumulative_explained_variance",
        "threshold": 0.9,
    },
    "svd_solver": "full",
    "whiten": False,
}

KMEANS_CONFIG = {
    "candidate_k": list(range(2, 11)),
    "init": "k-means++",
    "n_init": 50,
    "max_iter": 500,
    "tol": 1e-4,
    "algorithm": "lloyd",
    "random_state": 20260725,
    "selection": {
        "metric": "silhouette",
        "within_maximum_tolerance": 0.01,
        "tie_break": "smallest_k",
    },
}

SUMMARY_CONFIG = {
    "model": "anthropic/claude-sonnet-5",
    "reasoning": {"effort": "high"},
    "provider": {
        "allow_fallbacks": False,
        "data_collection": "deny",
    },
    "prompt_version": "reasoning-cluster-summary-v3",
    "generation": {
        "temperature": 0,
        "max_tokens": 300,
    },
}

SUMMARY_REQUEST_INSTRUCTION = (
    "Summarize the shared reasoning across these responses in one clear "
    "paragraph of fewer than 50 words. Focus on the rationale: what the "
    "responses are trying to achieve, the trade-offs they consider, and what "
    "distinguishes this group. Use only the reasoning content. Do not add a "
    "heading or bullets."
)

GAME_ORDER = ("Dictator", "Prisoner's Dilemma")
MODEL_ORDER = (
    "DeepSeek V3 thinking",
    "Gemini 3.1 Pro thinking",
    "GPT-5.2 thinking",
)
MODE_ORDER = ("Atomic", "ChunkN=10")
MODEL_COLORS = {
    "DeepSeek V3 thinking": "#1B9E77",
    "Gemini 3.1 Pro thinking": "#386CB0",
    "GPT-5.2 thinking": "#D95F02",
}
MODE_LINE_STYLES = {"Atomic": "-", "ChunkN=10": "--"}


@dataclass(frozen=True)
class SimulationSpec:
    simulation_id: str
    game: str
    model_label: str
    model_id: str
    mode: str
    chunk_n: int
    expected_sessions: int
    expected_failed_sessions: int = 0


SIMULATION_SPECS = (
    SimulationSpec(
        "a43fb348-4da5-42c8-b5a2-cec5fdab892f",
        "Dictator",
        "DeepSeek V3 thinking",
        "deepseek/deepseek-v3.2",
        "Atomic",
        1,
        100,
        1,
    ),
    SimulationSpec(
        "6e9fd4fb-e60d-4070-ac68-0b21f65ef458",
        "Dictator",
        "DeepSeek V3 thinking",
        "deepseek/deepseek-v3.2",
        "ChunkN=10",
        10,
        10,
    ),
    SimulationSpec(
        "0dbbca0f-deec-4786-96c2-172e4e89c4aa",
        "Dictator",
        "Gemini 3.1 Pro thinking",
        "google/gemini-3.1-pro-preview",
        "Atomic",
        1,
        100,
    ),
    SimulationSpec(
        "f40ec6c9-f5a0-4d7b-9428-4925779ad075",
        "Dictator",
        "Gemini 3.1 Pro thinking",
        "google/gemini-3.1-pro-preview",
        "ChunkN=10",
        10,
        10,
    ),
    SimulationSpec(
        "0f5891ab-d243-4436-8a14-9883cae2071e",
        "Dictator",
        "GPT-5.2 thinking",
        "openai/gpt-5.2",
        "Atomic",
        1,
        100,
    ),
    SimulationSpec(
        "668da5f0-e691-4fd7-a954-ade87651682b",
        "Dictator",
        "GPT-5.2 thinking",
        "openai/gpt-5.2",
        "ChunkN=10",
        10,
        10,
    ),
    SimulationSpec(
        "3de77348-6b2f-4824-a34e-2776640ddf55",
        "Prisoner's Dilemma",
        "DeepSeek V3 thinking",
        "deepseek/deepseek-v3.2",
        "Atomic",
        1,
        100,
    ),
    SimulationSpec(
        "9926dd93-3db2-4f31-a479-ffb2a3427e69",
        "Prisoner's Dilemma",
        "DeepSeek V3 thinking",
        "deepseek/deepseek-v3.2",
        "ChunkN=10",
        10,
        10,
    ),
    SimulationSpec(
        "9e536bff-36d1-4318-95b0-793f175fa07c",
        "Prisoner's Dilemma",
        "Gemini 3.1 Pro thinking",
        "google/gemini-3.1-pro-preview",
        "Atomic",
        1,
        100,
    ),
    SimulationSpec(
        "24803e98-3de5-4c9e-8d96-34207d2fc641",
        "Prisoner's Dilemma",
        "Gemini 3.1 Pro thinking",
        "google/gemini-3.1-pro-preview",
        "ChunkN=10",
        10,
        10,
    ),
    SimulationSpec(
        "7b4f1aaa-ed7c-478a-8054-772ca5f368a7",
        "Prisoner's Dilemma",
        "GPT-5.2 thinking",
        "openai/gpt-5.2",
        "Atomic",
        1,
        100,
    ),
    SimulationSpec(
        "95ba4114-2a08-4ba1-90a6-b437ebf2b784",
        "Prisoner's Dilemma",
        "GPT-5.2 thinking",
        "openai/gpt-5.2",
        "ChunkN=10",
        10,
        10,
    ),
)


def _validate_simulation(spec: SimulationSpec, simulation: dict) -> None:
    simulation_config = simulation.get("simulation_config") or {}
    instruction_config = simulation.get("instruction_config") or {}
    llm_config = simulation.get("llm_config") or {}
    session_ids = simulation.get("simulation_sessions")
    failed_ids = simulation.get("failed_sessions")
    extra_flag = simulation.get("extraFlag")
    if extra_flag not in (None, [], ()):
        raise ValueError(f"{spec.simulation_id}: extraFlag is not empty.")
    checks = {
        "phase": simulation.get("phase_name") == "phase_2",
        "completed": simulation.get("completed") is True,
        "archived": simulation.get("archived") is False,
        "game": simulation_config.get("game_type") == spec.game,
        "chunk_n": simulation_config.get("batch_simulation_n") == spec.chunk_n,
        "reasoning": instruction_config.get("explain_reasoning") is True,
        "reasoning_enabled": llm_config.get("reasoning_enabled") is True,
        "model": llm_config.get("model") == spec.model_id,
        "sessions": isinstance(session_ids, list)
        and len(session_ids) == spec.expected_sessions
        and len(session_ids) == len(set(session_ids)),
        "failures": isinstance(failed_ids, list)
        and len(failed_ids) == spec.expected_failed_sessions,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            f"{spec.simulation_id}: corpus checks failed: {', '.join(failed)}."
        )


def _simulation_observations(
    spec: SimulationSpec,
    simulation: dict,
    sessions_by_id: dict[str, dict],
) -> list[dict]:
    session_ids = simulation["simulation_sessions"]
    if not set(session_ids).issubset(sessions_by_id):
        raise ValueError(f"{spec.simulation_id}: referenced sessions are missing.")
    observations = []
    for session_id in sorted(session_ids):
        session = sessions_by_id[session_id]
        if (
            session.get("simulation_id") != spec.simulation_id
            or session.get("agent_response_success") is not True
            or session.get("schema_check_pass") is not True
        ):
            raise ValueError(f"{session_id}: session provenance is invalid.")
        decisions = session.get("decisions")
        if not isinstance(decisions, list) or not decisions:
            raise ValueError(f"{session_id}: decisions are missing.")
        for decision_index, decision in enumerate(decisions):
            if (
                not isinstance(decision, (list, tuple))
                or len(decision) != 2
                or not isinstance(decision[1], str)
                or not decision[1].strip()
            ):
                raise ValueError(f"{session_id}: reasoning is invalid.")
            observations.append(
                {
                    "game": spec.game,
                    "model_label": spec.model_label,
                    "mode": spec.mode,
                    "simulation_id": spec.simulation_id,
                    "simulation_session_id": session_id,
                    "decision_index": decision_index,
                    "reasoning": decision[1],
                }
            )
    if len(observations) != 100:
        raise ValueError(f"{spec.simulation_id}: expected 100 reasonings.")
    return observations


def _load_corpus(db) -> tuple[pd.DataFrame, np.ndarray]:
    normalized_config = normalize_embedding_config(EMBEDDING_CONFIG)
    simulation_ids = [spec.simulation_id for spec in SIMULATION_SPECS]
    simulations = list(db["simulations"].find({"_id": {"$in": simulation_ids}}))
    simulations_by_id = {simulation["_id"]: simulation for simulation in simulations}
    if set(simulations_by_id) != set(simulation_ids):
        raise ValueError("One or more canonical reasoning simulations are missing.")
    for spec in SIMULATION_SPECS:
        _validate_simulation(spec, simulations_by_id[spec.simulation_id])
    session_ids = [
        session_id
        for spec in SIMULATION_SPECS
        for session_id in simulations_by_id[spec.simulation_id]["simulation_sessions"]
    ]
    sessions = list(db["simulation_sessions"].find({"_id": {"$in": session_ids}}))
    sessions_by_id = {session["_id"]: session for session in sessions}
    if set(sessions_by_id) != set(session_ids):
        raise ValueError("One or more canonical reasoning sessions are missing.")
    observations = [
        observation
        for spec in SIMULATION_SPECS
        for observation in _simulation_observations(
            spec,
            simulations_by_id[spec.simulation_id],
            sessions_by_id,
        )
    ]
    embedding_ids = [
        embedding_entity_id(
            observation["simulation_session_id"],
            observation["decision_index"],
            normalized_config,
        )
        for observation in observations
    ]
    entities = list(db["embeddings"].find({"_id": {"$in": embedding_ids}}))
    entities_by_id = {entity["_id"]: entity for entity in entities}
    if set(entities_by_id) != set(embedding_ids):
        raise ValueError("One or more canonical reasoning embeddings are missing.")
    rows = []
    vectors = []
    for observation, embedding_id in zip(observations, embedding_ids):
        entity = entities_by_id[embedding_id]
        if entity.get("success") is not True:
            raise ValueError(f"{embedding_id}: successful embedding is missing.")
        validate_embedding_entity(entity)
        vector = (entity.get("output") or {}).get("vector")
        if (
            entity.get("embedding_config") != normalized_config
            or entity.get("input_text") != observation["reasoning"]
            or not isinstance(vector, list)
            or len(vector) != 4096
            or entity["output"].get("vector_dimension") != 4096
        ):
            raise ValueError(f"{embedding_id}: embedding provenance is invalid.")
        rows.append({**observation, "embedding_id": embedding_id})
        vectors.append(vector)
    order = sorted(range(len(rows)), key=lambda index: rows[index]["embedding_id"])
    frame = pd.DataFrame([rows[index] for index in order])
    matrix = np.asarray([vectors[index] for index in order], dtype=float)
    if frame.shape[0] != 1200 or matrix.shape != (1200, 4096):
        raise ValueError("Canonical reasoning corpus must be 1,200 by 4,096.")
    if not np.isfinite(matrix).all() or frame["embedding_id"].duplicated().any():
        raise ValueError("Canonical reasoning embeddings are invalid or duplicated.")
    cell_counts = frame.groupby(["game", "model_label", "mode"]).size()
    if len(cell_counts) != 12 or set(cell_counts) != {100}:
        raise ValueError("Canonical reasoning corpus must contain 12 balanced cells.")
    return frame, matrix


def _clustering_config(component_count: int) -> dict:
    return {
        **KMEANS_CONFIG,
        "feature_space": "retained_pca_90pct",
        "pca_component_indices": list(range(component_count)),
    }


def _render_cluster_prompt(game: str, cluster_id: int, frame: pd.DataFrame) -> str:
    reasonings = sorted(frame.loc[frame["cluster_id"].eq(cluster_id), "reasoning"])
    if not reasonings:
        raise ValueError(f"{game} cluster {cluster_id} is empty.")
    blocks = [
        f"--- Reasoning {position:04d} ---\n{reasoning}"
        for position, reasoning in enumerate(reasonings, start=1)
    ]
    return "\n\n".join(
        [
            f"Reasoning cluster summary request: {SUMMARY_CONFIG['prompt_version']}",
            f"Game: {game}",
            f"Full game instruction/setting:\n{GAME_DESCRIPTION[game]}",
            "Anonymized reasoning responses:\n\n" + "\n\n".join(blocks),
            SUMMARY_REQUEST_INSTRUCTION,
        ]
    )


def _summary_rows(game: str, frame: pd.DataFrame, kmeans_entity: dict) -> list[dict]:
    config_hash = summary_config_hash(SUMMARY_CONFIG)
    matching_runs = [
        run
        for run in kmeans_entity.get("summaries", [])
        if run.get("summary_config_hash") == config_hash
        and run.get("summary_config") == SUMMARY_CONFIG
    ]
    if len(matching_runs) != 1:
        raise ValueError(f"{game}: active-v3 summary run is missing or duplicated.")
    clusters = {
        cluster["cluster_id"]: cluster for cluster in matching_runs[0]["clusters"]
    }
    if set(clusters) != {0, 1}:
        raise ValueError(f"{game}: K=2 summaries are incomplete.")
    rows = []
    for cluster_id in (0, 1):
        cluster = clusters[cluster_id]
        prompt = _render_cluster_prompt(game, cluster_id, frame)
        summary = (cluster.get("output") or {}).get("summary")
        if (
            cluster.get("status") != "complete"
            or cluster.get("exact_prompt_verified") is not True
            or cluster.get("prompt") != prompt
            or cluster.get("prompt_hash") != rendered_prompt_hash(prompt)
            or not isinstance(summary, str)
            or not summary.strip()
        ):
            raise ValueError(f"{game} cluster {cluster_id}: summary is invalid.")
        rows.append(
            {
                "game": game,
                "cluster": cluster_id,
                "observations": int(frame["cluster_id"].eq(cluster_id).sum()),
                "summary": summary,
            }
        )
    return rows


def _load_game_evidence(
    db,
    game: str,
    frame: pd.DataFrame,
    matrix: np.ndarray,
) -> dict:
    mask = frame["game"].eq(game).to_numpy()
    game_frame = frame.loc[mask].reset_index(drop=True)
    game_matrix = matrix[mask]
    embedding_ids = game_frame["embedding_id"].tolist()
    pca_identity = build_pca_analysis(embedding_ids, PCA_CONFIG)
    pca_entity = find_pca_analysis(db, pca_identity["_id"])
    if pca_entity is None:
        raise ValueError(f"{game}: exact PCA entity is missing.")
    validate_pca_analysis(pca_entity)
    pca_output = pca_entity.get("output") or {}
    coordinate_ids = [item["embedding_id"] for item in pca_output.get("coordinates", [])]
    if (
        pca_entity.get("status") != "complete"
        or pca_entity.get("pca_config") != PCA_CONFIG
        or pca_entity.get("embedding_ids") != embedding_ids
        or pca_output.get("n_samples") != 600
        or pca_output.get("n_input_dimensions") != 4096
        or pca_output.get("n_components") != 103
        or coordinate_ids != embedding_ids
    ):
        raise ValueError(f"{game}: exact PCA provenance is invalid.")

    clustering_config = _clustering_config(pca_output["n_components"])
    kmeans_identity = build_kmeans_analysis(
        embedding_ids,
        {"kind": "pca", "pca_analysis_id": pca_entity["_id"]},
        clustering_config,
    )
    kmeans_entity = find_kmeans_analysis(db, kmeans_identity["_id"])
    if kmeans_entity is None:
        raise ValueError(f"{game}: exact K-means entity is missing.")
    validate_kmeans_analysis(kmeans_entity)
    clustering = kmeans_entity.get("clustering") or {}
    output = clustering.get("output") or {}
    diagnostics = output.get("diagnostics") or {}
    assignments = output.get("assignments") or []
    assignment_map = {item["embedding_id"]: item["cluster_id"] for item in assignments}
    candidates = diagnostics.get("candidates") or []
    if (
        kmeans_entity.get("feature_source")
        != {"kind": "pca", "pca_analysis_id": pca_entity["_id"]}
        or kmeans_entity.get("clustering_config") != clustering_config
        or kmeans_entity.get("embedding_ids") != embedding_ids
        or clustering.get("status") != "complete"
        or output.get("n_clusters") != 2
        or output.get("n_features") != 103
        or diagnostics.get("selected_k") != 2
        or [item.get("k") for item in candidates] != list(range(2, 11))
        or set(assignment_map) != set(embedding_ids)
    ):
        raise ValueError(f"{game}: exact K-means provenance is invalid.")
    game_frame["cluster_id"] = game_frame["embedding_id"].map(assignment_map)
    summaries = _summary_rows(game, game_frame, kmeans_entity)
    return {
        "frame": game_frame,
        "matrix": game_matrix,
        "pca": pca_entity,
        "kmeans": kmeans_entity,
        "summaries": summaries,
    }


def _cosine_rows(evidence: dict) -> tuple[list[dict], pd.DataFrame]:
    frame = evidence["frame"]
    matrix = evidence["matrix"]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Canonical reasoning embeddings include a zero vector.")
    normalized = matrix / norms
    rows = []
    pair_frames = []
    for simulation_id, simulation in frame.groupby("simulation_id", sort=True):
        positions = frame.index.get_indexer(simulation.index)
        similarities = normalized[positions] @ normalized[positions].T
        values = np.clip(similarities[np.triu_indices(100, k=1)], -1.0, 1.0)
        if len(simulation) != 100 or len(values) != 4_950:
            raise ValueError(f"{simulation_id}: cosine-pair coverage is invalid.")
        model_label = simulation["model_label"].iloc[0]
        mode = simulation["mode"].iloc[0]
        rows.append(
            {
                "game": simulation["game"].iloc[0],
                "model_label": model_label,
                "mode": mode,
                "observations": 100,
                "unique_pairs": 4_950,
                "mean": float(np.mean(values)),
                "standard_deviation": float(np.std(values, ddof=0)),
            }
        )
        pair_frames.append(
            pd.DataFrame(
                {
                    "game": simulation["game"].iloc[0],
                    "model_label": model_label,
                    "mode": mode,
                    "cosine_similarity": values,
                }
            )
        )
    return rows, pd.concat(pair_frames, ignore_index=True)


def _paired_cosine_table(rows: list[dict]) -> pd.DataFrame:
    source = pd.DataFrame(rows)
    records = []
    for game in GAME_ORDER:
        for model_label in MODEL_ORDER:
            cells = source.loc[
                source["game"].eq(game) & source["model_label"].eq(model_label)
            ].set_index("mode")
            atomic = cells.loc["Atomic"]
            chunk = cells.loc["ChunkN=10"]
            records.append(
                {
                    "game": game,
                    "model": model_label.removesuffix(" thinking"),
                    "atomic_mean": atomic["mean"],
                    "atomic_sd": atomic["standard_deviation"],
                    "chunk_mean": chunk["mean"],
                    "chunk_sd": chunk["standard_deviation"],
                    "mean_difference": atomic["mean"] - chunk["mean"],
                }
            )
    return pd.DataFrame(records)


def _k_selection_table(game_evidence: dict[str, dict]) -> pd.DataFrame:
    records = []
    for game in GAME_ORDER:
        diagnostics = game_evidence[game]["kmeans"]["clustering"]["output"][
            "diagnostics"
        ]
        for candidate in diagnostics["candidates"]:
            records.append(
                {
                    "game": game,
                    "k": candidate["k"],
                    "selected": candidate["k"] == diagnostics["selected_k"],
                    "silhouette": candidate["silhouette"],
                    "inertia": candidate["inertia"],
                    "calinski_harabasz": candidate["calinski_harabasz"],
                    "davies_bouldin": candidate["davies_bouldin"],
                    "iterations": candidate["n_iter"],
                    "converged": candidate["converged"],
                    "size_range": (
                        f"{candidate['minimum_cluster_size']}--"
                        f"{candidate['maximum_cluster_size']}"
                    ),
                }
            )
    return pd.DataFrame(records)


def _composition_table(game_evidence: dict[str, dict]) -> pd.DataFrame:
    records = []
    for game in GAME_ORDER:
        frame = game_evidence[game]["frame"]
        for model_label in MODEL_ORDER:
            for mode in MODE_ORDER:
                cell = frame.loc[
                    frame["model_label"].eq(model_label) & frame["mode"].eq(mode)
                ]
                counts = cell["cluster_id"].value_counts()
                records.append(
                    {
                        "game": game,
                        "model": model_label.removesuffix(" thinking"),
                        "mode": mode,
                        "cluster_0": int(counts.get(0, 0)),
                        "cluster_1": int(counts.get(1, 0)),
                        "total": len(cell),
                    }
                )
    table = pd.DataFrame(records)
    if set(table["total"]) != {100}:
        raise ValueError("Every canonical composition row must total 100.")
    return table


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def _write_cosine_table(table: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{center}",
        r"\normalsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}llrrrrr@{}}",
        r"\toprule",
        r"Game & Model & Atomic M & Atomic SD & Chunk M & Chunk SD & $\Delta$ \\",
        r"\midrule",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            " & ".join(
                [
                    _latex_escape(row.game),
                    _latex_escape(row.model),
                    f"{row.atomic_mean:.3f}",
                    f"{row.atomic_sd:.3f}",
                    f"{row.chunk_mean:.3f}",
                    f"{row.chunk_sd:.3f}",
                    f"{row.mean_difference:.3f}",
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{center}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_k_selection_table(table: pd.DataFrame, path: Path) -> None:
    lines = [r"\begin{center}", r"\normalsize", r"\setlength{\tabcolsep}{3pt}"]
    for panel, game in zip(("A", "B"), GAME_ORDER):
        lines.extend(
            [
                rf"\textit{{Panel {panel}: {_latex_escape(game)}}}\\[2pt]",
                r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}rrrrrrrrr@{}}",
                r"\toprule",
                r"$K$ & Sel. & Sil. & Inertia & CH & DB & Iter. & Conv. & Size range \\",
                r"\midrule",
            ]
        )
        for row in table.loc[table["game"].eq(game)].itertuples(index=False):
            values = [
                str(row.k),
                r"\checkmark" if row.selected else "",
                f"{row.silhouette:.4f}",
                f"{row.inertia:.2f}",
                f"{row.calinski_harabasz:.2f}",
                f"{row.davies_bouldin:.3f}",
                str(row.iterations),
                "Yes" if row.converged else "No",
                row.size_range,
            ]
            if row.selected:
                values = [rf"\textbf{{{value}}}" if value else "" for value in values]
            lines.append(" & ".join(values) + r" \\")
        lines.extend([r"\bottomrule", r"\end{tabular*}"])
        if panel == "A":
            lines.append(r"\par\medskip")
    lines.extend([r"\end{center}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_composition_table(table: pd.DataFrame, path: Path) -> None:
    lines = [r"\begin{center}", r"\normalsize", r"\setlength{\tabcolsep}{3pt}"]
    for panel, game in zip(("A", "B"), GAME_ORDER):
        lines.extend(
            [
                rf"\textit{{Panel {panel}: {_latex_escape(game)}}}\\[2pt]",
                r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrr@{}}",
                r"\toprule",
                r"Model & Mode & C0 & C1 & Total \\",
                r"\midrule",
            ]
        )
        for row in table.loc[table["game"].eq(game)].itertuples(index=False):
            lines.append(
                " & ".join(
                    [
                        _latex_escape(row.model),
                        _latex_escape(row.mode),
                        str(row.cluster_0),
                        str(row.cluster_1),
                        str(row.total),
                    ]
                )
                + r" \\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular*}"])
        if panel == "A":
            lines.append(r"\par\medskip")
    lines.extend([r"\end{center}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_summary_table(table: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{center}",
        r"\normalsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lcrp{0.59\textwidth}@{}}",
        r"\toprule",
        r"Game & Cluster & N & LLM-powered summary \\",
        r"\midrule",
    ]
    rows = list(table.itertuples(index=False))
    for index, row in enumerate(rows):
        lines.append(
            " & ".join(
                [
                    _latex_escape(row.game),
                    str(row.cluster),
                    str(row.observations),
                    _latex_escape(row.summary),
                ]
            )
            + r" \\"
        )
        if index < len(rows) - 1:
            lines.append(r"\addlinespace[0.4em]")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{center}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_density_figure(pairs: pd.DataFrame, path: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), sharex=True, sharey=True)
    for axis, panel, game in zip(axes, ("A", "B"), GAME_ORDER):
        game_pairs = pairs.loc[pairs["game"].eq(game)]
        for model_label in MODEL_ORDER:
            for mode in MODE_ORDER:
                cell = game_pairs.loc[
                    game_pairs["model_label"].eq(model_label)
                    & game_pairs["mode"].eq(mode)
                ]
                sns.kdeplot(
                    data=cell,
                    x="cosine_similarity",
                    color=MODEL_COLORS[model_label],
                    linestyle=MODE_LINE_STYLES[mode],
                    linewidth=2.0,
                    bw_adjust=0.8,
                    clip=(-1.0, 1.0),
                    label=f"{model_label.removesuffix(' thinking')}, {mode}",
                    ax=axis,
                )
        axis.set_title(f"Panel {panel}: {game}")
        axis.set_xlabel("Cosine similarity")
        axis.set_xlim(0.2, 1.0)
        axis.legend().remove()
    axes[0].set_ylabel("Density")
    axes[1].set_ylabel("")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0.13, 1, 1))
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _write_manifest(
    game_evidence: dict[str, dict],
    cosine: pd.DataFrame,
    k_selection: pd.DataFrame,
    composition: pd.DataFrame,
    summaries: pd.DataFrame,
    path: Path,
) -> None:
    payload = {
        "ticket": "EEA-0011-02",
        "decision": "HD-0003",
        "simulation_count": len(SIMULATION_SPECS),
        "observation_count": 1200,
        "simulation_specs": [asdict(spec) for spec in SIMULATION_SPECS],
        "embedding_config": normalize_embedding_config(EMBEDDING_CONFIG),
        "embedding_dimension": 4096,
        "pca_config": PCA_CONFIG,
        "kmeans_config": KMEANS_CONFIG,
        "summary_config": SUMMARY_CONFIG,
        "games": {
            game: {
                "pca_analysis_id": game_evidence[game]["pca"]["_id"],
                "retained_components": game_evidence[game]["pca"]["output"][
                    "n_components"
                ],
                "kmeans_analysis_id": game_evidence[game]["kmeans"]["_id"],
                "selected_k": game_evidence[game]["kmeans"]["clustering"][
                    "output"
                ]["n_clusters"],
            }
            for game in GAME_ORDER
        },
        "tables": {
            "cosine": cosine.to_dict(orient="records"),
            "k_selection": k_selection.to_dict(orient="records"),
            "k2_composition": composition.to_dict(orient="records"),
            "k2_summaries": summaries.to_dict(orient="records"),
        },
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def generate_reasoning_outputs(db) -> list[str]:
    """Generate only the five reasoning artifacts selected in HD-0003."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    MAIN_ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame, matrix = _load_corpus(db)
    game_evidence = {
        game: _load_game_evidence(db, game, frame, matrix) for game in GAME_ORDER
    }
    cosine_rows = []
    pair_frames = []
    for game in GAME_ORDER:
        game_rows, game_pairs = _cosine_rows(game_evidence[game])
        cosine_rows.extend(game_rows)
        pair_frames.append(game_pairs)
    cosine = _paired_cosine_table(cosine_rows)
    k_selection = _k_selection_table(game_evidence)
    composition = _composition_table(game_evidence)
    summaries = pd.DataFrame(
        [
            row
            for game in GAME_ORDER
            for row in game_evidence[game]["summaries"]
        ]
    )
    pairs = pd.concat(pair_frames, ignore_index=True)

    _write_cosine_table(cosine, TABLES_DIR / "reasoning_cosine.tex")
    _write_k_selection_table(
        k_selection,
        TABLES_DIR / "reasoning_k_selection.tex",
    )
    _write_composition_table(
        composition,
        TABLES_DIR / "reasoning_k2_composition.tex",
    )
    _write_summary_table(
        summaries,
        TABLES_DIR / "reasoning_k2_summary.tex",
    )
    _write_density_figure(pairs, FIGS_DIR / "reasoning_cosine_density.png")
    _write_manifest(
        game_evidence,
        cosine,
        k_selection,
        composition,
        summaries,
        MAIN_ANALYSIS_OUTPUT_DIR / "reasoning_analysis_results.json",
    )
    return [
        "reasoning_cosine.tex",
        "reasoning_k_selection.tex",
        "reasoning_k2_composition.tex",
        "reasoning_k2_summary.tex",
        "reasoning_cosine_density.png",
        "reasoning_analysis_results.json",
    ]
