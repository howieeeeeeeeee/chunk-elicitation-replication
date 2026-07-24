# Chunk Elicitation Replication

This is the official replication package for Jian and Chen (2026), *The
sampling unit shapes behavioural similarity in large language model
simulations*. Simulation outputs are included as local JSON files, so readers
can inspect the existing data and run the analysis without rerunning the
simulations. This repository does not connect to MongoDB.

## Repository Layout

- `src/db_ops/`: local JSON database layer with a small PyMongo-like API.
- `scripts/01_Process_Benchmark.py`: rebuilds processed benchmark JSON from raw benchmark files.
- `scripts/02_Run_Experiment_1.py`: runs Experiment 1 into `data/exp1/`.
- `scripts/03_Run_Experiment_2.py`: runs Experiment 2 into `data/exp2/`.
- `scripts/04_Run_Experiment_3.py`: runs Experiment 3 into `data/exp3/`.
- `scripts/05_Run_Analysis.py`: reads local JSON and writes `tex/` artifacts.
- `scripts/06_Run_Embeddings.py`: guarded reasoning-embedding runner for experiment-local JSON.
- `data/raw/`: duplicated raw benchmark inputs used by the benchmark processor.
- `data/benchmark/benchmarks.json`: generated benchmark records.
- `data/exp*/simulations.json`: simulation-level records.
- `data/exp*/simulation_sessions.json`: session-level model outputs.
- `data/exp*/embeddings.json`: deterministic decision-level reasoning embeddings and attempt history.
- `data/derived/pca_analyses.json`: shared PCA lifecycle and aligned coordinates for exported embeddings.
- `data/derived/kmeans_analyses.json`: shared raw/PCA clustering lifecycle and nested per-cluster summary histories.
- `docs/data-schema.md`: readable schemas and lifecycle examples for all five simulation/derived collections.
- `tex/tables/` and `tex/figs/`: analysis outputs for paper tables/figures.

## Setup

From this folder:

```bash
uv sync
cp .env.example .env
```

Fill in `OPENROUTER_API_KEY` in `.env` if you plan to run new simulations.
The included runner scripts default to OpenRouter model IDs.

## Rebuild Benchmarks From Raw Files

The raw benchmark inputs are duplicated inside this replication folder under
`data/raw/`. The benchmark processor does not read from the main project and
does not use MongoDB. It rebuilds the processed benchmark JSON from scratch.

```bash
uv run python scripts/01_Process_Benchmark.py
```

This overwrites:

- `data/benchmark/benchmarks.json`
- `data/benchmark/benchmark_manifest.json`

The JSON shape matches the main project benchmark documents:

```json
{
  "_id": "uuid",
  "game_type": "Dictator",
  "decisions": [[50], [0], [20]]
}
```

## Running Simulations

Each runner keeps its experiment settings directly in the script so the exact
games, models, phase, tags, and treatment combinations are easy to inspect.
The scripts write only to `data/exp1`, `data/exp2`, and `data/exp3` as local
JSON. Use `--dry-run` to inspect existing and remaining cells without calling
an LLM API.

First inspect each plan without calling any LLM APIs:

```bash
uv run python scripts/02_Run_Experiment_1.py --dry-run
uv run python scripts/03_Run_Experiment_2.py --dry-run
uv run python scripts/04_Run_Experiment_3.py --dry-run
```

| Runner | Formal identity | Planned / included |
| --- | --- | ---: |
| `02_Run_Experiment_1.py` | `phase_2_context`, empty tag; eight behavioral games, GPT-5.2, Atomic/ChunkN=10, reasoning and prompt treatments | 672 / 672 |
| `03_Run_Experiment_2.py` | `phase_2`, empty main tag plus `small_experiments_on_removing_reasoning` | 1,320 / 1,319 |
| `04_Run_Experiment_3.py` | `phase_2`, empty main tag plus `small_experiments_on_temperature` | 165 / 161 |

The main repository runners use the same normalized signatures against
MongoDB. Historical missing/null defaults and an empty tag normalize safely;
operational worker count and human-readable simulation names are not identity.
Unmatched dry-run cells are report-only and must not be retried implicitly.

The one-way export includes only completed records matching these formal
manifests. `data/export_manifest.json` records included counts and compact
reasons for excluding legacy or non-formal completed rows. Those exclusions do
not delete or archive source MongoDB records. `scripts/05_Run_Analysis.py`
implements the manuscript analysis against this included snapshot.

Then run a script with confirmation skipped:

```bash
uv run python scripts/02_Run_Experiment_1.py --yes --max-workers 1
```

Each runner saves to its experiment folder. For example, Experiment 1 writes
`data/exp1/simulations.json` and `data/exp1/simulation_sessions.json`.

The top-level output folder is controlled by `--data-root`:

```bash
uv run python scripts/03_Run_Experiment_2.py --data-root data --yes
```

## Running Analysis

To analyze the included simulation data without rerunning the simulations:

```bash
uv run python scripts/05_Run_Analysis.py
```

The analysis script reads `data/exp1`, `data/exp2`, `data/exp3`, and
`data/benchmark`, then writes:

- `tex/tables/*.tex`
- `tex/figs/*.png`
- `tex/result.tex`

The generated `tex/result.tex` is a compact article-style wrapper that inputs
the tables and figures.

## Running Reasoning Embeddings

The embedding runner is intentionally non-runnable as checked in: its
simulation list is empty and its model is a placeholder. Confirm the safe
default without creating files, calling OpenRouter, or writing local JSON:

```bash
uv run python scripts/06_Run_Embeddings.py --dry-run
```

Select one experiment, explicit simulation ids, and a concrete OpenRouter
embedding model. Dry-run reads only the selected experiment JSON and reports
eligibility and existing-state counts:

```bash
uv run python scripts/06_Run_Embeddings.py \
  --experiment exp1 \
  --simulation-id <simulation-id> \
  --model <openrouter-embedding-model> \
  --dry-run
```

Applied mode runs sequentially and requires confirmation or `--yes`. It reads
`OPENROUTER_API_KEY` only from the environment, skips existing successes, and
allows one new attempt for a prior failure in a later invocation.

## Local JSON Contract

See [the complete data-schema guide](docs/data-schema.md) for document examples,
identity rules, lifecycle transitions, usage/cost fields, and the relationship
between simulations, sessions, embeddings, PCA, clustering, and summaries.

Each experiment folder stores three collections:

- `simulations.json`: one record per simulation configuration.
- `simulation_sessions.json`: one record per LLM call/session.
- `embeddings.json`: one record per simulation-session decision index and
  sanitized embedding configuration.

Derived analyses are shared across experiments because one selected embedding
set may span `exp1`, `exp2`, and `exp3`:

- `data/derived/pca_analyses.json` stores PCA identities, configurations,
  attempts, and aligned coordinates.
- `data/derived/kmeans_analyses.json` stores clustering over raw embeddings or
  one completed PCA entity plus independently resumable per-cluster summaries.

Both experiment-local and combined database handles expose these two shared
collections through `src/db_ops/`; they are never duplicated into experiment
folders. These modules define persistence only and do not compute PCA, run
k-means, or call OpenRouter.

The main repository export is an exact one-way snapshot. A later run of
`src/scripts/export-to-replication-folder.py` in the main repository atomically
replaces managed experiment JSON, including `embeddings.json`; replication-local
embedding records may therefore be intentionally replaced. The same rule
applies to managed files under `data/derived/`. A PCA record is exported only
when all referenced embeddings exist in the public snapshot; a PCA-derived
k-means record also requires its referenced PCA entity.

The simulation record keeps references to session IDs:

```json
{
  "_id": "simulation uuid",
  "phase_name": "phase_2",
  "simulation_config": {"game_type": "Dictator"},
  "instruction_config": {"explain_reasoning": true},
  "llm_config": {"model": "openai/gpt-5.2"},
  "simulation_sessions": ["session uuid"],
  "failed_sessions": [],
  "completed": true
}
```

The local database layer supports the small subset of queries used by the
simulation and analysis code: `find`, `find_one`, dotted keys, `$in`, `$or`,
and `$exists`.
