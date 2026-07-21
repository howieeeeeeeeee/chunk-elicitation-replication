# Chunk Elicitation Replication

This folder is a standalone replication bundle for the chunk-elicitation
simulation and analysis pipeline. Simulation outputs are saved only as local
JSON files, and the analysis script reads those files to regenerate LaTeX
tables and figures. This repository never connects to MongoDB.

## Repository Layout

- `src/db_ops/`: local JSON database layer with a small PyMongo-like API.
- `scripts/01_Process_Benchmark.py`: rebuilds processed benchmark JSON from raw benchmark files.
- `scripts/02_Run_Experiment_1.py`: runs Experiment 1 into `data/exp1/`.
- `scripts/03_Run_Experiment_2.py`: runs Experiment 2 into `data/exp2/`.
- `scripts/04_Run_Experiment_3.py`: runs Experiment 3 into `data/exp3/`.
- `scripts/05_Run_Analysis.py`: reads local JSON and writes `tex/` artifacts.
- `data/raw/`: duplicated raw benchmark inputs used by the benchmark processor.
- `data/benchmark/benchmarks.json`: generated benchmark records.
- `data/exp*/simulations.json`: simulation-level records.
- `data/exp*/simulation_sessions.json`: session-level model outputs.
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

The three runners use declarative manifests in `scripts/experiment_specs.py`.
They save only to the local-JSON database in this repository and never connect
to MongoDB. Their semantic manifests match the MongoDB-backed formal runners in
the main project; only the persistence adapter and CLI storage option differ.
`scripts/plan.py` expands the plain manifests, while `scripts/_common.py` only
handles the local-JSON CLI and execution.

| Runner | Phase and tags | Formal scope | Planned cells |
| --- | --- | --- | ---: |
| `02_Run_Experiment_1.py` | `phase_2_context`, `extraFlag=[]` | Eight games, GPT-5.2, Atomic/ChunkN=10, reasoning on/off, context-incentive and privacy treatments | 672 |
| `03_Run_Experiment_2.py` | `phase_2`; main `extraFlag=[]`; reasoning-removal tag | Ten behavioral games, eight models, ChunkN 1/10/20/25/50/100 | 1,320 |
| `04_Run_Experiment_3.py` | `phase_2`; main `extraFlag=[]`; temperature tag | Four deterministic tasks and random-number generation, Atomic/ChunkN=10, Atomic temperature=2 | 165 |

The robustness tags are exactly
`small_experiments_on_removing_reasoning` and
`small_experiments_on_temperature`. A planned cell can be missing or failed;
the dry-run report distinguishes planned, completed, incomplete, missing,
duplicate, and legacy records. Existing failed/incomplete cells are not retried
unless `--retry-incomplete` is supplied explicitly.

First inspect each plan without calling any LLM APIs:

```bash
uv run python scripts/02_Run_Experiment_1.py --dry-run
uv run python scripts/03_Run_Experiment_2.py --dry-run
uv run python scripts/04_Run_Experiment_3.py --dry-run
```

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

Dry runs read the JSON file directly and do not create output directories or
write JSON. The included exported snapshot can contain legacy records outside
the formal manifests, including the excluded Gemini 3 Pro preview and broader
historical Experiment 3 configurations. They remain as provenance but are
reported as legacy and are not added to a formal rerun plan.

## Running Analysis

After rebuilding benchmarks and running simulations:

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

## Local JSON Contract

Each experiment folder stores two collections:

- `simulations.json`: one record per simulation configuration.
- `simulation_sessions.json`: one record per LLM call/session.

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
