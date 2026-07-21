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
