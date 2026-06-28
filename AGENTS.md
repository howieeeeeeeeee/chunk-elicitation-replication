# AGENTS.md

## Development Environment

- **Package Manager**: Use `uv` for all Python package management and running scripts.
- **Running Python**: Always use `uv run python` or `uv run` when executing Python code.

## Main Repository Mirror

- Treat this repository as the standalone replication mirror for analysis code and simulation implementation changes from `../econ_experiment_agents`.
- Before planning or implementing mirrored analysis/simulation changes here, read `README.md` in detail and map the equivalent files in `../econ_experiment_agents`. The two repositories can have different structures.
- When analysis or simulation implementation changes are made here, make the corresponding change in `../econ_experiment_agents` as well, adapting to that repo's MongoDB-backed structure when needed.
- For every new analysis addition, not simulation-only additions, run the full analysis pipeline end to end in both repos:
  - replication repo: `uv run python scripts/05_Run_Analysis.py`
  - main repo: from `../econ_experiment_agents`, run `uv run python src/scripts/main_analysis.py`
- Commit the related changes in both repositories, keeping each repo's commit focused on its own mirrored changes.
