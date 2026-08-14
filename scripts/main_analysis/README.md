# Replication Analysis Pipeline

Run the complete standalone analysis from this repository root:

```bash
uv run python scripts/05_Run_Analysis.py
```

The command delegates to `main_analysis/pipeline.py`, which owns the full run order:

1. Load `phase_2_context` simulations from the combined local-JSON database and attach KS/Wasserstein results.
2. Write `tex/tables/regression_w1_phase2_context.tex`.
3. Load `phase_2` simulations from `data/exp1`, `data/exp2`, and `data/exp3` and attach KS/Wasserstein results.
4. Write behavioral W1, reasoning, ground-truth, and random-number summary tables under `tex/tables/`.
5. Generate paper figures, first under `output/figures/`, then mirror them to `tex/figs/`.
6. Generate the five HD-0001 within-response mechanism artifacts directly under `tex/tables/` and `tex/figs/`.
7. Validate the exact HD-0002 corpus and persisted derived entities, then generate the five HD-0003 reasoning artifacts without embedding or summary API calls.

`tex/result.tex` is static. The pipeline does not rewrite it. When adding a new analysis output, generate the table under `tex/tables/` or the figure under `tex/figs/`, then manually add the corresponding `\input{...}` or `\includegraphics{...}` block to `tex/result.tex`.

## Module Map

- `environment.py`: repository paths and local-JSON database factory.
- `config.py`: shared constants, ordering, labels, and small model/dataframe helpers.
- `latex.py`: LaTeX escaping, numeric formatting, and tabular writing helpers.
- `regression_frame.py`: statsmodels-safe DataFrame/formula helper functions.
- `regression_tables.py`: phase_2_context regression table generation.
- `summary_tables.py`: phase_2 summary table generation.
- `figures.py`: phase_2 figure generation and mirroring to `tex/figs/`.
- `mechanism.py`: the selected first-answer, answer-position, and cumulative first-k mechanism outputs. The permutation, chunk-call correlation, marginal-return, and `ChunkN=20` diagnostics are not part of the replication pipeline.
- `reasoning.py`: the selected within-simulation cosine, PCA/K-means diagnostic, K=2 composition, and LLM-powered summary outputs. The K=3 through K=5 details, PC1-PC2 figures, full PCA tables, frequent terms, and cluster-colored projections remain outside the canonical replication analysis.
- `pipeline.py`: complete orchestration layer used by the compatibility entrypoint.
