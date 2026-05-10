# Raw Benchmark Data

This folder duplicates the raw benchmark inputs needed by the replication
bundle. The benchmark processor reads from this folder only:

```bash
uv run python scripts/01_Process_Benchmark.py
```

The script rebuilds `data/benchmark/benchmarks.json` and
`data/benchmark/benchmark_manifest.json` from scratch.

