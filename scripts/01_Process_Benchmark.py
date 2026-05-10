from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import pandas as pd


REPLICATION_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPLICATION_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from db_ops.local_json_db import to_jsonable


DATA_ROOT = REPLICATION_ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw"
BENCHMARK_DIR = DATA_ROOT / "benchmark"


def _benchmark(game_type: str, decisions: list) -> dict:
    return {
        "_id": str(uuid.uuid4()),
        "game_type": game_type,
        "decisions": decisions,
    }


def _write_json(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(records), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _single_round_users(df: pd.DataFrame) -> pd.DataFrame:
    user_counts = df["UserID"].value_counts()
    return df[df["UserID"].isin(user_counts[user_counts == 1].index)].copy()


def build_benchmarks(raw_root: Path = RAW_ROOT) -> list[dict]:
    pnas = raw_root / "pnas"
    benchmarks: list[dict] = []

    df = _single_round_users(pd.read_csv(pnas / "dictator.csv"))
    df = df[
        (df["Round"] == 1)
        & (df["Role"] == "first")
        & (df["Total"] == 100)
        & (df["move"] >= 0)
    ].copy()
    benchmarks.append(
        _benchmark("Dictator", [[v] for v in (df["move"] / df["Total"] * 100)])
    )

    df = _single_round_users(pd.read_csv(pnas / "public_goods_linear_water.csv"))
    df = df[
        (df["Round"] == 1)
        & (df["Total"] == 20)
        & (df["move"] >= 0)
        & (df["groupSize"] == 4)
    ].copy()
    benchmarks.append(_benchmark("Linear Public Good", [[v] for v in df["move"]]))

    df = _single_round_users(pd.read_csv(pnas / "ultimatum_strategy.csv"))
    df = df[(df["Round"] == 1) & (df["Total"] == 100)].copy()
    proposer = []
    responder = []
    for raw_decision in df["move"]:
        decision = json.loads(str(raw_decision).replace("'", '"'))
        if decision[0] >= 0 and decision[1] >= 0:
            proposer.append([decision[0]])
            responder.append([decision[1]])
    benchmarks.append(_benchmark("Ultimatum Strategy (Proposer)", proposer))
    benchmarks.append(_benchmark("Ultimatum Strategy (Responder)", responder))

    df = _single_round_users(pd.read_csv(pnas / "bomb_risk.csv"))
    df = df[(df["Round"] == 1) & (df["move"] >= 0)].copy()
    benchmarks.append(_benchmark("Bomb Risk", [int(v) for v in df["move"]]))

    df = _single_round_users(pd.read_csv(pnas / "push_pull.csv"))
    df = df[(df["Round"] == 1) & (df["move"] >= 0)].copy()
    benchmarks.append(_benchmark("Prisoner's Dilemma", [int(v) for v in df["move"]]))

    benchmarks.append(
        _benchmark("Random Number Generation", [[i] for i in range(1, 101)])
    )

    trustor = [[1] for _ in range(45)] + [[0] for _ in range(27)]
    benchmarks.append(_benchmark("Trust in CC09 (trustor)", trustor))

    trustee = []
    for amount, count in [[30, 6], [25, 1], [20, 10], [15, 1], [11, 1], [10, 10], [0, 43]]:
        trustee.extend([[amount] for _ in range(count)])
    benchmarks.append(_benchmark("Trust in CC09 (trustee)", trustee))

    benchmarks.append(
        _benchmark("BoS in CDJFR89", [[2] for _ in range(416)] + [[1] for _ in range(244)])
    )
    benchmarks.append(
        _benchmark("Stag Hunt in CDFR92", [[1] for _ in range(325)] + [[2] for _ in range(4)])
    )
    benchmarks.append(_benchmark("Arithmetic Verification", [[41] for _ in range(100)]))
    benchmarks.append(_benchmark("Trivial Dominance", [[1] for _ in range(100)]))
    benchmarks.append(_benchmark("TicTacToe Logic", [[2] for _ in range(100)]))
    benchmarks.append(_benchmark("TicTacToe Logic - L2", [[6] for _ in range(100)]))

    return benchmarks


def main() -> None:
    benchmarks = build_benchmarks()
    _write_json(BENCHMARK_DIR / "benchmarks.json", benchmarks)
    manifest = [
        {"game_type": row["game_type"], "n": len(row["decisions"])}
        for row in benchmarks
    ]
    _write_json(BENCHMARK_DIR / "benchmark_manifest.json", manifest)
    print(f"Wrote {len(benchmarks)} benchmark records to {BENCHMARK_DIR}")


if __name__ == "__main__":
    main()

