from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import os
import threading
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable


EXPERIMENT_COLLECTIONS = (
    "simulations",
    "simulation_sessions",
    "embeddings",
    "benchmarks",
    "findings",
)
DERIVED_COLLECTIONS = (
    "pca_analyses",
    "kmeans_analyses",
)
COLLECTIONS = EXPERIMENT_COLLECTIONS + DERIVED_COLLECTIONS


@dataclass(frozen=True)
class _ParsedRecordCache:
    signature: tuple[int, int, int] | None
    records: list[dict]
    id_index: dict[Any, dict]


_LOCKS_GUARD = threading.RLock()
_LOCKS: dict[Path, threading.RLock] = {}
_RECORD_CACHES: dict[Path, _ParsedRecordCache] = {}


def to_jsonable(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    return value


def _get_by_dot_path(document: dict, dot_path: str) -> Any:
    current: Any = document
    for part in dot_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _matches_operator(value: Any, operator: str, expected: Any) -> bool:
    if operator == "$in":
        return value in expected
    if operator == "$exists":
        exists = value is not None
        return exists is bool(expected)
    raise ValueError(f"Unsupported query operator: {operator}")


def matches_filter(document: dict, query: dict | None) -> bool:
    if not query:
        return True
    for key, expected in query.items():
        if key == "$or":
            if not any(matches_filter(document, branch) for branch in expected):
                return False
            continue
        value = _get_by_dot_path(document, key)
        if isinstance(expected, dict) and any(k.startswith("$") for k in expected):
            for operator, operand in expected.items():
                if not _matches_operator(value, operator, operand):
                    return False
            continue
        if value != expected:
            return False
    return True


def _apply_projection(document: dict, projection: dict | None) -> dict:
    if not projection:
        return copy.deepcopy(document)
    include_keys = [key for key, enabled in projection.items() if enabled]
    if not include_keys:
        return copy.deepcopy(document)
    return {
        key: copy.deepcopy(_get_by_dot_path(document, key))
        for key in include_keys
        if _get_by_dot_path(document, key) is not None
    }


class LocalJsonCollection:
    def __init__(self, path: Path):
        self.path = path
        resolved = self.path.resolve()
        self._resolved_path = resolved
        with _LOCKS_GUARD:
            if resolved not in _LOCKS:
                _LOCKS[resolved] = threading.RLock()
            self._lock = _LOCKS[resolved]

    def _file_signature(self) -> tuple[int, int, int] | None:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return None
        return stat.st_mtime_ns, stat.st_size, stat.st_ino

    def _cache_records(
        self,
        records: list[dict],
        signature: tuple[int, int, int] | None,
    ) -> _ParsedRecordCache:
        id_index: dict[Any, dict] = {}
        for record in records:
            if "_id" not in record:
                continue
            try:
                id_index.setdefault(record["_id"], record)
            except TypeError:
                continue
        cache = _ParsedRecordCache(signature, records, id_index)
        _RECORD_CACHES[self._resolved_path] = cache
        return cache

    def _load_cache(self) -> _ParsedRecordCache:
        signature = self._file_signature()
        cached = _RECORD_CACHES.get(self._resolved_path)
        if cached is not None and cached.signature == signature:
            return cached
        if signature is None:
            return self._cache_records([], signature)
        with self.path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict) and "records" in data:
            data = data["records"]
        if not isinstance(data, list):
            raise ValueError(f"{self.path} must contain a JSON array.")
        return self._cache_records(data, signature)

    def _load(self) -> list[dict]:
        return self._load_cache().records

    def _write(self, records: list[dict]) -> None:
        jsonable_records = to_jsonable(records)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_name(
            f".{self.path.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            with temporary_path.open("w", encoding="utf-8") as fh:
                json.dump(jsonable_records, fh, indent=2, sort_keys=True)
                fh.write("\n")
                fh.flush()
                os.fsync(fh.fileno())
            temporary_path.replace(self.path)
            self._cache_records(jsonable_records, self._file_signature())
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def find(self, query: dict | None = None, projection: dict | None = None):
        with self._lock:
            rows = [
                _apply_projection(row, projection)
                for row in self._load()
                if matches_filter(row, query)
            ]
        return rows

    def find_one(self, query: dict | None = None, projection: dict | None = None):
        with self._lock:
            cache = self._load_cache()
            if isinstance(query, dict) and set(query) == {"_id"}:
                record_id = query["_id"]
                if not isinstance(record_id, dict):
                    try:
                        row = cache.id_index.get(record_id)
                    except TypeError:
                        row = None
                    if row is not None:
                        return _apply_projection(row, projection)
                    try:
                        hash(record_id)
                    except TypeError:
                        pass
                    else:
                        return None
            for row in cache.records:
                if matches_filter(row, query):
                    return _apply_projection(row, projection)
        return None

    def bulk_upsert(self, records: Iterable[dict]) -> dict:
        records = [to_jsonable(record) for record in records]
        with self._lock:
            existing = copy.deepcopy(self._load())
            positions = {row.get("_id"): i for i, row in enumerate(existing)}
            matched = 0
            upserted = 0
            for record in records:
                record_id = record.get("_id")
                if record_id is None:
                    raise ValueError("Every record must contain an '_id' field.")
                if record_id in positions:
                    existing[positions[record_id]] = record
                    matched += 1
                else:
                    positions[record_id] = len(existing)
                    existing.append(record)
                    upserted += 1
            self._write(existing)
        return {"matched_count": matched, "upserted_count": upserted}

    def replace_all(self, records: Iterable[dict]) -> None:
        with self._lock:
            self._write([to_jsonable(record) for record in records])


class CombinedJsonCollection:
    def __init__(self, collections: list[LocalJsonCollection]):
        self.collections = collections

    def find(self, query: dict | None = None, projection: dict | None = None):
        rows: list[dict] = []
        for collection in self.collections:
            rows.extend(collection.find(query=query, projection=projection))
        return rows

    def find_one(self, query: dict | None = None, projection: dict | None = None):
        for collection in self.collections:
            row = collection.find_one(query=query, projection=projection)
            if row is not None:
                return row
        return None


class LocalJsonDatabase:
    def __init__(
        self,
        base_dir: Path,
        benchmark_dir: Path | None = None,
        derived_dir: Path | None = None,
    ):
        self.base_dir = Path(base_dir)
        benchmark_base = Path(benchmark_dir) if benchmark_dir else self.base_dir
        derived_base = (
            Path(derived_dir) if derived_dir else self.base_dir.parent / "derived"
        )
        self._collections = {
            name: LocalJsonCollection(
                (benchmark_base if name == "benchmarks" else self.base_dir)
                / f"{name}.json"
            )
            for name in EXPERIMENT_COLLECTIONS
        }
        self._collections.update(
            {
                name: LocalJsonCollection(derived_base / f"{name}.json")
                for name in DERIVED_COLLECTIONS
            }
        )

    def __getitem__(self, collection_name: str):
        return self._collections[collection_name]

    def __getattr__(self, collection_name: str):
        try:
            return self._collections[collection_name]
        except KeyError as exc:
            raise AttributeError(collection_name) from exc


class CombinedJsonDatabase:
    def __init__(
        self,
        experiment_dirs: Iterable[Path],
        benchmark_dir: Path,
        derived_dir: Path | None = None,
    ):
        derived_base = (
            Path(derived_dir)
            if derived_dir
            else Path(benchmark_dir).parent / "derived"
        )
        experiment_dbs = [
            LocalJsonDatabase(
                Path(exp_dir),
                benchmark_dir=benchmark_dir,
                derived_dir=derived_base,
            )
            for exp_dir in experiment_dirs
        ]
        benchmark_db = LocalJsonDatabase(
            benchmark_dir,
            derived_dir=derived_base,
        )
        self._collections = {
            "simulations": CombinedJsonCollection(
                [db.simulations for db in experiment_dbs]
            ),
            "simulation_sessions": CombinedJsonCollection(
                [db.simulation_sessions for db in experiment_dbs]
            ),
            "embeddings": CombinedJsonCollection(
                [db.embeddings for db in experiment_dbs]
            ),
            "benchmarks": benchmark_db.benchmarks,
            "findings": CombinedJsonCollection([db.findings for db in experiment_dbs]),
            "pca_analyses": LocalJsonCollection(
                derived_base / "pca_analyses.json"
            ),
            "kmeans_analyses": LocalJsonCollection(
                derived_base / "kmeans_analyses.json"
            ),
        }

    def __getitem__(self, collection_name: str):
        return self._collections[collection_name]

    def __getattr__(self, collection_name: str):
        try:
            return self._collections[collection_name]
        except KeyError as exc:
            raise AttributeError(collection_name) from exc
