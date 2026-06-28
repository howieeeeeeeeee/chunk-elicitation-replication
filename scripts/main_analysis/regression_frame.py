"""Small statsmodels-safe regression frame helpers."""

from __future__ import annotations

import re

import pandas as pd


def _to_identifier(name: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_").lower()
    if not cleaned:
        cleaned = "col"
    if cleaned[0].isdigit():
        cleaned = f"c_{cleaned}"
    return cleaned


def _prepare_regression_frame(df, dependent_var, regressors):
    """Return a statsmodels-safe frame and display-to-identifier mapping."""
    if dependent_var not in df.columns:
        raise KeyError(f"Dependent variable '{dependent_var}' not found in data.")

    missing = [c for c in regressors if c not in df.columns]
    if missing:
        raise KeyError(f"Regressor(s) not in data: {missing}")

    used = [dependent_var, *regressors]
    work = df.loc[:, used].copy()

    name_map = {}
    used_ids = set()
    for col in used:
        ident = _to_identifier(col)
        base, suffix = ident, 1
        while ident in used_ids:
            suffix += 1
            ident = f"{base}_{suffix}"
        used_ids.add(ident)
        name_map[col] = ident
    work = work.rename(columns=name_map)

    dep_id = name_map[dependent_var]
    regressor_ids = [name_map[c] for c in regressors]

    work[dep_id] = pd.to_numeric(work[dep_id], errors="coerce")
    work = work.dropna(subset=[dep_id, *regressor_ids])

    return work, dep_id, regressor_ids, name_map


def _is_categorical_column(series: pd.Series) -> bool:
    if isinstance(series.dtype, pd.CategoricalDtype):
        return True
    if series.dtype == bool:
        return True
    if pd.api.types.is_numeric_dtype(series):
        return False
    return True


def _build_formula(dep_id: str, regressor_ids, work_df) -> str:
    if not regressor_ids:
        return f"{dep_id} ~ 1"
    parts = []
    for rid in regressor_ids:
        if _is_categorical_column(work_df[rid]):
            parts.append(f"C({rid})")
        else:
            parts.append(rid)
    return f"{dep_id} ~ " + " + ".join(parts)
