"""Regression table generation for phase_2_context."""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from .config import (
    CONTEXT_INCENTIVE_BASELINE,
    CONTEXT_INCENTIVE_REGRESSOR,
    DEFAULT_REGRESSORS,
)
from .latex import _format_number, _latex_escape, _sig_marker, _write_tabular
from .regression_frame import _is_categorical_column, _prepare_regression_frame

# ---------------------------------------------------------------------------
# Regression label shortening + reference-group overrides
# ---------------------------------------------------------------------------

# Categorical regressors that should use their no-prompt level as the reference
# whenever that level is present in the data. Matched against the *display* name.
_NOT_SPECIFIED_REF_DISPLAYS = {
    "Context",
    "Incentive Size",
    CONTEXT_INCENTIVE_REGRESSOR,
    "Privacy Treatment",
}
_REFERENCE_LEVELS_BY_DISPLAY = {
    CONTEXT_INCENTIVE_REGRESSOR: CONTEXT_INCENTIVE_BASELINE,
}

# Compact labels for the regression table.  Keys are (display name, level);
# value is the string shown in the leftmost column.  Anything not in this
# table falls back to "<short-var> = <level>".
_LEVEL_SHORT: dict[tuple[str, str], str] = {
    (CONTEXT_INCENTIVE_REGRESSOR, "Lab / Not Specified"): "Lab",
    (CONTEXT_INCENTIVE_REGRESSOR, "Lab / Standard"): "Lab x Low",
    (CONTEXT_INCENTIVE_REGRESSOR, "Lab / High"): "Lab x High",
    (CONTEXT_INCENTIVE_REGRESSOR, "Classroom / Not Specified"): "Classroom",
    (CONTEXT_INCENTIVE_REGRESSOR, "Classroom / Standard"): "Classroom x Low",
    (CONTEXT_INCENTIVE_REGRESSOR, "Classroom / High"): "Classroom x High",
    ("Context", "Lab"): "Lab",
    ("Context", "Classroom"): "Classroom",
    ("Incentive Size", "Standard"): "Std Pay",
    ("Incentive Size", "High"): "High Pay",
    ("Privacy Treatment", "Private"): "Private",
    ("Privacy Treatment", "Public"): "Public",
    ("Mode", "Chunk"): "Chunk",
    ("Mode", "Atomic"): "Atomic",
    ("Explain Reasoning", "True"): "Reasoning",
    ("Explain Reasoning", "False"): "No Reasoning",
    ("Theoretical Prediction", "True"): "ThePred",
    ("Theoretical Prediction", "False"): "No ThePred",
}

_LEVEL_LATEX: dict[tuple[str, str], str] = {
    (CONTEXT_INCENTIVE_REGRESSOR, "Lab / Not Specified"): "Lab",
    (CONTEXT_INCENTIVE_REGRESSOR, "Lab / Standard"): r"Lab $\times$ Low",
    (CONTEXT_INCENTIVE_REGRESSOR, "Lab / High"): r"Lab $\times$ High",
    (CONTEXT_INCENTIVE_REGRESSOR, "Classroom / Not Specified"): "Classroom",
    (CONTEXT_INCENTIVE_REGRESSOR, "Classroom / Standard"): r"Classroom $\times$ Low",
    (CONTEXT_INCENTIVE_REGRESSOR, "Classroom / High"): r"Classroom $\times$ High",
}

_VAR_SHORT: dict[str, str] = {
    CONTEXT_INCENTIVE_REGRESSOR: "Prompt",
    "Explain Reasoning": "Reasoning",
    "Theoretical Prediction": "ThePred",
    "Incentive Size": "Inc",
    "Privacy Treatment": "Priv",
    "LLM Model": "LLM",
}


def _short_label(display_name: str, level: str | None) -> str:
    if level is None:
        return _VAR_SHORT.get(display_name, display_name)
    key = (display_name, level)
    if key in _LEVEL_SHORT:
        return _LEVEL_SHORT[key]
    return f"{_VAR_SHORT.get(display_name, display_name)} = {level}"


def _short_label_latex(display_name: str, level: str | None) -> str:
    if level is not None:
        key = (display_name, level)
        if key in _LEVEL_LATEX:
            return _LEVEL_LATEX[key]
    return _latex_escape(_short_label(display_name, level))


def _clean_prompt_factor(value) -> str:
    if value is None or pd.isna(value):
        return "Not Specified"
    text = str(value).strip()
    return text if text else "Not Specified"


def _add_context_incentive_cell(df: pd.DataFrame) -> pd.DataFrame:
    """Add the observed Context x Incentive prompt-condition cell."""
    required = ["Context", "Incentive Size"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(
            f"Cannot build '{CONTEXT_INCENTIVE_REGRESSOR}'; missing columns: {missing}"
        )
    out = df.copy()
    context = out["Context"].map(_clean_prompt_factor)
    incentive = out["Incentive Size"].map(_clean_prompt_factor)
    out[CONTEXT_INCENTIVE_REGRESSOR] = context + " / " + incentive
    return out

# ---------------------------------------------------------------------------
# Section 1: Regression (phase_2_context, OLS + Fractional Logit, W1)
# ---------------------------------------------------------------------------


def _build_formula_with_refs(
    dep_id: str,
    regressor_ids: list[str],
    work: pd.DataFrame,
    ident_to_display: dict[str, str],
) -> str:
    """Build a statsmodels formula with no-prompt reference levels."""
    if not regressor_ids:
        return f"{dep_id} ~ 1"
    parts: list[str] = []
    for rid in regressor_ids:
        series = work[rid]
        if _is_categorical_column(series):
            display = ident_to_display.get(rid, rid)
            levels = {str(v) for v in series.dropna().unique()}
            ref_level = _REFERENCE_LEVELS_BY_DISPLAY.get(display, "Not Specified")
            if display in _NOT_SPECIFIED_REF_DISPLAYS and ref_level in levels:
                parts.append(f"C({rid}, Treatment(reference='{ref_level}'))")
            else:
                parts.append(f"C({rid})")
        else:
            parts.append(rid)
    return f"{dep_id} ~ " + " + ".join(parts)


def _fit_ols(
    work: pd.DataFrame,
    dep_id: str,
    regressor_ids: list[str],
    ident_to_display: dict[str, str],
):
    formula = _build_formula_with_refs(dep_id, regressor_ids, work, ident_to_display)
    return smf.ols(formula=formula, data=work).fit(), formula


def _fit_fractional_logit(
    work: pd.DataFrame,
    dep_id: str,
    regressor_ids: list[str],
    ident_to_display: dict[str, str],
):
    formula = _build_formula_with_refs(dep_id, regressor_ids, work, ident_to_display)
    model = smf.glm(formula=formula, data=work, family=sm.families.Binomial())
    return model.fit(cov_type="HC1"), formula


# Matches both ``C(name)[T.level]`` and
# ``C(name, Treatment(reference='X'))[T.level]`` forms that statsmodels emits.
# ``.*`` is greedy so the trailing ``)`` anchor lands on the outermost paren
# of ``C(...)`` even when ``Treatment(...)`` introduces nested parens.
_TERM_RE = re.compile(
    r"^C\((?P<name>[A-Za-z_][A-Za-z_0-9]*)(?:\s*,.*)?\)\[T\.(?P<level>.+?)\]$"
)


def _prettify_term(term: str, ident_to_display: dict[str, str]) -> str:
    """Turn a statsmodels term name into a compact, human-readable label.

    ``ident_to_display`` maps normalized identifiers back to their original
    display column names (e.g. ``"context"`` -> ``"Context"``).
    """

    def _display(name: str) -> str:
        return ident_to_display.get(name, name)

    if term == "Intercept":
        return "Intercept"

    m = _TERM_RE.match(term)
    if m:
        return _short_label(_display(m.group("name")), m.group("level"))

    # Fallback: strip a trailing ``[T.level]`` off a plain identifier.
    m = re.match(r"^(?P<name>.+?)\[T\.(?P<level>.+)\]$", term)
    if m:
        return _short_label(_display(m.group("name")), m.group("level"))

    return _short_label(_display(term), None)


def _prettify_term_latex(term: str, ident_to_display: dict[str, str]) -> str:
    """Return a LaTeX-safe display label for a statsmodels term."""

    def _display(name: str) -> str:
        return ident_to_display.get(name, name)

    if term == "Intercept":
        return "Intercept"

    m = _TERM_RE.match(term)
    if m:
        return _short_label_latex(_display(m.group("name")), m.group("level"))

    m = re.match(r"^(?P<name>.+?)\[T\.(?P<level>.+)\]$", term)
    if m:
        return _short_label_latex(_display(m.group("name")), m.group("level"))

    return _short_label_latex(_display(term), None)


def _drop_zero_variance(work: pd.DataFrame, regressor_ids: list[str]) -> list[str]:
    usable = []
    for rid in regressor_ids:
        if work[rid].nunique(dropna=True) >= 2:
            usable.append(rid)
    return usable


def _build_regression_table(
    df: pd.DataFrame,
    dependent_var: str,
    regressors: list[str],
) -> tuple[list[str], list[list[str]]]:
    if (
        CONTEXT_INCENTIVE_REGRESSOR in regressors
        and CONTEXT_INCENTIVE_REGRESSOR not in df.columns
    ):
        df = _add_context_incentive_cell(df)

    work, dep_id, regressor_ids, name_map = _prepare_regression_frame(
        df, dependent_var, regressors
    )
    regressor_ids = _drop_zero_variance(work, regressor_ids)
    ident_to_display = {ident: display for display, ident in name_map.items()}

    ols_res, _ = _fit_ols(work, dep_id, regressor_ids, ident_to_display)
    glm_res, _ = _fit_fractional_logit(work, dep_id, regressor_ids, ident_to_display)

    # Union of parameter names, preserving OLS ordering first.
    ordered_terms: list[str] = []
    seen = set()
    for term in list(ols_res.params.index) + list(glm_res.params.index):
        if term not in seen:
            seen.add(term)
            ordered_terms.append(term)

    # Reorder to the preferred presentation order, with Intercept pushed last.
    _DISPLAY_ORDER = [
        "Chunk",
        "Lab",
        "Lab x Low",
        "Lab x High",
        "Classroom",
        "Classroom x Low",
        "Classroom x High",
        "Private",
        "Public",
        "Reasoning",
    ]
    _display_rank = {name: idx for idx, name in enumerate(_DISPLAY_ORDER)}

    def _term_rank(term: str) -> tuple[int, int]:
        if term == "Intercept":
            return (2, 0)
        label = _prettify_term(term, ident_to_display)
        if label in _display_rank:
            return (0, _display_rank[label])
        return (1, 0)

    ordered_terms.sort(key=_term_rank)

    def _cell(res, term: str) -> tuple[str, str]:
        if term not in res.params.index:
            return "--", ""
        coef = res.params[term]
        se = res.bse[term]
        p = res.pvalues[term]
        marker = _sig_marker(p)
        coef_str = _format_number(coef, 3)
        if marker:
            coef_str = f"{coef_str}$^{{{marker}}}$"
        se_str = f"({_format_number(se, 3)})"
        return coef_str, se_str

    rows: list[list[str]] = []
    for term in ordered_terms:
        label = _prettify_term_latex(term, ident_to_display)
        ols_coef, ols_se = _cell(ols_res, term)
        glm_coef, glm_se = _cell(glm_res, term)
        rows.append([label, ols_coef, glm_coef])
        rows.append(["", ols_se, glm_se])

    # Footer rows.
    rows.append(["\\midrule N", f"{int(ols_res.nobs):,}", f"{int(glm_res.nobs):,}"])
    rows.append(
        [
            "R$^2$",
            _format_number(getattr(ols_res, "rsquared", None), 3),
            "--",
        ]
    )
    rows.append(
        [
            "Adj.\\ R$^2$",
            _format_number(getattr(ols_res, "rsquared_adj", None), 3),
            "--",
        ]
    )
    pseudo_r2 = None
    try:
        pseudo_r2 = 1.0 - (glm_res.deviance / glm_res.null_deviance)
    except Exception:  # noqa: BLE001
        pseudo_r2 = None
    rows.append(["Pseudo R$^2$", "--", _format_number(pseudo_r2, 3)])
    rows.append(
        [
            "Log-likelihood",
            _format_number(getattr(ols_res, "llf", None), 3),
            _format_number(getattr(glm_res, "llf", None), 3),
        ]
    )

    header = [
        "",
        "(1) OLS",
        "(2) Fractional Logit",
    ]
    return header, rows


def write_regression_table(
    df: pd.DataFrame,
    out_path: Path,
    *,
    dependent_var: str = "Wasserstein-1",
    regressors: Iterable[str] = DEFAULT_REGRESSORS,
) -> None:
    header, rows = _build_regression_table(df, dependent_var, list(regressors))
    _write_tabular(
        out_path,
        header=header,
        rows=rows,
        col_spec="lcc",
    )
    # Append a trailing \par with the significance legend so the \input site
    # gets the legend automatically.
    with out_path.open("a", encoding="utf-8") as fh:
        fh.write(
            "\n\\par\\smallskip\\footnotesize "
            "\\textit{Notes:} Dependent variable = Wasserstein-1. "
            "Standard errors in parentheses (HC1 robust for Fractional Logit). "
            "Prompt-condition reference = ``Not Specified / Not Specified'' "
            "(no context, no incentive). Privacy Treatment reference = "
            "``Not Specified''. "
            "Significance: $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.\n"
        )

