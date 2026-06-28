"""LaTeX formatting helpers used by generated tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------

_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _latex_escape(value) -> str:
    s = "" if value is None else str(value)
    return "".join(_LATEX_ESCAPES.get(ch, ch) for ch in s)


def _format_number(x, digits: int = 3) -> str:
    if x is None:
        return "--"
    try:
        if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
            return "--"
    except TypeError:
        pass
    if isinstance(x, (int, np.integer)) and not isinstance(x, bool):
        return f"{int(x):,}"
    try:
        return f"{float(x):.{digits}f}"
    except (TypeError, ValueError):
        return _latex_escape(x)


def _sig_marker(p_value: float) -> str:
    if p_value is None or np.isnan(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def _write_tabular(
    path: Path,
    header: list[str],
    rows: list[list[str]],
    *,
    col_spec: str,
    pre_header: str | None = None,
    midrule_after_header: bool = True,
) -> None:
    """Write a bare tabular environment to ``path``.

    ``header`` and each row in ``rows`` must already be LaTeX-safe strings.
    """
    lines = [f"\\begin{{tabular}}{{{col_spec}}}", "\\toprule"]
    if pre_header:
        lines.append(pre_header)
    lines.append(" & ".join(header) + r" \\")
    if midrule_after_header:
        lines.append("\\midrule")
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

