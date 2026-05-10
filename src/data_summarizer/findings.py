import pandas as pd
from datetime import datetime


def show_all_findings_df(_db, include_deleted=False):
    """Fetch all findings and return as DataFrame."""
    query = {} if include_deleted else {"deleted": False}
    findings = list(_db.findings.find(query))
    if not findings:
        return pd.DataFrame()

    df = (
        pd.DataFrame(findings)
        .drop(columns=["related_simulations", "note", "deleted"])
        .set_index("_id")
    )
    return df
