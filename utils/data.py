"""Utilities to read Excel by ID and maintain a CSV of original rows and predictions.

Requirements implemented:
1) Read the Excel, find a line by ID, and return the line.
2) Write a new CSV that contains the original line and a line of predictions based on the predicted column.
3) Each time the ID already exists in the CSV and the value of that column is filled, replace the value; if empty, add the new value.
4) The CSV follows the structure (column order/names) of the original Excel.

Notes:
- Uses pandas for I/O.
- ID column can be an index (int) or a column name (str). Default assumes the first column.
- For step (2), we create a CSV with the header copied from the Excel. The prediction is written into the predicted column for an extra row labeled with a special marker unless in-place update is desired.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence

import pandas as pd
import numpy as np

default_csv_path_output = os.path.join('data', 'output.csv')
default_csv_path= os.path.join('data', 'Resultados_Patricia-Rodada-03.xlsx')

def _resolve_id_col(df: pd.DataFrame, id_col: int | str = 0) -> str:
    """Return the column name for id_col which may be index or name."""
    return df.columns[id_col] if isinstance(id_col, int) else id_col


def _read_table(path: str) -> pd.DataFrame:
    """Read a table from either CSV or Excel based on file extension."""
    lower = path.lower()
    if lower.endswith((".csv", ".txt")):
        return pd.read_csv(path)
    # default to Excel for .xlsx/.xls and other spreadsheet formats
    return pd.read_excel(path)


essential_marker_col = "__row_type__"  # distinguishes 'original' vs 'prediction' rows


def read_row_by_id(source_path: str, id_value: Any, id_col: int | str = 0) -> Optional[Dict[str, Any]]:
    """Read an Excel or CSV file and return the row as a dict where id column equals id_value.

    Returns None if not found.
    """
    df = _read_table(source_path)
    key = _resolve_id_col(df, id_col)
    row = df[df[key] == id_value]
    if row.empty:
        return None
    return row.iloc[0].to_dict() # basically converts series to a object


def create_output_csv(csv_path: str = default_csv_path, csv_path_output: str = default_csv_path_output) -> str:
    """Create a new CSV with the same columns as the source (Excel or CSV).

    Adds an extra hidden column to distinguish original vs prediction rows.
    """
    df = _read_table(csv_path)
    # Ensure CSV directory exists
    os.makedirs(os.path.dirname(csv_path_output) or ".", exist_ok=True)
    # Add marker col (kept at the end)
    df[essential_marker_col] = "original"
    # Write only header (no rows) to CSV to start clean structure
    header = df.columns
    pd.DataFrame(columns=header).to_csv(csv_path_output, index=False)
    return csv_path_output


def _ensure_csv(csv_path: str, columns: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    if not os.path.exists(csv_path):
        pd.DataFrame(columns=columns).to_csv(csv_path, index=False)


def upsert_prediction_row(
    csv_path: str,
    csv_path_output: str,
    id_value: Any,
    predicted_col: str | int,
    predicted_value: Any,
    id_col: int | str = 0,
) -> str:
    """Ensure CSV exists with Excel structure and upsert a prediction row for id_value.

    Behavior:
    - If CSV doesn't exist, create it using Excel's columns plus marker; add the original row and a prediction row.
    - If CSV exists but original row missing, append the original row and the prediction row.
    - If a prediction row for this id exists:
        * If predicted_col is filled, replace with the new value.
        * If empty, fill with the new value.
    Column handling:
    - predicted_col can be name or index; it must exist in the Excel columns.
    """
    # Load source (Excel or CSV) to get structure and original row
    src = _read_table(csv_path)
    id_key = _resolve_id_col(src, id_col)
    pred_key = src.columns[predicted_col] if isinstance(predicted_col, int) else predicted_col
    if pred_key not in src.columns:
        raise ValueError(f"predicted_col '{pred_key}' not found in Excel columns")

    original = src[src[id_key] == id_value]
    if original.empty:
        raise ValueError(f"ID {id_value} not found in source Excel")
    original = original.iloc[0].copy()

    # Prepare column order with marker
    cols_with_marker = list(src.columns) + [essential_marker_col]

    # Ensure CSV exists with right header
    _ensure_csv(csv_path_output, cols_with_marker)

    # Load CSV
    out = pd.read_csv(csv_path_output)

    # Convert id column type to be comparable
    if id_key in out.columns:
        # Try to coerce both to string for robust matching
        out[id_key] = out[id_key].astype(str)
        id_value_str = str(id_value)
    else:
        raise ValueError(f"CSV missing id column '{id_key}'. Is the CSV schema correct?")

    # Check for original row presence
    mask_orig = (out.get(essential_marker_col, "original") == "original") & (out[id_key] == id_value_str)
    if not mask_orig.any():
        # Append original row
        orig_row = original.to_dict()
        orig_row[essential_marker_col] = "original"
        out = pd.concat([out, pd.DataFrame([orig_row], columns=cols_with_marker)], ignore_index=True)

    # Locate existing prediction row for this ID
    mask_pred = (out.get(essential_marker_col, "prediction") == "prediction") & (out[id_key] == id_value_str)
    if mask_pred.any():
        # Replace regardless; could extend logic to track previous for auditing.
        out.loc[mask_pred, pred_key] = predicted_value
    else:
        # Create a new prediction row: copy original then set marker and predicted value
        pred_row = original.to_dict()
        pred_row[pred_key] = predicted_value
        pred_row[essential_marker_col] = "prediction"
        out = pd.concat([out, pd.DataFrame([pred_row], columns=cols_with_marker)], ignore_index=True)

    # Persist
    out.to_csv(csv_path_output, index=False)
    return csv_path_output
