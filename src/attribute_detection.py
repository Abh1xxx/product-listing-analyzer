"""Rule-based attribute detection grounded in available dataset fields."""

import re
from collections.abc import Iterable

import pandas as pd


ATTRIBUTE_FIELDS = ("color", "material", "size", "features")


def _values_from_column(frame: pd.DataFrame, column: str | None) -> set[str]:
    if column is None or column not in frame:
        return set()
    return {
        str(value).strip()
        for value in frame[column].dropna().unique()
        if str(value).strip()
    }


def _feature_values(frame: pd.DataFrame, columns: dict) -> set[str]:
    return {
        value
        for column in columns.get("features", [])
        for value in _values_from_column(frame, column)
    }


def _contains_phrase(text: str, phrase: str) -> bool:
    pattern = r"(?<!\w)" + r"\s+".join(re.escape(part) for part in phrase.split()) + r"(?!\w)"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def detect_attributes(
    text: str,
    frame: pd.DataFrame,
    columns: dict[str, str | list[str] | None],
) -> dict[str, list[str]]:
    """Find values explicitly present in the corpus's attribute columns and text."""
    candidates = {
        "color": _values_from_column(frame, columns.get("color")),
        "material": _values_from_column(frame, columns.get("material")),
        "size": _values_from_column(frame, columns.get("size")),
        "features": _feature_values(frame, columns),
    }
    return {
        field: sorted(
            (value for value in values if _contains_phrase(text, value)),
            key=str.casefold,
        )
        for field, values in candidates.items()
    }


def available_attribute_fields(columns: dict) -> list[str]:
    """Return attribute categories supported by the actual dataset schema."""
    available = []
    for field in ATTRIBUTE_FIELDS:
        if field == "features":
            if columns.get("features"):
                available.append(field)
        elif columns.get(field):
            available.append(field)
    return available
