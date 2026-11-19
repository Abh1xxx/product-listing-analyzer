"""Dataset loading and schema-aware column discovery."""

from pathlib import Path
import re

import pandas as pd


COLUMN_ALIASES = {
    "title": ("product title", "title", "name"),
    "description": ("product description", "description", "details"),
    "category": ("category", "product category"),
    "color": ("color", "colour"),
    "material": ("material", "fabric"),
    "size": ("size or variant", "size", "variant"),
    "product_type": ("product type", "type"),
}


def _normalize_column_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def identify_columns(columns: list[str]) -> dict[str, str | None]:
    """Map known product fields to columns that actually exist in the dataset."""
    normalized = {_normalize_column_name(column): column for column in columns}
    result: dict[str, str | None] = {}
    for field, aliases in COLUMN_ALIASES.items():
        result[field] = next(
            (normalized[alias] for alias in aliases if alias in normalized),
            None,
        )
    result["features"] = [
        column
        for column in columns
        if re.fullmatch(r"feature(?:[\s_]*\d+)?", str(column), flags=re.IGNORECASE)
    ]
    return result


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, dict[str, str | list[str] | None]]:
    """Load CSV or Excel data and remove exact duplicate records."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset was not found: {dataset_path}")

    suffix = dataset_path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(dataset_path)
    elif suffix in {".xlsx", ".xls"}:
        if suffix == ".xlsx":
            with pd.ExcelFile(dataset_path) as workbook:
                sheet_name = (
                    "Corpus"
                    if "Corpus" in workbook.sheet_names
                    else workbook.sheet_names[0]
                )
            frame = pd.read_excel(dataset_path, sheet_name=sheet_name)
        else:
            frame = pd.read_excel(dataset_path)
    else:
        raise ValueError("Dataset must be a CSV or Excel file.")

    original_row_count = len(frame)
    frame = frame.drop_duplicates().reset_index(drop=True)
    frame.attrs["duplicate_rows_removed"] = original_row_count - len(frame)
    return frame, identify_columns(frame.columns.tolist())


def create_demo_dataset() -> pd.DataFrame:
    """Return a tiny, clearly labelled fallback corpus for offline demonstrations."""
    return pd.DataFrame(
        [
            {
                "Product_Title": "Everyday Cotton T-Shirt",
                "Product_Description": "Soft cotton fabric with a comfortable fit for daily wear.",
                "Category": "Clothing",
                "Color": "blue",
                "Material": "cotton",
                "Size_or_Variant": "medium",
                "Feature_1": "comfortable",
                "Feature_2": "lightweight",
            },
            {
                "Product_Title": "Lightweight Canvas Backpack",
                "Product_Description": "Durable canvas backpack with adjustable straps and a spacious main compartment.",
                "Category": "Bags",
                "Color": "black",
                "Material": "canvas",
                "Size_or_Variant": "large",
                "Feature_1": "adjustable",
                "Feature_2": "durable",
            },
            {
                "Product_Title": "Classic Stainless Steel Watch",
                "Product_Description": "A stainless steel watch with a clear dial and an adjustable strap for everyday use.",
                "Category": "Watches",
                "Color": "silver",
                "Material": "stainless steel",
                "Size_or_Variant": "one size",
                "Feature_1": "adjustable",
                "Feature_2": "classic",
            },
        ]
    )
