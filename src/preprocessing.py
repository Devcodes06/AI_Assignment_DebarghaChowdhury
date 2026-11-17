import re
import sys
from pathlib import Path

import pandas as pd


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\S+@\S+\.\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    print(f"Loaded shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Dtypes:\n{df.dtypes}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nMissing values per column:\n{df.isnull().sum()}")
    print(f"\nDuplicate rows (all columns): {df.duplicated().sum()}")
    return df


def drop_empty_description_category(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    mask = df["ticket_description"].isna() | (df["ticket_description"].str.strip() == "")
    mask = mask | df["category"].isna() | (df["category"].str.strip() == "")
    df = df[~mask].copy()
    print(f"Dropped {before - len(df)} rows with empty description or category")
    return df


def fill_unknown(df: pd.DataFrame) -> pd.DataFrame:
    df["priority"] = df["priority"].fillna("Unknown")
    df["priority"] = df["priority"].replace("", "Unknown")
    df["status"] = df["status"].fillna("Unknown")
    df["status"] = df["status"].replace("", "Unknown")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df)} exact duplicate rows")

    before = len(df)
    df = df.drop_duplicates(subset=["ticket_id"], keep="first")
    print(f"Dropped {before - len(df)} rows with duplicate ticket_id")
    return df


def create_cleaned_description(df: pd.DataFrame) -> pd.DataFrame:
    df["cleaned_description"] = df["ticket_description"].apply(clean_text)
    before = len(df)
    df = df[df["cleaned_description"] != ""].copy()
    print(f"Dropped {before - len(df)} rows with empty cleaned_description")
    return df


def save_output(df: pd.DataFrame, out_path: Path, summary_path: Path, row_counts: dict):
    df = df[["ticket_id", "ticket_description", "cleaned_description", "category", "priority", "status"]]
    df.to_csv(out_path, index=False, encoding="utf-8")

    lines = ["Row counts through preprocessing pipeline:\n"]
    for step, count in row_counts.items():
        lines.append(f"  {step}: {count}")
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved final dataset to {out_path}")
    print(f"Saved summary to {summary_path}")


def main():
    project_root = Path(__file__).parent.parent
    in_path = project_root / "data" / "raw_tickets_synthetic.csv"
    out_path = project_root / "data" / "tickets.csv"
    summary_path = project_root / "data" / "prepare_summary.txt"

    row_counts = {}

    df = load_data(in_path)
    row_counts["Raw rows"] = len(df)

    df = drop_empty_description_category(df)
    row_counts["After dropping empty description/category"] = len(df)

    df = fill_unknown(df)

    df = remove_duplicates(df)
    row_counts["After duplicate removal"] = len(df)

    df = create_cleaned_description(df)
    row_counts["After text cleaning (empty cleaned_description dropped)"] = len(df)

    save_output(df, out_path, summary_path, row_counts)
    print(f"\nFinal shape: {df.shape}")
    print(f"Final columns: {list(df.columns)}")


if __name__ == "__main__":
    main()