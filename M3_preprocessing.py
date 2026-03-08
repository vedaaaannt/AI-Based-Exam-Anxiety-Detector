import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Label mapping — string label to integer ID
LABEL2ID = {"Low Anxiety": 0, "Moderate Anxiety": 1, "High Anxiety": 2}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}

# Alternate label formats that may appear in raw data
LABEL_MAP = {
    "Low Anxiety":      "Low Anxiety",
    "Moderate Anxiety": "Moderate Anxiety",
    "High Anxiety":     "High Anxiety",
    "Low":              "Low Anxiety",
    "Moderate":         "Moderate Anxiety",
    "High":             "High Anxiety",
    "0":                "Low Anxiety",
    "1":                "Moderate Anxiety",
    "2":                "High Anxiety",
    "Mild":             "Low Anxiety",
    "Severe":           "High Anxiety",
}


# ── Activity 3.1 — Handling Missing Text Data ─────────────────
# Remove nulls, empty strings, duplicates, and non-English rows
def handle_missing_text(df):
    before = len(df)

    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    df = df[df["text"].str.contains(r"[a-zA-Z]", regex=True)]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    after = len(df)
    print(f"Rows before: {before} | After cleaning: {after} | Removed: {before - after}\n")
    return df


# ── Activity 3.2 — Understanding Original Labels ─────────────
# Print all unique label values and fix casing issues
def understand_labels(df):
    unique_labels = df["label"].unique()
    print(f"Unique labels found ({len(unique_labels)}):")
    for lbl in sorted(unique_labels):
        count = (df["label"] == lbl).sum()
        print(f"  '{lbl}' → {count} samples")

    # Fix inconsistent casing or whitespace
    df["label_clean"] = df["label"].astype(str).str.strip().str.title()
    issues = (df["label"] != df["label_clean"]).sum()
    if issues:
        print(f"\n{issues} labels had casing issues — auto-fixed")
    df["label"] = df["label_clean"]
    df.drop(columns=["label_clean"], inplace=True)
    print()
    return df


# ── Activity 3.3 — Anxiety Level Label Mapping ───────────────
# Map raw label strings to standardised Low/Moderate/High values
def map_labels(df):
    print("Label mapping:")
    for orig, mapped in LABEL_MAP.items():
        print(f"  '{orig}' → '{mapped}'")

    df["label"] = df["label"].map(LABEL_MAP)

    # Drop rows with labels that could not be mapped
    unmapped = df["label"].isnull().sum()
    if unmapped:
        print(f"\n{unmapped} rows could not be mapped — dropping them")
        df = df.dropna(subset=["label"]).reset_index(drop=True)

    print("\nFinal label counts:")
    for lbl, count in df["label"].value_counts().items():
        print(f"  {lbl}: {count}")
    print()
    return df


# ── Activity 3.4 — Creating Numerical Labels ─────────────────
# Convert string labels to integers (0, 1, 2) for BERT training
def create_numerical_labels(df):
    df["label_id"] = df["label"].map(LABEL2ID)

    print("Numeric encoding:")
    for name, num in LABEL2ID.items():
        print(f"  {num} → {name}")

    print("\nSample rows:")
    print(df[["text", "label", "label_id"]].head(5).to_string(index=False))
    print()
    return df


# ── Activity 3.5 — Validation of Label Mapping ───────────────
# Confirm label and label_id columns are consistent and complete
def validate_label_mapping(df):
    checks = {
        "No null labels":            df["label"].isnull().sum() == 0,
        "No null label_ids":         df["label_id"].isnull().sum() == 0,
        "Exactly 3 unique labels":   df["label"].nunique() == 3,
        "label_id range is 0–2":     df["label_id"].between(0, 2).all(),
        "label ↔ label_id match":    all(
            df[df["label"] == lbl]["label_id"].eq(idx).all()
            for lbl, idx in LABEL2ID.items()
        ),
    }

    for check, passed in checks.items():
        print(f"{'✅' if passed else '❌'}  {check}")
    print()


# ── Activity 3.6 — Final Dataset Preparation ─────────────────
# Split data into train/val/test sets and save as CSV files
def prepare_final_dataset(df):
    df = df[["text", "label", "label_id"]].copy()

    # 70% train, 15% val, 15% test
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df["label_id"])
    val_df, test_df   = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df["label_id"])

    os.makedirs("data/processed", exist_ok=True)
    train_df.to_csv("data/processed/train.csv", index=False)
    val_df.to_csv("data/processed/val.csv",     index=False)
    test_df.to_csv("data/processed/test.csv",   index=False)

    print(f"Train : {len(train_df)} samples → data/processed/train.csv")
    print(f"Val   : {len(val_df)}   samples → data/processed/val.csv")
    print(f"Test  : {len(test_df)}  samples → data/processed/test.csv")

    # Save label map for use in backend
    with open("data/processed/label_map.json", "w") as f:
        json.dump({"LABEL2ID": LABEL2ID, "ID2LABEL": ID2LABEL}, f, indent=2)
    print("Label map → data/processed/label_map.json\n")
    return train_df, val_df, test_df


if __name__ == "__main__":
    print("MILESTONE 3 - Data Preprocessing & Label Mapping\n")

    path = "data/processed/clean_dataset.csv"
    if not os.path.exists(path):
        print(f"{path} not found — run M2 first")
        exit()

    df = pd.read_csv(path)
    print(f"Loaded: {path} ({len(df)} rows)\n")

    df = handle_missing_text(df)
    df = understand_labels(df)
    df = map_labels(df)
    df = create_numerical_labels(df)
    validate_label_mapping(df)
    prepare_final_dataset(df)
    print("MILESTONE 3 COMPLETE")
