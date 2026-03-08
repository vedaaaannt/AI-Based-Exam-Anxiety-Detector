import os
import pandas as pd
import numpy as np

# ── Activity 2.1 — Dataset Selection ─────────────────────────
# Build a synthetic dataset that mirrors real student anxiety text
def select_dataset():
    low_anxiety = [
        "I feel well prepared for my exam. A little nervous but confident.",
        "I've studied all the chapters and I'm ready for tomorrow.",
        "Feeling calm and collected. I trust my preparation.",
        "Mild butterflies but mostly okay. I reviewed everything twice.",
        "I'm not worried at all — I've been consistent in my studies.",
        "Looking forward to showing what I know in the exam.",
        "The syllabus is complete and I feel in control of the material.",
        "I'm slightly anxious but it's manageable. Nothing unusual.",
        "I had good sleep and feel fresh for the exam.",
        "I've practiced enough questions and feel confident.",
    ] * 15

    moderate_anxiety = [
        "I'm worried I might forget the formulas under pressure.",
        "The syllabus is huge and I'm not sure I covered everything.",
        "I feel stressed; I keep second-guessing my preparation.",
        "There are a few topics I haven't revised thoroughly.",
        "What if the exam has questions I haven't prepared for?",
        "I'm nervous and having some trouble sleeping.",
        "I studied but still feel uneasy about the outcome.",
        "My heart races whenever I think about the exam hall.",
        "I'm anxious about not finishing the paper on time.",
        "I fear blanking out even though I know most of the content.",
    ] * 15

    high_anxiety = [
        "I'm completely overwhelmed and can't think straight.",
        "I feel like I'm going to fail no matter how much I study.",
        "I can't breathe properly every time I open my textbook.",
        "I've been crying; the pressure is truly unbearable right now.",
        "My mind goes completely blank the moment I think about exams.",
        "I'm terrified — I haven't slept in two days because of anxiety.",
        "I feel nauseous and shaking just thinking about walking in.",
        "The fear of failure is paralyzing me and I can't function.",
        "I cannot focus at all and panic attacks keep happening.",
        "I want to give up. I feel helpless, hopeless, and exhausted.",
    ] * 15

    rows = []
    for label, texts in [("Low Anxiety", low_anxiety),
                          ("Moderate Anxiety", moderate_anxiety),
                          ("High Anxiety", high_anxiety)]:
        for text in texts:
            rows.append({"text": text, "label": label})

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/anxiety_dataset.csv", index=False)
    print(f"Dataset saved → data/raw/anxiety_dataset.csv ({len(df)} samples)\n")
    return df


# ── Activity 2.2 — Dataset Loading ───────────────────────────
# Load dataset from CSV file
def load_dataset(path="data/raw/anxiety_dataset.csv"):
    if not os.path.exists(path):
        print("File not found. Generating dataset...")
        return select_dataset()
    df = pd.read_csv(path)
    print(f"Loaded: {path}")
    print(f"Shape : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}\n")
    return df


# ── Activity 2.3 — Understanding Dataset Structure ────────────
# Show data types, sample rows, and text length stats
def understand_structure(df):
    print(f"Shape     : {df.shape}")
    print(f"Data Types:\n{df.dtypes.to_string()}\n")
    print("First 3 rows:")
    print(df.head(3).to_string(index=False))

    # Add temporary columns for text stats
    df["text_length"] = df["text"].astype(str).apply(len)
    df["word_count"]  = df["text"].astype(str).apply(lambda x: len(x.split()))
    print("\nText Length Stats:")
    print(df[["text_length", "word_count"]].describe().round(2).to_string())
    df.drop(columns=["text_length", "word_count"], inplace=True)
    print()


# ── Activity 2.4 — Checking Missing Values ───────────────────
# Find and remove rows with null or empty text
def check_missing_values(df):
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
    print(report.to_string())

    before = len(df)
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].astype(str).str.strip() != ""]
    after = len(df)
    print(f"\nRows removed: {before - after}")
    print(f"Remaining  : {after}\n")
    return df


# ── Activity 2.5 — Class Distribution Analysis ───────────────
# Show count and percentage of each anxiety class
def class_distribution(df):
    dist = df["label"].value_counts()
    pct  = (dist / len(df) * 100).round(2)

    print(f"{'Label':<25} {'Count':>8} {'Percent':>10}")
    print("-" * 46)
    for lbl in dist.index:
        bar = "█" * int(pct[lbl] / 2)
        print(f"{lbl:<25} {dist[lbl]:>8} {pct[lbl]:>9.1f}%  {bar}")

    # Check imbalance ratio
    ratio = dist.max() / dist.min()
    print(f"\nImbalance ratio: {ratio:.2f}")
    if ratio <= 1.5:
        print("Dataset is well balanced\n")
    else:
        print("Consider oversampling or class weights\n")


# ── Activity 2.6 — Dataset Suitability Assessment ────────────
# Validate dataset meets minimum requirements for BERT training
def suitability_assessment(df):
    checks = {
        "Minimum 200 samples":         len(df) >= 200,
        "Has 'text' column":           "text" in df.columns,
        "Has 'label' column":          "label" in df.columns,
        "Exactly 3 unique labels":     df["label"].nunique() == 3,
        "No null text values":         df["text"].isnull().sum() == 0,
        "Avg text length >= 10 chars": df["text"].astype(str).apply(len).mean() >= 10,
    }

    for check, passed in checks.items():
        print(f"{'✅' if passed else '❌'}  {check}")

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/clean_dataset.csv", index=False)
    print(f"\nClean dataset saved → data/processed/clean_dataset.csv\n")


if __name__ == "__main__":
    print("MILESTONE 2 - Dataset Collection & EDA\n")
    df = select_dataset()
    df = load_dataset()
    understand_structure(df)
    df = check_missing_values(df)
    class_distribution(df)
    suitability_assessment(df)
    print("MILESTONE 2 COMPLETE")
