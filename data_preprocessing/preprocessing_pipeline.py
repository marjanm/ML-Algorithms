"""
Data Preprocessing Pipeline — End-to-End Demo
==============================================
Every real ML project starts here. This demonstrates:
  1. Handling missing values (imputation strategies)
  2. Encoding categoricals (one-hot, ordinal, target encoding)
  3. Scaling / normalisation (StandardScaler, MinMaxScaler, RobustScaler)
  4. Feature selection (variance threshold, mutual information)
  5. sklearn Pipeline — composable, reproducible preprocessing

Uses a messy synthetic dataset with mixed types, missing values, and outliers.

Run:
    python preprocessing_pipeline.py
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    OneHotEncoder, OrdinalEncoder, FunctionTransformer,
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def create_messy_dataset(n=1000, random_state=42):
    """Generate a realistic messy dataset with mixed types and missing values."""
    rng = np.random.RandomState(random_state)

    age = rng.normal(40, 15, n).clip(18, 80)
    income = rng.exponential(50000, n).clip(10000, 500000)
    credit_score = rng.normal(650, 100, n).clip(300, 850)
    years_employed = rng.exponential(5, n).clip(0, 40)

    education = rng.choice(["high_school", "bachelors", "masters", "phd"], n, p=[0.3, 0.35, 0.25, 0.1])
    region = rng.choice(["north", "south", "east", "west"], n)
    employment = rng.choice(["full_time", "part_time", "self_employed", "unemployed"], n, p=[0.5, 0.2, 0.2, 0.1])

    # target: likelihood of loan default
    score = (
        -0.02 * age + 0.00001 * income - 0.005 * credit_score
        + 0.1 * years_employed
        + (education == "phd").astype(float) * -0.5
        + rng.normal(0, 0.5, n)
    )
    target = (score > np.median(score)).astype(int)

    df = pd.DataFrame({
        "age": age, "income": income, "credit_score": credit_score,
        "years_employed": years_employed,
        "education": education, "region": region, "employment": employment,
        "target": target,
    })

    # inject missing values
    for col in ["age", "income", "credit_score"]:
        mask = rng.rand(n) < 0.08
        df.loc[mask, col] = np.nan
    mask = rng.rand(n) < 0.05
    df.loc[mask, "education"] = np.nan

    # inject outliers in income
    outlier_idx = rng.choice(n, 10, replace=False)
    df.loc[outlier_idx, "income"] = rng.uniform(800000, 2000000, 10)

    return df


def run_preprocessing_demo():
    lines = [
        "=" * 65,
        "  DATA PREPROCESSING PIPELINE  —  Demo",
        "=" * 65, "",
    ]

    df = create_messy_dataset()
    lines.append(f"  Dataset shape: {df.shape}")
    lines.append(f"  Target distribution: {dict(df['target'].value_counts())}")

    # ── Step 1: inspect missing values ──
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    lines += ["", "  ── Step 1: Missing Values ──"]
    for col, cnt in missing.items():
        lines.append(f"    {col:20s}: {cnt:3d} missing ({cnt/len(df)*100:.1f}%)")

    # ── Step 2: compare imputation strategies ──
    lines += ["", "  ── Step 2: Imputation Strategies ──"]

    X = df.drop("target", axis=1)
    y = df["target"]
    num_cols = ["age", "income", "credit_score", "years_employed"]
    cat_cols = ["education", "region", "employment"]

    strategies = {
        "mean": SimpleImputer(strategy="mean"),
        "median": SimpleImputer(strategy="median"),
        "most_frequent": SimpleImputer(strategy="most_frequent"),
    }
    for name, imp in strategies.items():
        X_num_imp = pd.DataFrame(imp.fit_transform(X[num_cols]), columns=num_cols)
        lines.append(f"    {name:15s} → income mean after impute: {X_num_imp['income'].mean():,.0f}")

    # ── Step 3: compare scalers ──
    lines += ["", "  ── Step 3: Scaling Comparison (on income column) ──"]
    X_num_clean = X[num_cols].fillna(X[num_cols].median())
    income = X_num_clean[["income"]]

    scalers = {
        "StandardScaler": StandardScaler(),     # zero mean, unit variance
        "MinMaxScaler": MinMaxScaler(),          # scales to [0, 1]
        "RobustScaler": RobustScaler(),          # uses median & IQR, robust to outliers
    }
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    axes[0].hist(income, bins=40, color="steelblue", edgecolor="white")
    axes[0].set_title("Original")
    for i, (name, scaler) in enumerate(scalers.items()):
        scaled = scaler.fit_transform(income)
        axes[i + 1].hist(scaled, bins=40, color="coral", edgecolor="white")
        axes[i + 1].set_title(name)
        lines.append(f"    {name:18s} → range: [{scaled.min():.2f}, {scaled.max():.2f}], mean: {scaled.mean():.2f}")
    plt.suptitle("Scaling Comparison — Income (with outliers)", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "scaling_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    lines.append(f"    [saved] → plots/scaling_comparison.png")

    # ── Step 4: encoding categoricals ──
    lines += ["", "  ── Step 4: Categorical Encoding ──"]

    edu_order = ["high_school", "bachelors", "masters", "phd"]
    edu_clean = X["education"].fillna("high_school")

    ohe = OneHotEncoder(sparse_output=False, drop="first")
    ohe_result = ohe.fit_transform(edu_clean.values.reshape(-1, 1))
    lines.append(f"    OneHotEncoder (education) → {ohe_result.shape[1]} columns (drop=first)")

    oe = OrdinalEncoder(categories=[edu_order])
    oe_result = oe.fit_transform(edu_clean.values.reshape(-1, 1))
    lines.append(f"    OrdinalEncoder (education) → values: {sorted(set(oe_result.flatten()))}")

    # ── Step 5: full sklearn Pipeline ──
    lines += ["", "  ── Step 5: Full sklearn Pipeline ──"]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),                       # robust to the outliers we injected
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ])

    # pipeline: preprocess → model
    pipelines = {
        "Logistic Regression": Pipeline([
            ("preprocess", preprocessor),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("preprocess", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]),
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lines.append(f"    Train/test split: {len(X_train)} / {len(X_test)}")
    lines.append("")

    for name, pipe in pipelines.items():
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")
        pipe.fit(X_train, y_train)
        test_acc = accuracy_score(y_test, pipe.predict(X_test))
        lines.append(f"    {name:25s} | CV acc: {cv_scores.mean():.4f} ± {cv_scores.std():.4f} | Test acc: {test_acc:.4f}")

    # ── Step 6: what the pipeline looks like ──
    lines += [
        "", "  ── Pipeline Structure ──",
        "    Pipeline(",
        "      ColumnTransformer([",
        "        ('num', Pipeline([Imputer(median), RobustScaler]), [age, income, ...]),",
        "        ('cat', Pipeline([Imputer(most_frequent), OneHotEncoder]), [education, region, ...]),",
        "      ]),",
        "      RandomForestClassifier(n_estimators=100)",
        "    )",
    ]

    # ── Step 7: visualise feature distributions before & after ──
    pipe_rf = pipelines["Random Forest"]
    X_train_transformed = pipe_rf.named_steps["preprocess"].transform(X_train)
    feature_names = (
        num_cols
        + list(pipe_rf.named_steps["preprocess"]
               .named_transformers_["cat"]
               .named_steps["onehot"]
               .get_feature_names_out(cat_cols))
    )

    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    for i, col in enumerate(num_cols):
        axes[0, i].hist(X_train[col].dropna(), bins=30, color="steelblue", edgecolor="white")
        axes[0, i].set_title(f"Before: {col}")
        axes[1, i].hist(X_train_transformed[:, i], bins=30, color="coral", edgecolor="white")
        axes[1, i].set_title(f"After: {col}")
    plt.suptitle("Numeric Features — Before vs After Preprocessing", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "before_after.png"), dpi=150, bbox_inches="tight")
    plt.close()
    lines.append(f"\n    [saved] → plots/before_after.png")

    lines += ["", "=" * 65]

    output_text = "\n".join(lines)
    print("\n" + output_text)
    with open(os.path.join(OUTPUT_DIR, "output.txt"), "w") as f:
        f.write(output_text)


if __name__ == "__main__":
    run_preprocessing_demo()
