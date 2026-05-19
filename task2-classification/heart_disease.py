import sys
import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, learning_curve
)
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

DATA_PATH = Path(__file__).parent / "data" / "heart.csv"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

FEATURE_LABELS = {
    "age": "Age",
    "sex": "Sex (1=M)",
    "cp": "Chest Pain Type",
    "trestbps": "Resting BP",
    "chol": "Cholesterol",
    "fbs": "Fasting Blood Sugar",
    "restecg": "Rest ECG",
    "thalach": "Max Heart Rate",
    "exang": "Exercise Angina",
    "oldpeak": "ST Depression",
    "slope": "ST Slope",
    "ca": "Major Vessels",
    "thal": "Thalassemia",
    "age_thalach_ratio": "Age/MaxHR Ratio",
    "chest_pain_exang": "ChestPain x Angina",
    "oldpeak_slope": "STDep x Slope",
}


def load_data():
    df = pd.read_csv(DATA_PATH)
    df.replace("?", np.nan, inplace=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["target"] = (df["target"] > 0).astype(int)
    X = df.drop("target", axis=1)
    y = df["target"]
    print(f"Dataset: {df.shape[0]} patients, {X.shape[1]} features")
    print(f"Class balance: {y.value_counts().to_dict()} (0=healthy, 1=disease)")
    missing = X.isnull().sum()
    if missing.any():
        print(f"Missing values: {missing[missing > 0].to_dict()}")
    return X, y, df


def engineer_features(X):
    X = X.copy()
    X["age_thalach_ratio"] = X["age"] / (X["thalach"] + 1e-5)
    X["chest_pain_exang"] = X["cp"] * X["exang"]
    X["oldpeak_slope"] = X["oldpeak"] * (X["slope"] + 1)
    return X


def run_eda(df):
    print("\n--- EDA ---")
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, annot_kws={"size": 8})
    ax.set_title("Feature Correlation Matrix", fontsize=14, pad=12)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(fig)
    print("Saved: correlation_heatmap.png")


def build_pipeline(clf):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])


def compare_models(X, y):
    print("\n--- Model Comparison (5-Fold Stratified CV) ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = [
        ("Baseline (Majority)", DummyClassifier(strategy="most_frequent")),
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
        ("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=5)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("Support Vector Machine", SVC(probability=True, random_state=42)),
    ]

    results = []
    baseline_score = None
    best_name, best_score, best_clf = None, 0.0, None

    for name, clf in models:
        pipe = build_pipeline(clf)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
        mean, std = scores.mean(), scores.std()
        results.append((name, mean, std))

        if name == "Baseline (Majority)":
            baseline_score = mean

        if name != "Baseline (Majority)" and mean > best_score:
            best_score = mean
            best_name = name
            best_clf = clf

    header = f"{'Model':<30} {'Accuracy':>10} {'Std':>8} {'vs Baseline':>14}"
    print(header)
    print("-" * len(header))
    for name, mean, std in results:
        if baseline_score and name != "Baseline (Majority)":
            delta = f"+{(mean - baseline_score) * 100:.1f}%"
        else:
            delta = "--"
        marker = " << Best" if name == best_name else ""
        print(f"{name:<30} {mean:>10.4f} {std:>8.4f} {delta:>14}{marker}")

    return best_name, best_clf


def evaluate_best(X, y, best_clf, X_test, y_test):
    print(f"\n--- Detailed Evaluation: Best Model ---")

    pipe = build_pipeline(best_clf)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2,
                                               random_state=42, stratify=y)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred, target_names=["Healthy", "Disease"]))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Healthy", "Disease"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix", fontsize=13, pad=10)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print("Saved: confusion_matrix.png")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#2196F3", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#9E9E9E", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontsize=13)
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "roc_curve.png", dpi=150)
    plt.close(fig)
    print(f"Saved: roc_curve.png  (AUC = {roc_auc:.3f})")

    return pipe


def plot_analysis(X, y, best_name, best_clf):
    print("\n--- Feature Analysis & Learning Curves ---")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf_pipe = build_pipeline(
        RandomForestClassifier(n_estimators=100, random_state=42)
    )
    rf_pipe.fit(X_train, y_train)
    rf_clf = rf_pipe.named_steps["clf"]

    feat_names = list(X.columns)
    labels = [FEATURE_LABELS.get(f, f) for f in feat_names]

    rf_importance = pd.Series(rf_clf.feature_importances_, index=labels).sort_values()
    perm = permutation_importance(rf_pipe, X_test, y_test, n_repeats=10,
                                  random_state=42, n_jobs=-1)
    perm_importance = pd.Series(perm.importances_mean, index=labels).sort_values()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].barh(rf_importance.index, rf_importance.values, color="#4CAF50")
    axes[0].set_title("RF Feature Importance", fontsize=12)
    axes[0].set_xlabel("Importance Score")

    axes[1].barh(perm_importance.index, perm_importance.values, color="#FF9800")
    axes[1].set_title("Permutation Importance", fontsize=12)
    axes[1].set_xlabel("Mean Accuracy Drop")

    plt.suptitle("Feature Importance Analysis", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: feature_importance.png")

    best_pipe = build_pipeline(best_clf)
    train_sizes, train_scores, val_scores = learning_curve(
        best_pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
        train_sizes=np.linspace(0.1, 1.0, 8), scoring="accuracy", n_jobs=-1
    )
    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_mean, "o-", color="#2196F3", label="Training score")
    ax.plot(train_sizes, val_mean, "o-", color="#FF5722", label="CV score")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="#2196F3")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="#FF5722")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Learning Curve", fontsize=13)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "learning_curve.png", dpi=150)
    plt.close(fig)
    print("Saved: learning_curve.png")


def plot_shap(X_train, X_test, y_train):
    try:
        import shap
        rf_for_shap = build_pipeline(RandomForestClassifier(n_estimators=100, random_state=42))
        rf_for_shap.fit(X_train, y_train)

        imputer = rf_for_shap.named_steps["imputer"]
        scaler  = rf_for_shap.named_steps["scaler"]
        rf_clf  = rf_for_shap.named_steps["clf"]

        X_transformed = scaler.transform(imputer.transform(X_test))
        feat_names = list(X_test.columns)
        labels = [FEATURE_LABELS.get(f, f) for f in feat_names]

        explainer = shap.TreeExplainer(rf_clf)
        shap_values = explainer.shap_values(X_transformed)

        vals = shap_values[1] if isinstance(shap_values, list) else shap_values
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(vals, X_transformed, feature_names=labels,
                          plot_type="dot", show=False)
        plt.title("SHAP Feature Impact (Heart Disease Prediction)", fontsize=13, pad=12)
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved: shap_summary.png")
    except ImportError:
        print("shap not installed -- skipping SHAP plot")
    except Exception as e:
        print(f"SHAP plot skipped: {e}")


def feature_engineering_comparison(X, y, best_clf_class):
    print("\n--- Feature Engineering Impact ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    baseline_scores = cross_val_score(
        build_pipeline(best_clf_class), X, y, cv=cv, scoring="accuracy"
    )
    X_eng = engineer_features(X)
    eng_scores = cross_val_score(
        build_pipeline(best_clf_class), X_eng, y, cv=cv, scoring="accuracy"
    )

    b_mean, e_mean = baseline_scores.mean(), eng_scores.mean()
    delta = (e_mean - b_mean) * 100
    print(f"Without feature engineering: {b_mean:.4f} +/- {baseline_scores.std():.4f}")
    print(f"With feature engineering:    {e_mean:.4f} +/- {eng_scores.std():.4f}  ({delta:+.1f}%)")
    if delta > 0:
        print("Feature engineering improved performance -- using engineered features.")
        return X_eng
    else:
        print("Feature engineering did not help here -- sticking with original features.")
        return X


def interactive_predict(models_dict, X_columns):
    print("\n--- Heart Disease Risk Assessment ---")
    print("Enter patient data (press Enter to skip / use median):\n")

    prompts = [
        ("age", "Age (years)", 54),
        ("sex", "Sex (1=male, 0=female)", 1),
        ("cp", "Chest pain type (0=typical angina, 1=atypical, 2=non-anginal, 3=asymptomatic)", 0),
        ("trestbps", "Resting blood pressure (mm Hg)", 130),
        ("chol", "Serum cholesterol (mg/dl)", 246),
        ("fbs", "Fasting blood sugar > 120 mg/dl? (1=yes, 0=no)", 0),
        ("restecg", "Resting ECG (0=normal, 1=ST-T abnormality, 2=LV hypertrophy)", 0),
        ("thalach", "Max heart rate achieved", 150),
        ("exang", "Exercise-induced angina (1=yes, 0=no)", 0),
        ("oldpeak", "ST depression induced by exercise (0.0-6.0)", 1.0),
        ("slope", "Slope of peak exercise ST (0=upsloping, 1=flat, 2=downsloping)", 1),
        ("ca", "Number of major vessels colored by fluoroscopy (0-3)", 0),
        ("thal", "Thalassemia (1=normal, 2=fixed defect, 3=reversible defect)", 2),
    ]

    values = {}
    for key, label, default in prompts:
        try:
            raw = input(f"  {label} [{default}]: ").strip()
            values[key] = float(raw) if raw else float(default)
        except ValueError:
            values[key] = float(default)

    row = pd.DataFrame([values])
    if "age_thalach_ratio" in X_columns:
        row["age_thalach_ratio"] = row["age"] / (row["thalach"] + 1e-5)
        row["chest_pain_exang"] = row["cp"] * row["exang"]
        row["oldpeak_slope"] = row["oldpeak"] * (row["slope"] + 1)

    row = row[X_columns]

    print("\nRisk Assessment (Ensemble):")
    print("-" * 40)
    risk_votes = 0
    for name, pipe in models_dict.items():
        prob = pipe.predict_proba(row)[0][1] * 100
        risk_votes += (prob > 50)
        print(f"  {name:<28}: {prob:.1f}% risk")

    print("-" * 40)
    verdict = "HIGH RISK" if risk_votes >= len(models_dict) / 2 else "LOW RISK"
    print(f"  Consensus: {verdict} ({risk_votes}/{len(models_dict)} models agree)\n")


def main():
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    print("=" * 55)
    print("  Heart Disease Classification - ML Pipeline")
    print("=" * 55)

    X_raw, y, df = load_data()
    run_eda(df)

    clf_for_comparison = RandomForestClassifier(n_estimators=100, random_state=42)
    X = feature_engineering_comparison(X_raw, y, clf_for_comparison)

    best_name, best_clf = compare_models(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    trained_pipe = evaluate_best(X, y, best_clf, X_test, y_test)
    plot_analysis(X, y, best_name, best_clf)
    plot_shap(X_train, X_test, y_train)

    print("\n--- Training Final Ensemble for Interactive Prediction ---")
    model_specs = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
        ("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=5)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("Support Vector Machine", SVC(probability=True, random_state=42)),
    ]
    trained_models = {}
    for name, clf in model_specs:
        pipe = build_pipeline(clf)
        pipe.fit(X_train, y_train)
        trained_models[name] = pipe

    print("\nAll charts saved to output/")
    print(f"Best model: {best_name}")

    try:
        interactive_predict(trained_models, list(X.columns))
    except (KeyboardInterrupt, EOFError):
        print("\nSkipped interactive prediction.")


if __name__ == "__main__":
    main()
