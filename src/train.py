from pathlib import Path
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
from preprocessing import clean_text


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8")


def leakage_guard(df: pd.DataFrame):
    before = len(df)
    df = df.drop_duplicates(subset=["cleaned_description"], keep="first")
    dropped = before - len(df)
    print(f"Leakage guard: dropped {dropped} duplicate cleaned_description rows")
    print(f"Final row count after leakage guard: {len(df)}")
    return df, dropped


def split_data(df: pd.DataFrame):
    X = df["cleaned_description"]
    y = df["category"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTraining set: {len(X_train)} rows")
    print(f"Test set: {len(X_test)} rows")
    print(f"\nTraining class counts:\n{y_train.value_counts()}")
    print(f"\nTest class counts:\n{y_test.value_counts()}")
    return X_train, X_test, y_train, y_test


def compare_models(X_train, y_train):
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)

    model_a = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    model_b = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", MultinomialNB()),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores_a = cross_val_score(model_a, X_train, y_train, cv=cv, scoring="f1_macro")
    scores_b = cross_val_score(model_b, X_train, y_train, cv=cv, scoring="f1_macro")

    mean_a, std_a = scores_a.mean(), scores_a.std()
    mean_b, std_b = scores_b.mean(), scores_b.std()

    print(f"\n5-fold Cross-Validation (f1_macro) on training set:")
    print(f"  Model A (Logistic Regression): mean={mean_a:.4f}, std={std_a:.4f}")
    print(f"  Model B (Naive Bayes):         mean={mean_b:.4f}, std={std_b:.4f}")

    chosen = "Logistic Regression" if mean_a >= mean_b else "Naive Bayes"
    print(f"  Chosen model: {chosen}")
    return chosen, mean_a, std_a, mean_b, std_b


def train_and_evaluate(X_train, X_test, y_train, y_test, chosen: str):
    if chosen == "Logistic Regression":
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000)),
        ])
    else:
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("clf", MultinomialNB()),
        ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro"
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted"
    )

    print(f"\nTest Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision (macro): {precision_macro:.4f}, Recall (macro): {recall_macro:.4f}, F1 (macro): {f1_macro:.4f}")
    print(f"  Precision (weighted): {precision_weighted:.4f}, Recall (weighted): {recall_weighted:.4f}, F1 (weighted): {f1_weighted:.4f}")
    print(f"\nClassification report:\n{classification_report(y_test, y_pred)}")

    cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
    print(f"\nConfusion matrix:")
    print(cm)

    return pipeline, accuracy, precision_macro, recall_macro, f1_macro, precision_weighted, recall_weighted, f1_weighted, cm


def save_confusion_matrix_plot(y_test, y_pred, classes, out_path: Path):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Category")
    ax.set_ylabel("True Category")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def save_top_features(pipeline, X_train, out_path: Path):
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["clf"]
    feature_names = vectorizer.get_feature_names_out()

    lines = ["Top 8 most influential words per class:\n"]
    print("\nTop 8 most influential words per class:")
    if hasattr(classifier, "coef_"):
        for i, class_label in enumerate(classifier.classes_):
            coef = classifier.coef_[i]
            top_indices = coef.argsort()[-8:][::-1]
            top_words = [feature_names[j] for j in top_indices]
            top_weights = [coef[j] for j in top_indices]
            lines.append(f"  {class_label}:")
            print(f"  {class_label}:")
            for w, wt in zip(top_words, top_weights):
                lines.append(f"    {w}: {wt:.4f}")
                print(f"    {w}: {wt:.4f}")
    else:
        log_prob = classifier.feature_log_prob_
        for i, class_label in enumerate(classifier.classes_):
            top_indices = log_prob[i].argsort()[-8:][::-1]
            top_words = [feature_names[j] for j in top_indices]
            top_weights = [log_prob[i][j] for j in top_indices]
            lines.append(f"  {class_label}:")
            print(f"  {class_label}:")
            for w, wt in zip(top_words, top_weights):
                lines.append(f"    {w}: {wt:.4f}")
                print(f"    {w}: {wt:.4f}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_misclassified(X_test, y_test, y_pred, out_path: Path):
    misclassified = []
    for text, true_label, pred_label in zip(X_test, y_test, y_pred):
        if true_label != pred_label:
            misclassified.append((text, true_label, pred_label))

    lines = ["Misclassified test tickets:\n"]
    print(f"\nMisclassified tickets: {len(misclassified)}")
    for i, (text, true_label, pred_label) in enumerate(misclassified[:10]):
        lines.append(f"  {i+1}. True: {true_label} | Predicted: {pred_label} | Text: {text[:100]}...")
        print(f"  {i+1}. True: {true_label} | Predicted: {pred_label} | Text: {text[:100]}...")

    lines.append("\nWhy misclassifications likely occurred:")
    lines.append("  1. Some descriptions contain words from multiple categories (ambiguous tickets).")
    lines.append("  2. The synthetic data includes label noise (~2% wrong labels).")
    lines.append("  3. Short descriptions with limited discriminative words confuse the model.")
    print("\nWhy misclassifications likely occurred:")
    print("  1. Some descriptions contain words from multiple categories (ambiguous tickets).")
    print("  2. The synthetic data includes label noise (~2% wrong labels).")
    print("  3. Short descriptions with limited discriminative words confuse the model.")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_results(accuracy, precision_macro, recall_macro, f1_macro,
                 precision_weighted, recall_weighted, f1_weighted,
                 report, cm, classes, chosen, mean_a, std_a, mean_b, std_b, out_path: Path):
    lines = []
    lines.append("MODEL TRAINING RESULTS")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Chosen model: {chosen}")
    lines.append("")
    lines.append("5-fold Cross-Validation (f1_macro) on training set:")
    lines.append(f"  Model A (Logistic Regression): mean={mean_a:.4f}, std={std_a:.4f}")
    lines.append(f"  Model B (Naive Bayes):         mean={mean_b:.4f}, std={std_b:.4f}")
    lines.append("")
    lines.append("Test set results:")
    lines.append(f"  Accuracy: {accuracy:.4f}")
    lines.append(f"  Precision (macro): {precision_macro:.4f}")
    lines.append(f"  Recall (macro): {recall_macro:.4f}")
    lines.append(f"  F1 (macro): {f1_macro:.4f}")
    lines.append(f"  Precision (weighted): {precision_weighted:.4f}")
    lines.append(f"  Recall (weighted): {recall_weighted:.4f}")
    lines.append(f"  F1 (weighted): {f1_weighted:.4f}")
    lines.append("")
    lines.append("Classification report:")
    lines.append(report)
    lines.append("")
    lines.append("Confusion matrix:")
    lines.append(str(cm))
    lines.append("")
    lines.append("Classes: " + ", ".join(classes))
    lines.append("")
    lines.append("Results in simple language:")
    lines.append(f"  The model correctly classified {accuracy*100:.1f}% of the test tickets. "
                 f"This means that out of every 100 tickets, about {int(accuracy*100)} "
                 f"were assigned to the right category. The macro-averaged F1 score of "
                 f"{f1_macro:.3f} shows how well the model performs across all categories "
                 f"equally, while the weighted-averaged F1 of {f1_weighted:.3f} accounts "
                 f"for class sizes. The model is performing {'satisfactorily' if accuracy >= 0.70 else 'below expectations'} - "
                 f"{'accuracy is above 70%, indicating the category signal is learnable' if accuracy >= 0.70 else 'accuracy is below 70%, indicating weak signal'}.")
    lines.append(f"  {'WARNING: Accuracy above 98% suggests templates may be too easy.' if accuracy > 0.98 else ''}")
    lines.append(f"  {'WARNING: Accuracy below 70% suggests weak signal between text and labels.' if accuracy < 0.70 else ''}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nMetrics saved to {out_path}")


def main():
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "tickets.csv"
    metrics_path = project_root / "data" / "metrics.txt"
    top_features_path = project_root / "data" / "top_features.txt"
    misclassified_path = project_root / "data" / "misclassified.txt"
    model_path = project_root / "model" / "ticket_classifier.joblib"
    labels_path = project_root / "model" / "labels.txt"
    cm_plot_path = project_root / "screenshots" / "confusion_matrix.png"

    df = load_data(data_path)
    df, dropped = leakage_guard(df)

    X_train, X_test, y_train, y_test = split_data(df)

    chosen, mean_a, std_a, mean_b, std_b = compare_models(X_train, y_train)

    pipeline, accuracy, p_macro, r_macro, f1_macro, p_weighted, r_weighted, f1_weighted, cm = train_and_evaluate(
        X_train, X_test, y_train, y_test, chosen
    )

    save_confusion_matrix_plot(y_test, pipeline.predict(X_test), pipeline.classes_, cm_plot_path)
    print(f"Saved confusion matrix to {cm_plot_path}")

    save_top_features(pipeline, X_train, top_features_path)
    save_misclassified(X_test, y_test, pipeline.predict(X_test), misclassified_path)

    save_results(accuracy, p_macro, r_macro, f1_macro,
                 p_weighted, r_weighted, f1_weighted,
                 classification_report(y_test, pipeline.predict(X_test)),
                 cm, pipeline.classes_, chosen, mean_a, std_a, mean_b, std_b, metrics_path)

    joblib.dump(pipeline, model_path)
    with open(labels_path, "w", encoding="utf-8") as f:
        for label in pipeline.classes_:
            f.write(label + "\n")
    print(f"Saved model to {model_path}")
    print(f"Saved labels to {labels_path}")


if __name__ == "__main__":
    main()