from pathlib import Path
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
from preprocessing import clean_text

STOP_WORDS = {
    "the", "a", "an", "and", "to", "of", "is", "in", "it", "i",
    "my", "me", "for", "on", "with", "this", "that", "have", "has",
    "but", "not", "be", "are", "was", "you", "your", "please",
    "thanks", "hi", "hello", "team",
}


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8")


def top_words(texts, n=8):
    counter = Counter()
    for t in texts:
        for w in str(t).split():
            if w not in STOP_WORDS:
                counter[w] += 1
    return counter.most_common(n)


def print_eda(df: pd.DataFrame) -> dict:
    stats = {}

    total = len(df)
    stats["total"] = total
    print(f"Total tickets: {total}")

    cat_vc = df["category"].value_counts()
    cat_pct = df["category"].value_counts(normalize=True).mul(100).round(2)
    stats["category"] = cat_vc
    print("\nCategory distribution:")
    for cat in cat_vc.index:
        print(f"  {cat}: {cat_vc[cat]} ({cat_pct[cat]}%)")

    pri_vc = df["priority"].value_counts()
    pri_pct = df["priority"].value_counts(normalize=True).mul(100).round(2)
    stats["priority"] = pri_vc
    print("\nPriority distribution:")
    for p in pri_vc.index:
        print(f"  {p}: {pri_vc[p]} ({pri_pct[p]}%)")

    sta_vc = df["status"].value_counts()
    sta_pct = df["status"].value_counts(normalize=True).mul(100).round(2)
    stats["status"] = sta_vc
    print("\nStatus distribution:")
    for s in sta_vc.index:
        print(f"  {s}: {sta_vc[s]} ({sta_pct[s]}%)")

    word_counts = df["cleaned_description"].str.split().str.len()
    stats["wc_min"] = word_counts.min()
    stats["wc_mean"] = word_counts.mean()
    stats["wc_median"] = word_counts.median()
    stats["wc_max"] = word_counts.max()
    print(f"\nWord count stats: min={word_counts.min()}, mean={word_counts.mean():.1f}, "
          f"median={word_counts.median():.1f}, max={word_counts.max()}")

    desc_cat = df.groupby("cleaned_description")["category"].nunique()
    conflict_descs = desc_cat[desc_cat > 1]
    stats["num_conflicts"] = len(conflict_descs)
    print(f"\nLabel conflicts (same description under >1 category): {len(conflict_descs)}")
    if len(conflict_descs) > 0:
        print("  Examples (up to 5):")
        for d in conflict_descs.index[:5]:
            cats = df[df["cleaned_description"] == d]["category"].unique().tolist()
            print(f'    "{d[:90]}..." -> {cats}')

    stats["top_per_cat"] = {}
    print("\nTop 8 words per category:")
    for cat in sorted(df["category"].unique()):
        cat_texts = df[df["category"] == cat]["cleaned_description"]
        tc = top_words(cat_texts, 8)
        stats["top_per_cat"][cat] = tc
        print(f"  {cat}:")
        for w, c in tc:
            print(f"    {w}: {c}")

    return stats


def plot_category_distribution(df: pd.DataFrame, out_path: Path):
    vc = df["category"].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(vc.index, vc.values, color=sns.color_palette("viridis", len(vc)))
    ax.set_title("Tickets per Category")
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{int(h)}', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_priority_distribution(df: pd.DataFrame, out_path: Path):
    vc = df["priority"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(vc.index, vc.values, color=sns.color_palette("viridis", len(vc)))
    ax.set_title("Tickets per Priority")
    ax.set_xlabel("Priority")
    ax.set_ylabel("Count")
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{int(h)}', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_category_vs_priority(df: pd.DataFrame, out_path: Path):
    ct = pd.crosstab(df["category"], df["priority"])
    fig, ax = plt.subplots(figsize=(10, 6))
    ct.plot(kind="bar", ax=ax, color=sns.color_palette("viridis", len(ct.columns)))
    ax.set_title("Category vs Priority")
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    for container in ax.containers:
        ax.bar_label(container, fontsize=8, padding=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_dataset_table(df: pd.DataFrame, out_path: Path):
    display_df = df[["ticket_id", "cleaned_description", "category", "priority", "status"]].head(10).copy()
    display_df["cleaned_description"] = display_df["cleaned_description"].apply(
        lambda x: x[:60] + "..." if len(x) > 60 else x
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    table = ax.table(cellText=display_df.values, colLabels=display_df.columns,
                     cellLoc="left", loc="center", colColours=["#f0f0f0"] * 5)
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.8)
    ax.set_title("First 10 Rows of tickets.csv", fontsize=12, pad=20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "tickets.csv"
    summary_path = project_root / "data" / "eda_summary.txt"
    screenshots_dir = project_root / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(data_path)

    import io
    old_stdout = sys.stdout
    sys.stdout = mystdout = io.StringIO()

    print("=" * 60)
    print("EDA ON SYNTHETIC TICKET DATASET")
    print("=" * 60)

    stats = print_eda(df)

    print("\n" + "=" * 60)
    print("OBSERVATIONS")
    print("=" * 60)

    cat_counts = df["category"].value_counts()
    balance_ratio = cat_counts.min() / cat_counts.max()
    print(f"1. Class balance: 5 categories with {len(df)} total rows. "
          f"Balance ratio (min/max): {balance_ratio:.2f} - well balanced.")

    ct = pd.crosstab(df["category"], df["priority"])
    print("2. Priority pattern per category: High priority is most common for "
          f"Login Issue ({ct.loc['Login Issue', 'High']} tickets) and "
          f"Application Error ({ct.loc['Application Error', 'High']} tickets), "
          f"while Low priority dominates Report ({ct.loc['Report', 'Low']} tickets) "
          f"and Account Update ({ct.loc['Account Update', 'Low']} tickets).")

    wc = df["cleaned_description"].str.split().str.len()
    print(f"3. Text length: descriptions range from {wc.min()} to {wc.max()} words, "
          f"with a mean of {wc.mean():.1f} and median of {wc.median():.0f} words.")

    print(f"4. Label conflicts: {stats['num_conflicts']} cleaned descriptions appear "
          f"under more than one category, indicating some label noise or ambiguity.")

    print("5. Word frequency: category-specific words dominate each category's top words, "
          "which should help the classifier distinguish categories.")

    print("6. Priority and status are reasonably distributed across all categories, "
          "so no major bias in those fields.")

    sys.stdout = old_stdout
    output = mystdout.getvalue()
    print(output)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"\nSummary saved to {summary_path}")

    print("\nGenerating charts...")
    plot_category_distribution(df, screenshots_dir / "category_distribution.png")
    print("Saved: screenshots/category_distribution.png")
    plot_priority_distribution(df, screenshots_dir / "priority_distribution.png")
    print("Saved: screenshots/priority_distribution.png")
    plot_category_vs_priority(df, screenshots_dir / "category_vs_priority.png")
    print("Saved: screenshots/category_vs_priority.png")
    plot_dataset_table(df, screenshots_dir / "dataset.png")
    print("Saved: screenshots/dataset.png")


if __name__ == "__main__":
    main()