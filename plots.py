import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

PLOTS_DIR = "plots"

HIGHER_IS_BETTER = "#0f766e"
LOWER_IS_BETTER = "#d97706"


def format_metric_name(name):
    return name.replace("_", " ")


def truncate(text, limit=60):
    text = str(text)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def save_figure(fig, output_dir, filename):
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_average_scores(df, metrics, names, output_dir):
    averages = df[names].mean()
    colors = [
        LOWER_IS_BETTER if metrics[name].get("lower_is_better") else HIGHER_IS_BETTER
        for name in names
    ]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(names) + 1)))
    bars = ax.barh([format_metric_name(name) for name in names], averages, color=colors)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Average score")
    ax.set_title("Average score per metric")
    ax.grid(axis="x", alpha=0.3)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=HIGHER_IS_BETTER),
        plt.Rectangle((0, 0), 1, 1, color=LOWER_IS_BETTER),
    ]
    ax.legend(
        handles,
        ["Higher is better", "Lower is better"],
        loc="lower right",
        frameon=False,
    )

    return save_figure(fig, output_dir, "average_metric_scores.png")


def plot_metric_heatmap(df, metrics, names, output_dir):
    scores = df[names].to_numpy()
    labels = [format_metric_name(name) for name in names]
    questions = [truncate(question, 55) for question in df["user_input"]]

    fig, ax = plt.subplots(figsize=(max(10, 0.8 * len(names) + 3), max(4, 0.45 * len(df) + 2)))
    image = ax.imshow(scores, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(df)), labels=questions)
    ax.set_title("Metric scores by question")

    for row in range(scores.shape[0]):
        for column in range(scores.shape[1]):
            value = scores[row, column]
            color = "white" if value > 0.6 else "black"
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=7,
            )

    fig.colorbar(image, ax=ax, label="Score")
    return save_figure(fig, output_dir, "metric_scores_by_question.png")


def plot_metric_distributions(df, metrics, names, output_dir):
    values = [df[name].to_numpy() for name in names]
    labels = [format_metric_name(name) for name in names]
    colors = [
        LOWER_IS_BETTER if metrics[name].get("lower_is_better") else HIGHER_IS_BETTER
        for name in names
    ]

    fig, ax = plt.subplots(figsize=(max(9, 0.8 * len(names) + 3), 5))
    boxes = ax.boxplot(
        values,
        tick_labels=labels,
        patch_artist=True,
        medianprops={"color": "black"},
    )

    for box, color in zip(boxes["boxes"], colors, strict=True):
        box.set_facecolor(color)
        box.set_alpha(0.7)

    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Score distribution per metric")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    return save_figure(fig, output_dir, "metric_score_distributions.png")


def generate_plots(df, metrics, output_dir=PLOTS_DIR):
    if df.empty:
        return []

    os.makedirs(output_dir, exist_ok=True)
    names = list(metrics)

    return [
        plot_average_scores(df, metrics, names, output_dir),
        plot_metric_heatmap(df, metrics, names, output_dir),
        plot_metric_distributions(df, metrics, names, output_dir),
    ]
