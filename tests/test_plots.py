from pathlib import Path

import pandas as pd

from plots import format_metric_name, generate_plots, truncate

METRICS = {
    "answer_relevancy": {"inputs": ("user_input", "response")},
    "noise_sensitivity": {
        "inputs": ("user_input", "response"),
        "lower_is_better": True,
    },
}


def sample_dataframe():
    return pd.DataFrame(
        {
            "user_input": ["What is RAG?", "What is Chroma?"],
            "response": ["answer one", "answer two"],
            "retrieved_contexts": [["ctx one"], ["ctx two"]],
            "reference": ["ref one", "ref two"],
            "answer_relevancy": [0.9, 0.7],
            "noise_sensitivity": [0.1, 0.4],
        }
    )


def test_generate_plots_creates_directory_and_files(tmp_path):
    output_dir = tmp_path / "plots"

    paths = generate_plots(sample_dataframe(), METRICS, output_dir)

    assert output_dir.is_dir()
    assert len(paths) == 3
    assert {Path(path).name for path in paths} == {
        "average_metric_scores.png",
        "metric_scores_by_question.png",
        "metric_score_distributions.png",
    }
    expected = {
        "average_metric_scores.png",
        "metric_scores_by_question.png",
        "metric_score_distributions.png",
    }
    assert all((output_dir / name).stat().st_size > 0 for name in expected)


def test_generate_plots_empty_dataframe_returns_empty(tmp_path):
    output_dir = tmp_path / "plots"

    assert generate_plots(sample_dataframe().iloc[0:0], METRICS, output_dir) == []
    assert not output_dir.exists()


def test_metric_helpers():
    assert format_metric_name("answer_relevancy") == "answer relevancy"
    assert truncate("short") == "short"
    assert truncate("x" * 70) == "x" * 57 + "..."
