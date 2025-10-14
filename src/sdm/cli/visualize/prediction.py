import json
import os
from collections import defaultdict
from typing import List

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st

from sdm.config import *
from sdm.model_wrappers import MODEL_WRAPPERS
from sdm.utils import compute_recall_at_k

DATASETS = {
    "passage": [
        10_000,
        100_000,
        1_000_000,
        10_000_000,
        100_000_000,
    ],
    "document": [
        10_000,
        100_000,
        1_000_000,
        10_000_000,
    ],
}
K = list(range(10, 1000 + 1, 10))
COLORS = [
    plt.cm.Set1.colors[0],
    plt.cm.Set1.colors[1],
    plt.cm.Set1.colors[2],
    plt.cm.Set1.colors[3],
    plt.cm.Set1.colors[4],
    plt.cm.Set1.colors[6],
]
LABELS = {
    10_000: "10K",
    100_000: "100K",
    1_000_000: "1M",
    10_000_000: "10M",
    100_000_000: "100M",
}


def get_recall_wo_distractors(rankings: dict, num_relevant_docs: int = 10) -> dict:
    """
    Calculate recall at k without considering distractors.
    """
    recall_values = defaultdict(list)

    for _, ranking in rankings.items():
        relevant = ranking["relevant"]
        distractor = ranking["distractor"]

        rank_list = ["random"] * 1_000
        for _, rank in relevant.items():
            if rank <= 1_000:
                rank_list[rank - 1] = "relevant"
        for _, rank in distractor.items():
            if rank <= 1_000:
                rank_list[rank - 1] = "distractor"

        rank_list_wo_distractors = [r for r in rank_list if r != "distractor"]
        for k in K:
            recall_values[k].append(
                sum(
                    1
                    for i, r in enumerate(rank_list_wo_distractors)
                    if r == "relevant" and i <= k
                )
                / num_relevant_docs
            )

    return {k: sum(v) / len(v) for k, v in recall_values.items()}


def collect_data(
    model_name: str, document_length: str, corpus_sizes: List[int]
) -> dict:
    """
    Load the measured data for the given model.

    Args:
        model_name (str): The name of the model.
        document_length (str): The document length to filter results.

    Returns:
        list: A list of tuples containing (k, n, recall_at_k).
    """
    _, max_dim = MODEL_WRAPPERS[model_name]

    results_path = os.path.join(
        RESULTS_FOLDER,
        model_name,
        document_length,
        f"dim={max_dim}",
    )

    results = {}

    for corpus_size in corpus_sizes:
        file_path = os.path.join(results_path, f"{corpus_size}.json")

        if not os.path.exists(file_path):
            continue

        with open(file_path, "r") as f:
            data = json.load(f)
            rankings = data["rankings"]

        results[corpus_size] = get_recall_wo_distractors(rankings)

    return results


def plot_lines(
    data_dict: dict,
    ax: plt.Axes,
    document_length: str,
):
    for i, (corpus_size, data) in enumerate(data_dict.items()):
        df = pd.DataFrame(data.items(), columns=["k", "value"])

        ax.plot(df["k"], df["value"], label=LABELS[corpus_size], color=COLORS[i])

        if document_length == "passage":
            params_r = (-1.1354841658575139, 0.5999989559183001, 0.1640972863796495)
            params_n = (1.9481929392131, -0.06781844355611863, 0.09664460845810896)
            R = 10
            N = corpus_size - R - 100
            u = 0.054019863903522494
            xi = -0.09189467458446031
            beta = 0.055916698480706115
            cdf_bulk_u = st.skewnorm.cdf(u, *params_n)

        elif document_length == "document":
            params_r = (-0.8149857601520807, 0.43847698142456465, 0.20881832344587004)
            params_n = (1.872776999852325, -0.08486812316555292, 0.09372104622741906)
            R = 10
            N = corpus_size - R - 100
            u = 0.03376907110214234
            xi = -0.10099099932515726
            beta = 0.05416256408068892
            cdf_bulk_u = st.skewnorm.cdf(u, *params_n)

        else:
            raise ValueError(f"Unknown document_length: {document_length}")

        def hybrid_cdf(x, params_n):
            if x <= u:
                return st.skewnorm.cdf(x, *params_n)
            else:
                return cdf_bulk_u + (1 - cdf_bulk_u) * st.genpareto.cdf(
                    x - u, xi, loc=0, scale=beta
                )

        def alpha_hybrid(params, tau):
            """
            Tail probability P[S >= tau] for hybrid distribution with CDF fun_hybrid_cdf.
            """
            return 1.0 - hybrid_cdf(tau, params)

        expected_recalls = [
            compute_recall_at_k(k, params_r, params_n, R, N, alpha_hybrid)
            for k in df["k"]
        ]
        ax.plot(
            df["k"],
            expected_recalls,
            linestyle="--",
            color=COLORS[i],
            alpha=0.7,
        )

    if document_length == "passage":
        ax.set_yticks(np.arange(0.4, 1.1, 0.1))
        ax.legend(
            title="Corpus Size",
            loc="lower right",
            fontsize=9,
            ncol=3,
            title_fontsize=10,
        )
    else:
        ax.set_yticks(np.arange(0.2, 1.0, 0.1))
        ax.set_xlabel("k (Number of Retrieved Documents)", fontsize=11)
        ax.legend(["Empirical", "Predicted"], loc="lower right", fontsize=10)

    ax.set_title(document_length.capitalize(), fontsize=11, fontweight="bold", pad=4)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xticks(
        [0, 10] + list(range(100, 1001, 100)),
        ["", 10] + list(range(100, 1001, 100)),
    )
    ax.set_xlim([-15, 1015])
    ax.set_ylabel("Recall@k", fontsize=11)


@click.command()
@click.argument(
    "model_name",
    type=click.Choice(["all"] + list(MODEL_WRAPPERS.keys())),
    default="all",
)
@click.argument(
    "document_length",
    type=click.Choice(["both", "passage", "document"]),
    default="both",
)
def prediction(model_name: str, document_length: str):
    """
    Wrapper command to visualize the prediction results of the models.
    """
    model_names = list(MODEL_WRAPPERS.keys()) if model_name == "all" else [model_name]
    document_lengths = (
        ["passage", "document"] if document_length == "both" else [document_length]
    )
    for model_name in model_names:
        for document_length in document_lengths:
            try:
                _prediction(model_name, document_length)
            except Exception as e:
                click.echo(f"Error processing {model_name}, {document_length}: {e}")

    click.echo("Done")


def _prediction(model_name: str, document_length: str):
    """
    Visualize the relevance composition of the model's results.
    """
    plot_path = os.path.join(
        RESOURCES_FOLDER, "visualizations", model_name, "recall.png"
    )
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    _, axes = plt.subplots(2, 1, figsize=(4.5, 5.5), sharex=True)

    for ax, (document_length, corpus_sizes) in zip(axes, DATASETS.items()):

        # Collect data from the model directory
        data = collect_data(model_name, document_length, corpus_sizes)

        # Plot the line chart
        plot_lines(data, ax, document_length)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
