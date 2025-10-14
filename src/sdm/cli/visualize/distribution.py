import os
import pickle
import time
from typing import Optional, Tuple

import click
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from scipy.optimize import minimize_scalar

from sdm.config import *
from sdm.model_wrappers import MODEL_WRAPPERS
from sdm.utils import alpha_skewnorm, compute_recall_at_k

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
K = [10, 20, 50, 100, 200, 500, 1000]


def collect_data(
    model_name: str, document_length: str, dimensionality: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the precomputed cosine similarity scores for the given model and document length.
    """
    relevant_cos_sim_scores = []
    nonrelevant_cos_sim_scores = []

    results_path = os.path.join(
        RESOURCES_FOLDER, "cos_sim_scores", model_name, f"{document_length}.pkl"
    )

    with open(results_path, "rb") as f:
        results = pickle.load(f)

    for _, docid_dict in results["relevant_cos_sim_scores"][dimensionality].items():
        for _, cos_sim in docid_dict.items():
            relevant_cos_sim_scores.append(cos_sim)

    for _, cos_sim_list in results["nonrelevant_cos_sim_scores"][
        dimensionality
    ].items():
        nonrelevant_cos_sim_scores.extend(cos_sim_list)

    return np.array(relevant_cos_sim_scores), np.array(nonrelevant_cos_sim_scores)


def gpd_nll_fixed_beta(
    xi: float, y: np.ndarray, beta: float, eps: float = 1e-12
) -> float:
    """
    Negative log-likelihood for GPD with fixed scale (beta) and shape (xi).
    """
    z = y / beta
    # Safe handling of xi near 0 (exponential limit)
    if abs(xi) < 1e-8:
        return len(y) * np.log(beta) + np.sum(z)  # exact limit as xi -> 0

    t = 1.0 + xi * z
    if np.any(t <= 0.0):
        return 1e20  # large penalty instead of inf

    # Use log1p for stability: log(1 + xi*z)
    s = np.log1p(xi * z)
    return len(y) * np.log(beta) + (1.0 / xi + 1.0) * np.sum(s)


def fit_xi_fixed_beta(
    y: np.ndarray, beta: float, xi_low: Optional[float] = None, xi_high: float = 5.0
):
    """
    Fit xi (shape) for GPD with fixed scale (beta) using MLE.
    """
    y = np.asarray(y, dtype=float)
    # Data-driven lower bound from support
    xi_support_min = -beta / (np.max(y) + 1e-15)
    low = xi_support_min + 1e-9  # nudge inside feasible set
    if xi_low is not None:
        low = max(low, xi_low)
    res = minimize_scalar(
        lambda xi: gpd_nll_fixed_beta(xi, y, beta),
        bounds=(low, xi_high),
        method="bounded",
        options={"xatol": 1e-9, "maxiter": 10_000},
    )
    return res.x


def fit_gpd(excesses: np.ndarray, u: float, params_bulk) -> Tuple[float, float]:
    """
    Fit GPD (SciPy's genpareto) to exceedances (y >= 0), fixing loc=0.
    Returns: xi (shape), beta (scale)
    """
    assert len(excesses) > 0, "No exceedances to fit GPD"

    cdf_bulk_u = st.skewnorm.cdf(u, *params_bulk)
    pdf_bulk_u = st.skewnorm.pdf(u, *params_bulk)
    beta = (1 - cdf_bulk_u) / pdf_bulk_u
    xi = fit_xi_fixed_beta(excesses, beta, xi_low=-1.0)

    return xi, beta


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
def distribution(model_name: str, document_length: str):
    """
    Wrapper function to create line chart visualizing the score distributions.
    """
    model_names = list(MODEL_WRAPPERS.keys()) if model_name == "all" else [model_name]
    document_lengths = (
        ["passage", "document"] if document_length == "both" else [document_length]
    )
    for model_name in model_names:
        for document_length in document_lengths:
            try:
                _distribution(model_name, document_length)
            except Exception as e:
                click.echo(f"Error processing {model_name}, {document_length}: {e}")

    click.echo("Done")


def _distribution(model_name: str, document_length: str):
    """
    Create line chart visualizing the score distributions.
    """
    click.echo(f"Model: {model_name}")
    click.echo(f"Document Length: {document_length}")

    # Load max dim
    _, max_dim = MODEL_WRAPPERS[model_name]

    # Collect data from the model directory
    relevant_cos_sim_scores, nonrelevant_cos_sim_scores = collect_data(
        model_name=model_name,
        document_length=document_length,
        dimensionality=max_dim,
    )

    # Model relevant distribution as normal
    params_r = st.skewnorm.fit(relevant_cos_sim_scores)
    click.echo(f"Relevant distribution: {params_r}")

    # Non-relevant skew-normal distribution
    start = time.time()
    params_n = st.skewnorm.fit(nonrelevant_cos_sim_scores)
    click.echo(
        f"Non-relevant skew-normal fit: {params_n} (time: {time.time() - start:.2f}s)"
    )

    for percentile in [80.0]:
        click.echo(f"Using percentile = {percentile}")

        # Fit EVT-GPD to non-relevant distribution tail

        ## Compute threshold u at the given percentile
        u = float(np.percentile(nonrelevant_cos_sim_scores, percentile))
        excesses = nonrelevant_cos_sim_scores[nonrelevant_cos_sim_scores > u] - u
        click.echo(f"Threshold u = {u}, number of exceedances = {len(excesses)}")

        ## Fit GPD to the excesses
        xi, beta = fit_gpd(excesses, u, params_n)
        click.echo(f"GPD fit: shape={xi}, scale={beta}")

        ## Compute cdf_bulk_u
        cdf_bulk_u = st.skewnorm.cdf(u, *params_n)
        click.echo(f"CDF(u) = {cdf_bulk_u:.4f}")

        def hybrid_pdf(x, params_n):
            if x <= u:
                return st.skewnorm.pdf(x, *params_n)
            else:
                return (1 - cdf_bulk_u) * st.genpareto.pdf(x - u, xi, loc=0, scale=beta)

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

        # Compute expected Recall@k for each corpus size and k
        for corpus_size in DATASETS[document_length]:
            R = 10
            N = corpus_size - R - 100
            click.echo(f"Corpus size: {corpus_size}, R={R}, N={N}")

            for k in K:
                recall_k = compute_recall_at_k(
                    k, params_r, params_n, R, N, alpha_skewnorm
                )
                click.echo(f"  Recall@{k} (skewnorm): {recall_k:.4f}")

            for k in K:
                recall_k = compute_recall_at_k(
                    k, params_r, params_n, R, N, alpha_hybrid
                )
                click.echo(f"  Recall@{k} (hybrid EVT-GPD): {recall_k:.4f}")

        # Non-relevant normal distribution
        start = time.time()
        mu_n, std_n = st.norm.fit(nonrelevant_cos_sim_scores)
        click.echo(
            f"Non-relevant normal fit: mu={mu_n:.4f}, std={std_n:.4f} (time: {time.time() - start:.2f}s)"
        )

        # Plot distributions
        plt.figure(figsize=(6, 4))
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.xlabel("Cosine Similarity", fontsize=14)
        plt.ylabel("Density", fontsize=14, labelpad=10)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlim([mu_n - 3 * std_n, 1.0])
        plt.hist(
            relevant_cos_sim_scores,
            bins=100,
            density=True,
            alpha=0.4,
            color="green",
            label="Relevant",
        )
        plt.hist(
            nonrelevant_cos_sim_scores,
            bins=100,
            density=True,
            alpha=0.4,
            color="orange",
            label="Non-Relevant",
        )
        xs = np.linspace(-0.25, 1.0, 126)
        plt.plot(
            xs, st.skewnorm.pdf(xs, *params_r), "g--", label="Relevant (Skew-Normal)"
        )
        plt.plot(
            xs[xs <= u],
            st.skewnorm.pdf(xs[xs <= u], *params_n),
            "r--",
            label="Bulk (Skew-Normal)",
        )
        plt.plot(
            xs[xs > u],
            [hybrid_pdf(x, params_n) for x in xs[xs > u]],
            "b--",
            label="Tail (Pareto)",
        )
        plt.axvline(u, color="k", linestyle="--", label="Threshold u")
        plt.legend(fontsize=11.5)
        plt.tight_layout()

        plot_path = os.path.join(
            RESOURCES_FOLDER,
            "visualizations",
            model_name,
            document_length,
            str(percentile).replace(".", "_"),
            "score_distributions.png",
        )
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=300)
        plt.close()
