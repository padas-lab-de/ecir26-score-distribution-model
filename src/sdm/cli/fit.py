import os
import pickle
import time
from typing import Dict, List, Tuple

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.optimize import minimize, minimize_scalar

from sdm.config import *

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
DIMENSIONALITY = 768
K = [10, 20, 50, 100, 200, 500, 1000]


def collect_data(
    model_name: str, document_length: str, dimensionality: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the measured data for the given model.

    Args:
        model_name (str): The name of the model.
        document_length (str): The document length to filter results.
    """
    nonrelevant_cos_sim_scores = []
    relevant_cos_sim_scores = []

    results_path = os.path.join(
        RESULTS_FOLDER,
        model_name,
        document_length,
        "results.pkl",
    )

    with open(results_path, "rb") as f:
        results = pickle.load(f)

    for _, docid_dict in results["relevant_cos_sim_scores"][dimensionality].items():
        for _, cos_sim in docid_dict.items():
            # relevant_cos_sim_scores[qid][docid] = cos_sim
            relevant_cos_sim_scores.append(cos_sim)

    for _, cos_sim_list in results["random_cos_sim_scores"][dimensionality].items():
        nonrelevant_cos_sim_scores.extend(cos_sim_list)

    return np.array(relevant_cos_sim_scores), np.array(nonrelevant_cos_sim_scores)


def gpd_nll_fixed_beta(xi, y, beta, eps=1e-12):
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


def fit_xi_fixed_beta(y, beta, xi_low=None, xi_high=5.0):
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


def alpha_skewnorm(params, tau):
    """
    Tail probability P[S >= tau] for S ~ SkewNormal(a, loc, scale).
    """
    a, loc, scale = params
    assert scale > 0.0
    return 1.0 - st.skewnorm.cdf(tau, a, loc=loc, scale=scale)


def compute_recall_at_k(
    k, params_r, params_n, R, N, fun_alpha, tol=1e-10, max_iter=100
):
    """
    Numerically exact Recall@k by solving:
        R * alpha_r(tau_k) + N * alpha_n(tau_k) = k
    then Recall@k = alpha_r(tau_k).
    `method` can be "newton" (fast) or "bisection" (robust).
    """
    assert k > 0 and R > 0 and N > 0

    # Define g(tau) = expected count >= tau - k
    def g(tau):
        return R * alpha_skewnorm(params_r, tau) + N * fun_alpha(params_n, tau) - k

    # Bracket where g(lo) >= 0 and g(hi) <= 0
    lo = -0.7127793351188958
    hi = 1.912777246955496
    glo, ghi = g(lo), g(hi)

    # If k is extreme, clamp
    if (
        glo <= 0
    ):  # expected count at very low threshold already < k -> tau must be even lower
        tau_k = lo
        return alpha_skewnorm(params_r, tau_k)
    if (
        ghi >= 0
    ):  # expected count at very high threshold already > k -> tau must be even higher
        tau_k = hi
        return alpha_skewnorm(params_r, tau_k)

    # Robust bisection
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        # print(f"mid={mid}; g(mid)={g(mid)}; lo={lo}, hi={hi}")
        gm = g(mid)
        if gm > 0:
            lo = mid
        else:
            hi = mid
        if abs(gm) <= tol:
            break
    tau_k = 0.5 * (lo + hi)
    return alpha_skewnorm(params_r, tau_k)


@click.command()
@click.argument("model_name", type=str)
@click.argument("document_length", type=click.Choice(["passage", "document"]))
def evt_core(model_name: str, document_length: str):
    """
    Fit score distributions using Extreme Value Theory (EVT).
    Compute expected Recall@k analytically based on the fitted distributions.
    """
    click.echo(
        f"Fitting score distributions using EVT for {model_name} and {document_length}..."
    )

    # Collect data from the model directory
    relevant_cos_sim_scores, nonrelevant_cos_sim_scores = collect_data(
        model_name=model_name,
        document_length=document_length,
        dimensionality=DIMENSIONALITY,
    )

    # Model relevant distribution as normal
    params_r = st.skewnorm.fit(relevant_cos_sim_scores)
    click.echo(f"Relevant distribution: {params_r}")

    # Non-relevant skew-normal distribution
    # params_n = (1.9481929392131, -0.06781844355611863, 0.09664460845810896)  # Passage 10k
    # params_n = (1.7269847585802358, -0.061246023018398314, 0.09183822048300241)  # Passage 1M
    params_n = (
        1.872776999852325,
        -0.08486812316555292,
        0.09372104622741906,
    )  # Document 10k
    # start = time.time()
    # params_n = st.skewnorm.fit(nonrelevant_cos_sim_scores)
    # click.echo(f"Non-relevant skew-normal fit: {params_n} (time: {time.time() - start:.2f}s)")

    # for percentile in [75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0]:
    for percentile in [80.0]:
        # for percentile in [80.0, 82.0, 84.0, 86.0, 88.0, 90.0, 92.0, 94.0, 96.0, 98.0]:
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

        # Plot hybrid PDF
        xs = np.linspace(-0.25, 1.0, 126)
        pdf_vals = [hybrid_pdf(x, params_n) for x in xs]
        cdf_vals = [hybrid_cdf(x, params_n) for x in xs]

        # Non-relevant normal distribution
        start = time.time()
        mu_n, std_n = st.norm.fit(nonrelevant_cos_sim_scores)
        click.echo(
            f"Non-relevant normal fit: mu={mu_n:.4f}, std={std_n:.4f} (time: {time.time() - start:.2f}s)"
        )

        plt.figure(figsize=(6, 4))
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.xlabel("Cosine Similarity", fontsize=14)
        plt.ylabel("Density", fontsize=14, labelpad=10)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlim([-0.25, 1])
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
        plt.plot(
            xs, st.skewnorm.pdf(xs, *params_r), "g--", label="Relevant (Skew-Normal)"
        )
        # plt.plot(xs, pdf_vals, 'g--', label="Hybrid EVT-GPD")
        # plt.plot(xs, cdf_vals, 'g:', label="Hybrid CDF")
        # plt.plot(xs, st.norm.pdf(xs, mu_n, std_n), 'r--', label="Non-relevant Normal fit")
        # plt.plot(xs, skewt.pdf(xs, *params_skewt), 'm--', label="Non-relevant Skew-t fit")
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
        # plt.title("Score Distribution of Non-Relevant Documents with EVT tail splicing")
        plt.tight_layout()

        plot_path = os.path.join(
            RESOURCES_FOLDER,
            "evt",
            model_name,
            document_length,
            "evt_core_hybrid_distribution.png",
        )
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=300)
        plt.savefig(plot_path.replace(".png", ".pdf"))
        plt.close()

    click.echo("Done")
