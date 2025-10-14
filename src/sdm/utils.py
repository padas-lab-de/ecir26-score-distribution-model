from collections import defaultdict
from typing import Dict, Tuple

import click
import torch
from datasets import load_dataset
from scipy import stats as st

CoRE = {
    "passage": {
        "pass_core": 10_000,
        "pass_10k": 10_000,
        # "pass_100k": 100_000,
        # "pass_1M": 1_000_000,
        # "pass_10M": 10_000_000,
        # "pass_100M": 100_000_000,
    },
    "document": {
        "doc_core": 10_000,
        "doc_10k": 10_000,
        # "doc_100k": 100_000,
        # "doc_1M": 1_000_000,
        # "doc_10M": 10_000_000,
    },
}


def cos_sim(a, b):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.

    Return:
        Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a).float()

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b).float()

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def load_data(
    dataset_sub_corpus: str,
) -> Tuple[
    defaultdict, Dict[str, str], Dict[str, Dict[str, str]], Dict[str, Dict[str, int]]
]:
    """
    Loads the corpus, queries and qrels of the given CoRE dataset (passage or document) from the corresponding
    huggingface repo for all specified dataset sizes, i.e. 10k, 100k, 1M etc.

    Args:
        dataset_sub_corpus: The CoRE dataset to load, i.e., passage or document.

    Returns:
        The loaded corpus, queries, qrels and relevant qrels.
    """
    # Load queries dataset
    dataset_queries = load_dataset("PaDaS-Lab/CoRE", "queries")["test"]

    # Load the qrels dataset
    dataset_qrels = load_dataset("PaDaS-Lab/CoRE", "qrels")[dataset_sub_corpus]

    # Transform the datasets
    qrels = {}
    for q in dataset_qrels:
        query_id = q["query-id"]
        corpus_id = q["corpus-id"]
        qrels[query_id] = qrels.get(query_id, {})
        qrels[query_id][corpus_id] = q["type"]

    queries = {q["_id"]: q["text"] for q in dataset_queries if q["_id"] in qrels.keys()}
    click.echo(f"Loaded {len(queries)} queries")

    # Load the corpus datasets
    datasets_corpus = {}
    for split_name in CoRE[dataset_sub_corpus]:
        dataset_corpus = load_dataset("PaDaS-Lab/CoRE", "corpus")[split_name]
        datasets_corpus[split_name] = dataset_corpus

    # Transform the corpus datasets
    corpora = defaultdict(dict)
    for split_name, dataset_corpus in datasets_corpus.items():
        for d in dataset_corpus:
            corpora[CoRE[dataset_sub_corpus][split_name]][d["_id"]] = {
                "title": d["title"],
                "text": d["text"],
            }
    for corpus_size in corpora:
        click.echo(
            f"Loaded {len(corpora[corpus_size])} documents for corpus of size {corpus_size}"
        )

    # Simplify qrels
    qrels_relevant_only = {}
    for qid in qrels:
        qrels_relevant_only[qid] = {}
        for docid in qrels[qid]:
            if qrels[qid][docid] == "relevant":
                qrels_relevant_only[qid][docid] = 1

    return corpora, queries, qrels, qrels_relevant_only


def alpha_skewnorm(params: tuple, tau: float) -> float:
    """
    Tail probability P[S >= tau] for S ~ SkewNormal(a, loc, scale).
    """
    a, loc, scale = params
    assert scale > 0.0
    return 1.0 - st.skewnorm.cdf(tau, a, loc=loc, scale=scale)


def compute_recall_at_k(
    k: int,
    params_r: tuple,
    params_n: tuple,
    R: int,
    N: int,
    fun_alpha,
    tol: float = 1e-10,
    max_iter: int = 100,
):
    """
    Numerically exact Recall@k by solving:
        R * alpha_r(tau_k) + N * alpha_n(tau_k) = k
    then Recall@k = alpha_r(tau_k).
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
        gm = g(mid)
        if gm > 0:
            lo = mid
        else:
            hi = mid
        if abs(gm) <= tol:
            break
    tau_k = 0.5 * (lo + hi)
    return alpha_skewnorm(params_r, tau_k)
