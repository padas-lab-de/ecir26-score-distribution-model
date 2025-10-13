import torch
from collections import defaultdict
from datasets import load_dataset
from typing import Dict, Tuple
import click


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
    defaultdict, Dict[str, str], Dict[str, Dict[str, str]]
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

    return corpora, queries, qrels

