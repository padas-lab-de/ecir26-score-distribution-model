import os
import pickle
from collections import defaultdict
from typing import Any, Dict, List, Optional

import click
import torch
from tqdm import tqdm

from sdm.config import *
from sdm.model_wrappers import MODEL_WRAPPERS, AbstractModelWrapper, get_model_wrapper
from sdm.utils import cos_sim, load_data


def _load_embeddings(save_path: str) -> Optional[torch.Tensor]:
    """
    Attempts to load query or corpus embeddings at the given path and returns the on success.

    Args:
        save_path: The path where the embeddings are stored.
        batch_num: The number of the batch when loading corpus embeddings.
        queries: Whether to load query or corpus embeddings.

    Returns:
        A boolean indicating whether loading was successful along with the embeddings.
    """
    if save_path:
        try:
            embeddings = torch.load(save_path, weights_only=False)
            return embeddings
        except OSError:
            click.echo(f"Could not find any embeddings at {save_path}")
    return None


def _encode_queries(
    save_path: str,
    queries: List[str],
    model: AbstractModelWrapper,
    encode_kwargs: Dict[str, Any] = {},
) -> torch.Tensor:
    """
    Encodes the given queries and stores them as pytorch tensors.

    Args:
        save_path: The path where the embeddings are stored.
        queries: The queries to encode.
        model: The model used to encode the queries.
        encode_kwargs: The keyword arguments passed to the encoding function.

    Returns:
        The generated query embeddings.
    """
    query_embeddings = model.encode_queries(
        queries,
        **encode_kwargs,
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(query_embeddings, save_path)

    return query_embeddings


def _encode_corpus(
    save_path: str,
    corpus: List[Dict[str, str]],
    model: AbstractModelWrapper,
    encode_kwargs: Dict[str, Any] = {},
) -> torch.Tensor:
    """
    Encodes the given corpus documents and stores them as pytorch tensors.

    Args:
        save_path: The path where the embeddings are stored.
        corpus: The documents to encode.
        model: The model used to encode the queries.
        encode_kwargs: The keyword arguments passed to the encoding function.
    """
    sub_corpus_embeddings = model.encode_corpus(
        corpus,
        **encode_kwargs,
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(
            sub_corpus_embeddings,
            save_path,
        )

    return sub_corpus_embeddings


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
def compute_score_distributions(model_name: str, document_length: str):
    """
    Wrapper function to compute score distributions for the given model and document length.
    """
    model_names = list(MODEL_WRAPPERS.keys()) if model_name == "all" else [model_name]
    document_lengths = (
        ["passage", "document"] if document_length == "both" else [document_length]
    )
    for model_name in model_names:
        for document_length in document_lengths:
            try:
                _compute_score_distributions(model_name, document_length)
            except Exception as e:
                click.echo(
                    f"Error computing score distributions for {model_name}, {document_length}: {e}"
                )

    click.echo("Done")


def _compute_score_distributions(model_name: str, document_length: str):
    """
    Compute score distributions for the given model and document length.
    """
    click.echo("Computing score distributions")
    click.echo(f"Model: {model_name}")
    click.echo(f"Document Length: {document_length}")

    # Initialize save path
    save_path = os.path.join(EMBEDDINGS_FOLDER, model_name, document_length)

    # Load embedding model
    model, max_dim = get_model_wrapper(model_name)
    click.echo(f"Loaded model: {model_name} with max dim {max_dim}")

    corpora, queries, qrels = load_data(document_length)
    click.echo(f"Loaded {len(queries)} queries and {len(corpora)} corpora")

    # Initialize results dict
    dimensionalities = [
        d for d in sorted(DIMENSIONALITIES, reverse=True) if d <= max_dim
    ]
    query_ids = list(queries.keys())

    # Load relevant and distractor documents
    relevant_corpus_ids = {}
    distractor_corpus_ids = {}
    for qid in query_ids:
        relevant_corpus_ids[qid] = []
        distractor_corpus_ids[qid] = []
        for docid, _type in qrels[qid].items():
            if _type == "relevant":
                relevant_corpus_ids[qid].append(docid)
            elif _type == "distractor":
                distractor_corpus_ids[qid].append(docid)

    # Embed queries or load saved embeddings
    queries_save_path = os.path.join(save_path, "queries.pt")
    query_embeddings = _load_embeddings(queries_save_path)

    if query_embeddings is not None:
        click.echo(f"Loaded query embeddings at {queries_save_path}")
    else:
        queries = [queries[qid] for qid in queries]
        query_embeddings = _encode_queries(queries_save_path, queries, model)

    # Initialize dict with query IDs and embeddings
    query_id_embeddings = {}
    for query_id, query_embedding in zip(query_ids, query_embeddings):
        query_id_embeddings[query_id] = query_embedding

    # Initialize cos_sim dicts
    relevant_cos_sim_scores = {}
    nonrelevant_cos_sim_scores = {}
    for dimensionality in dimensionalities:
        relevant_cos_sim_scores[dimensionality] = defaultdict(dict)
        nonrelevant_cos_sim_scores[dimensionality] = defaultdict(list)

    # Compute scores for smallest corpus size: 10,000
    click.echo("Evaluating corpus of size 10,000")
    corpus_size = 10_000

    # Embed corpus
    corpus = corpora[corpus_size]
    corpus_ids = list(corpus.keys())
    corpus_ids = sorted(corpus_ids)
    corpus = [corpus[cid] for cid in corpus_ids]

    # Iterator over batches
    iterator = range(0, len(corpus), CORPUS_CHUNK_SIZE)

    # Encoding corpus in batches... Warning: This might take a while!
    for batch_num, corpus_start_idx in enumerate(iterator):
        batch_save_path = os.path.join(
            save_path, str(corpus_size), f"corpus_batch_{batch_num}.pt"
        )
        corpus_end_idx = min(corpus_start_idx + CORPUS_CHUNK_SIZE, len(corpus))

        sub_corpus_embeddings = _load_embeddings(batch_save_path)

        if sub_corpus_embeddings is not None:
            click.echo(
                f"Loaded corpus embeddings for batch {batch_num + 1}/{len(iterator)} at {batch_save_path}"
            )
        else:
            click.echo(f"Encoding Batch {batch_num + 1}/{len(iterator)}...")

            # Encode chunk of corpus
            sub_corpus_embeddings = _encode_corpus(
                batch_save_path,
                corpus[corpus_start_idx:corpus_end_idx],
                model,
            )

        click.echo("Computing cosine similarity scores")
        batch_corpus_ids = corpus_ids[corpus_start_idx:corpus_end_idx]
        for qid in tqdm(query_ids):
            for dimensionality in dimensionalities:
                query_embedding = query_id_embeddings[qid][:dimensionality]
                doc_embeddings = sub_corpus_embeddings[:, :dimensionality]

                # Compute cosine similarity
                cos_sim_scores = cos_sim(query_embedding, doc_embeddings)[0]

                for batch_index, docid in enumerate(batch_corpus_ids):
                    if docid in relevant_corpus_ids[qid]:
                        relevant_cos_sim_scores[dimensionality][qid][docid] = (
                            cos_sim_scores[batch_index].item()
                        )
                    elif docid not in distractor_corpus_ids[qid]:
                        nonrelevant_cos_sim_scores[dimensionality][qid].append(
                            cos_sim_scores[batch_index].item()
                        )

    # Clear CUDA cache
    torch.cuda.empty_cache()

    # Save results
    pickle_save_path = os.path.join(
        RESOURCES_FOLDER, "cos_sim_scores", model_name, f"{document_length}.pkl"
    )
    os.makedirs(os.path.dirname(pickle_save_path), exist_ok=True)
    results = {
        "relevant_cos_sim_scores": relevant_cos_sim_scores,
        "nonrelevant_cos_sim_scores": nonrelevant_cos_sim_scores,
    }
    click.echo(f"Saving results to {pickle_save_path}")
    with open(pickle_save_path, "wb") as f:
        pickle.dump(results, f)
