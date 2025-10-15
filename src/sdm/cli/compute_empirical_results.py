import heapq
import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union

import click
import pytrec_eval
import torch

from sdm.cli.compute_score_distributions import _encode_corpus, _encode_queries
from sdm.config import *
from sdm.model_wrappers import MODEL_WRAPPERS, AbstractModelWrapper, get_model_wrapper
from sdm.utils import cos_sim, load_data


def _get_rankings(
    results: dict,
    qrels_rel: Dict[str, Dict[str, int]],
    qrels: Dict[str, Dict[str, str]],
) -> Dict[Any, Dict[str, Dict[str, int]]]:
    """
    Stores the position of all relevant, random and distractor documents retrieved in the top-k results for a certain
    query in a dictionary.

    Args:
        results: Dictionary containing the results of the top-k retrieval.
        qrels_rel: The qrels containing only relevant documents.
        qrels: The qrels that also contain information on distractors.

    Returns:
        The constructed dictionary.
    """
    rankings = {}
    for qid in results:
        rankings[qid] = {
            "relevant": {},
            "distractor": {},
            "random": {},
        }
        scores = {}

        for cid in results[qid]:
            scores[cid] = results[qid][cid]

        scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))

        for idx, cid in enumerate(scores):
            if cid in qrels_rel[qid] and qrels_rel[qid][cid] > 0:
                rankings[qid]["relevant"][cid] = idx + 1
            elif cid in qrels[qid] and qrels[qid][cid] == "distractor":
                rankings[qid]["distractor"][cid] = idx + 1
            else:
                rankings[qid]["random"][cid] = idx + 1

    return rankings


def _evaluate_results(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    k_values: List[int],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float],]:
    """
    Evaluates the results using pytrec_eval and computes NDCG, MAP, Recall and Precision at different k values.
    """
    all_ndcgs, all_aps, all_recalls, all_precisions = {}, {}, {}, {}

    for k in k_values:
        all_ndcgs[f"NDCG@{k}"] = []
        all_aps[f"MAP@{k}"] = []
        all_recalls[f"Recall@{k}"] = []
        all_precisions[f"P@{k}"] = []

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {map_string, ndcg_string, recall_string, precision_string}
    )
    scores = evaluator.evaluate(results)

    for query_id in scores.keys():
        for k in k_values:
            all_ndcgs[f"NDCG@{k}"].append(scores[query_id]["ndcg_cut_" + str(k)])
            all_aps[f"MAP@{k}"].append(scores[query_id]["map_cut_" + str(k)])
            all_recalls[f"Recall@{k}"].append(scores[query_id]["recall_" + str(k)])
            all_precisions[f"P@{k}"].append(scores[query_id]["P_" + str(k)])

    ndcg, _map, recall, precision = (
        all_ndcgs.copy(),
        all_aps.copy(),
        all_recalls.copy(),
        all_precisions.copy(),
    )

    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(sum(ndcg[f"NDCG@{k}"]) / len(scores), 5)
        _map[f"MAP@{k}"] = round(sum(_map[f"MAP@{k}"]) / len(scores), 5)
        recall[f"Recall@{k}"] = round(sum(recall[f"Recall@{k}"]) / len(scores), 5)
        precision[f"P@{k}"] = round(sum(precision[f"P@{k}"]) / len(scores), 5)

    return ndcg, _map, recall, precision


def _compute_metrics(
    k_values: List[int],
    qrels_relevant_only: Dict[str, Dict[str, int]],
    qrels: Dict[str, Dict[str, str]],
    results: dict,
) -> Dict[str, Any]:
    """
    Calculates retrieval metrics at different k values for the given results.

    Args:
        k_values: The k values for which to compute the metrics.
        qrels_relevant_only: The qrels containing only relevant documents.
        qrels: The qrels that also contain information on distractors.
        results: The retrieval results per query.

    Returns:
        A dictionary containing the calculated metrics.
    """
    ndcg, _map, recall, precision = _evaluate_results(
        qrels_relevant_only.copy(),
        results,
        k_values,
    )

    scores: Dict[str, Any] = {
        **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
        **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
        **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
        **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
    }

    rankings = _get_rankings(results.copy(), qrels_relevant_only.copy(), qrels.copy())
    scores["rankings"] = rankings

    return scores


def _eval_results(
    dimensionalities: List[int],
    k_values: List[int],
    corpus_size: str,
    save_path: str,
    qrels_relevant_only: Dict[str, Dict[str, int]],
    qrels: Dict[str, Dict[str, str]],
    results: dict,
    max_dim: int,
):
    """
    Evaluates dense retrieval performance across different embedding vector lengths. The results
    are saved in a JSON file.

    Args:
        dimensionalities: The different embedding vector lengths.
        k_values: A list of k values for which metrics should be computed.
        corpus_size: The corpus size.
        save_path: The directory to save the results.
        qrels_relevant_only: The qrels containing only relevant documents.
        qrels: The qrels that also contain information on distractors.
        results: The results of the retrieval.
        max_dim: The model's maximum dimensionality.
    """
    for dimensionality in dimensionalities:
        # Evaluate the results
        scores = _compute_metrics(
            k_values,
            qrels_relevant_only,
            qrels,
            results[dimensionality],
        )
        click.echo(
            f"NDCG@10: {scores['ndcg_at_10']} [Dimensionality: {dimensionality}]"
        )
        dim_dir = max_dim if dimensionality > max_dim else dimensionality

        # Save the results
        results_path = os.path.join(
            save_path,
            f"dim={dim_dir}",
            f"{corpus_size}.json",
        )
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(scores, f, indent=4)


def _search(
    model: AbstractModelWrapper,
    corpora: Dict[str, Dict[str, Dict[str, str]]],
    queries: Dict[str, str],
    top_k: int,
    dimensionalities: List[int],
    save_path: str = EMBEDDINGS_FOLDER,
    encode_kwargs: Dict[str, Any] = {},
) -> Dict[Union[str, int], Dict[int, Dict[str, Dict[str, Dict[str, float]]]]]:
    """
    Search for the top-k documents for each query in the corpus.
    """
    # Initialize results dict
    query_ids = list(queries.keys())
    corpus_sizes = sorted(corpora.keys())
    results = defaultdict(dict)
    result_heaps = {}
    dimensionalities = sorted(dimensionalities, reverse=True)
    for dimensionality in dimensionalities:
        # Initialize one heaps dict for all corpus sizes
        result_heaps[dimensionality] = {qid: [] for qid in query_ids}

        for corpus_size in corpus_sizes:
            results[corpus_size][dimensionality] = {qid: {} for qid in query_ids}

    # Embed queries or load saved embeddings:
    queries_save_path = os.path.join(save_path, "queries.pt")

    if not os.path.exists(queries_save_path):
        _queries = [queries[qid] for qid in queries]
        query_embeddings = _encode_queries(
            queries_save_path, _queries, model, encode_kwargs
        )
    else:
        query_embeddings = torch.load(queries_save_path, weights_only=False)
        click.echo(f"Loaded query embeddings at {queries_save_path}")

    # Loop over corpora
    for corpus_size in corpus_sizes:
        click.echo(f"Evaluating corpus of size {corpus_size}...")

        # Embed corpus
        corpus = corpora[corpus_size]
        corpus_ids = list(corpus.keys())
        corpus_ids = sorted(corpus_ids)
        corpus = [corpus[cid] for cid in corpus_ids]

        # Encoding corpus in batches... Warning: This might take a while!
        iterator = range(0, len(corpus), CORPUS_CHUNK_SIZE)

        for batch_num, corpus_start_idx in enumerate(iterator):
            batch_save_path = os.path.join(
                save_path, str(corpus_size), f"corpus_batch_{batch_num}.pt"
            )

            if not os.path.exists(batch_save_path):
                click.echo(f"Encoding Batch {batch_num + 1}/{len(iterator)}...")
                corpus_end_idx = min(corpus_start_idx + CORPUS_CHUNK_SIZE, len(corpus))
                # Encode chunk of corpus
                sub_corpus_embeddings = _encode_corpus(
                    batch_save_path,
                    corpus[corpus_start_idx:corpus_end_idx],
                    model,
                    encode_kwargs,
                )
            else:
                sub_corpus_embeddings = torch.load(batch_save_path, weights_only=False)
                click.echo(
                    f"Loaded corpus embeddings for batch {batch_num + 1}/{len(iterator)} at {batch_save_path}"
                )

            # Loop over all quantizations and dimensions
            for dimensionality in dimensionalities:
                corpus_embeds = sub_corpus_embeddings[:, :dimensionality]
                query_embeds = query_embeddings[:, :dimensionality]

                # Compute cosine similarity
                similarity_scores = cos_sim(query_embeds, corpus_embeds).detach().cpu()

                # Check for NaN values
                assert torch.isnan(similarity_scores).sum() == 0

                # Get top-k values
                (
                    similarity_scores_top_k_values,
                    similarity_scores_top_k_idx,
                ) = torch.topk(
                    similarity_scores,
                    min(
                        top_k + 1,
                        len(similarity_scores[1])
                        if len(similarity_scores) > 1
                        else len(similarity_scores[-1]),
                    ),
                    dim=1,
                    largest=True,
                    sorted=False,
                )
                similarity_scores_top_k_values = (
                    similarity_scores_top_k_values.cpu().tolist()
                )
                similarity_scores_top_k_idx = similarity_scores_top_k_idx.cpu().tolist()

                for query_itr in range(len(query_embeddings)):
                    query_id = query_ids[query_itr]

                    for sub_corpus_id, score in zip(
                        similarity_scores_top_k_idx[query_itr],
                        similarity_scores_top_k_values[query_itr],
                    ):
                        corpus_id = corpus_ids[corpus_start_idx + sub_corpus_id]

                        if len(result_heaps[dimensionality][query_id]) < top_k:
                            # Push item on the heap
                            heapq.heappush(
                                result_heaps[dimensionality][query_id],
                                (score, corpus_id),
                            )
                        else:
                            # If item is larger than the smallest in the heap, push it on the heap then pop the smallest element
                            heapq.heappushpop(
                                result_heaps[dimensionality][query_id],
                                (score, corpus_id),
                            )

        # Clear CUDA cache
        torch.cuda.empty_cache()

        # Compute final top-k results
        for dimensionality in dimensionalities:
            for qid in result_heaps[dimensionality]:
                for score, cid in result_heaps[dimensionality][qid]:
                    results[corpus_size][dimensionality][qid][cid] = score

    torch.cuda.empty_cache()
    return results


@click.command()
@click.argument(
    "model_name",
    type=click.Choice(["all"] + list(MODEL_WRAPPERS.keys())),
    default="all",
)
@click.argument(
    "document_length",
    type=click.Choice(["passage", "document", "both"]),
    default="both",
)
def compute_empirical_results(model_name: str, document_length: str):
    """
    Wrapper function to compute empirical results for the given model and document length.
    """
    model_names = list(MODEL_WRAPPERS.keys()) if model_name == "all" else [model_name]
    document_lengths = (
        ["passage", "document"] if document_length == "both" else [document_length]
    )

    for document_length in document_lengths:
        corpora, queries, qrels, qrels_relevant_only = load_data(document_length)
        click.echo(f"Loaded {len(queries)} queries and {len(corpora)} corpora")

        for model_name in model_names:
            try:
                _compute_empirical_results(
                    model_name,
                    document_length,
                    data=(corpora, queries, qrels, qrels_relevant_only),
                )
            except Exception as e:
                click.echo(f"Error processing {model_name}, {document_length}: {e}")

    click.echo("Done")


def _compute_empirical_results(
    model_name: str, document_length: str, data: Tuple[dict, dict, dict, dict]
):
    """
    Compute empirical results for the given model and document length.
    """
    click.echo("Computing empirical results")
    click.echo(f"Model: {model_name}")
    click.echo(f"Document Length: {document_length}")

    model, max_dim = get_model_wrapper(model_name)

    # Loaded data
    corpora, queries, qrels, qrels_relevant_only = data

    # Initialize save path
    save_path = os.path.join(EMBEDDINGS_FOLDER, model_name, document_length)

    # Evaluate the dataset
    start = time.time()
    results = _search(
        model,
        corpora,
        queries,
        max(K_VALUES),
        DIMENSIONALITIES,
        save_path,
        encode_kwargs={"batch_size": 64},
    )
    end = time.time()
    click.echo(f"Time taken: {end - start:.2f} seconds")

    corpus_results = None

    # Loop over corpora
    for corpus_size in sorted(corpora.keys()):
        if corpus_results is None:
            corpus_results = results[corpus_size]
        else:
            for dim in results[corpus_size]:
                for q in results[corpus_size][dim]:
                    for qid in results[corpus_size][dim][q]:
                        for cid in results[corpus_size][dim][q][qid]:
                            corpus_results[dim][q][qid][cid] = results[corpus_size][
                                dim
                            ][q][qid][cid]

        save_path = os.path.join(RESULTS_FOLDER, model_name, document_length)
        _eval_results(
            DIMENSIONALITIES,
            K_VALUES,
            corpus_size,
            save_path,
            qrels_relevant_only,
            qrels,
            corpus_results,
            max_dim,
        )
