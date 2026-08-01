"""
Runs Pneuma (https://github.com/TheDataStation/pneuma) against this benchmark's table_retrieval
datasets, scoring with the same metric functions the integrated approaches use
(benchmark_src/utils/retrieval_metrics.py) and writing results.json in the same schema/location, so
it merges into benchmark_src/results_processing/gather_results.py's output for direct comparison
against e.g. GritLM. See standalone_comparisons/README.md for why Pneuma isn't wired into the
approaches/ plugin system instead.

Run via (after standalone_comparisons/pneuma/setup.sh):
  conda activate benchmark_env_pneuma
  python standalone_comparisons/pneuma/run_pneuma_table_retrieval.py \
      --dataset bird-validation --results-dir results_testing/pneuma_smoke_test
"""

import argparse
import ast
import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from omegaconf import OmegaConf
from pneuma import Pneuma
from tqdm import tqdm

from benchmark_src.dataset_creation.target.collect_all_target_datasets import get_target_dataset_by_name
from benchmark_src.dataset_creation.utils import table_2d_to_df
from benchmark_src.utils.resource_monitoring import monitor_resources
from benchmark_src.utils.retrieval_metrics import (
    RetrievalHit,
    calculate_summary_metrics,
    flatten_summary_metrics,
    process_search_results,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APPROACH_NAME = "pneuma"
TASK_NAME = "table_retrieval"

# $ per 1M tokens. Extend if you pass a different --llm-model/--embed-model.
CHAT_PRICING_PER_1M = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}
EMBEDDING_PRICING_PER_1M = {
    "text-embedding-3-small": 0.02,
}


class CostCapExceeded(RuntimeError):
    pass


class CostTracker:
    """Accumulates real spend from OpenAI response `usage` fields (see _install_openai_cost_tracking).

    Only accumulates - never raises itself. Pneuma does not reliably propagate internal API errors
    to its callers, so an exception raised from inside a monkey-patched client call can't be trusted
    to surface. Callers check .total_cost_usd at points *they* control instead (see main() and
    evaluate()).
    """

    def __init__(self):
        self.total_cost_usd = 0.0

    def add(self, model: Optional[str], usage, pricing_table: dict, is_embedding: bool) -> None:
        if usage is None or model is None:
            return
        pricing = pricing_table.get(model)
        if pricing is None:
            logger.warning(f"No known $/token pricing for model '{model}' - cost cap won't account for these calls.")
            return
        if is_embedding:
            cost = usage.prompt_tokens * pricing / 1_000_000
        else:
            cost = (usage.prompt_tokens * pricing["input"] + usage.completion_tokens * pricing["output"]) / 1_000_000
        self.total_cost_usd += cost


def _install_openai_cost_tracking(tracker: CostTracker) -> None:
    """
    Patches openai.OpenAI.__init__ so every client instance - including the ones Pneuma constructs
    internally (self.llm / self.embed_model, not exposed as something we can inject our own client
    into) - has its chat/embeddings calls' real token usage recorded against `tracker`. Wrapping at
    the client-instance level (rather than openai's internal resource classes) avoids depending on
    openai-python's internal module layout.
    """
    import openai

    original_init = openai.OpenAI.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

        original_chat_create = self.chat.completions.create

        def chat_create(*a, **kw):
            response = original_chat_create(*a, **kw)
            tracker.add(kw.get("model"), getattr(response, "usage", None), CHAT_PRICING_PER_1M, is_embedding=False)
            return response

        self.chat.completions.create = chat_create

        original_embeddings_create = self.embeddings.create

        def embeddings_create(*a, **kw):
            response = original_embeddings_create(*a, **kw)
            tracker.add(kw.get("model"), getattr(response, "usage", None), EMBEDDING_PRICING_PER_1M, is_embedding=True)
            return response

        self.embeddings.create = embeddings_create

    openai.OpenAI.__init__ = patched_init


def _table_to_df(table: Any) -> pd.DataFrame:
    if isinstance(table, pd.DataFrame):
        return table
    if isinstance(table, str):
        table = ast.literal_eval(table)
    if isinstance(table, list):
        return table_2d_to_df(table)
    return pd.DataFrame()


def _sanitize(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(value))


def _deduplicate_columns(columns) -> List[str]:
    """
    Wikipedia-table datasets (e.g. fetaqa) can have duplicate column names, unlike the SQL-schema
    datasets (bird, spider) where every column is unique - table[col] returns a DataFrame instead of
    a Series for a duplicate name, breaking any code (ours or Pneuma's) that assumes Series semantics.
    Matches the suffixing pandas.read_csv itself applies to duplicate headers on read.
    """
    seen: Dict[str, int] = {}
    deduped = []
    for col in columns:
        col = str(col)
        if col not in seen:
            seen[col] = 0
            deduped.append(col)
        else:
            seen[col] += 1
            deduped.append(f"{col}.{seen[col]}")
    return deduped


def _truncate_object_columns(table: pd.DataFrame, max_cell_length: int) -> pd.DataFrame:
    for col in table.columns:
        if table[col].dtype == object:
            table[col] = table[col].apply(
                lambda v: v[:max_cell_length] if isinstance(v, str) and len(v) > max_cell_length else v
            )
    return table


def materialize_corpus_csvs(
    corpus, csv_dir: Path, table_row_limit: int = 100, max_cell_length: int = 500
) -> Dict[str, Tuple[str, str]]:
    """
    Writes each corpus table to <csv_dir>/<stem>.csv and returns each CSV's full path (as a string)
    -> (database_id, table_id).

    Pneuma's registrar uses each CSV's full file path as its internal table id, and query_index()
    returns matches as that same full path (with its "_SEP_contents_SEP_<doc-type>-<n>" per-document
    suffix already stripped via table.split("_SEP_")[0], per src/pneuma/query_processor.py) - so the
    full path, not the filename stem, is what results need to be mapped back to (database_id,
    table_id) gold tuples through. The filename stem still encodes both ids so the mapping survives
    Pneuma's own path normalization.

    Rows and cell values are truncated (table_row_limit, max_cell_length) because Pneuma embeds an
    entire small corpus' documents in a single OpenAI request: one oversized cell (e.g. a raw
    Stack-Exchange post-revision body) can exceed the embedding model's per-input token limit and
    fail that whole batch, aborting index generation before it persists the fulltext index.
    """
    csv_dir.mkdir(parents=True, exist_ok=True)
    path_to_ids: Dict[str, Tuple[str, str]] = {}

    for i, row in enumerate(tqdm(corpus, desc="Materializing corpus CSVs")):
        table = _table_to_df(row["table"])
        if table.empty:
            logger.warning(
                f"Skipping empty table: database_id={row.get('database_id')}, table_id={row.get('table_id')}"
            )
            continue

        table.columns = _deduplicate_columns(table.columns)

        if table_row_limit > 0:
            table = table.head(table_row_limit)
        if max_cell_length > 0:
            table = _truncate_object_columns(table, max_cell_length)

        db_id, tbl_id = row["database_id"], row["table_id"]
        stem = f"{i:06d}__{_sanitize(db_id)}__{_sanitize(tbl_id)}"
        csv_path = csv_dir / f"{stem}.csv"
        table.to_csv(csv_path, index=False)
        path_to_ids[str(csv_path)] = (db_id, tbl_id)

    logger.info(f"Materialized {len(path_to_ids)} tables to {csv_dir}")
    return path_to_ids


def load_top_ks() -> List[int]:
    cfg_path = PROJECT_ROOT / "configs" / "task" / f"{TASK_NAME}.yaml"
    task_cfg = OmegaConf.load(cfg_path)
    return list(task_cfg.top_ks)


def query_pneuma(pneuma: Pneuma, index_name: str, query_text: str, k: int, n: int, alpha: float) -> Optional[List[str]]:
    """Returns the ranked list of full CSV table paths Pneuma retrieved for one query, or None on failure."""
    raw_response = pneuma.query_index(index_name=index_name, queries=query_text, k=k, n=n, alpha=alpha)
    response = json.loads(raw_response)

    if response.get("status") != "SUCCESS":
        logger.error(f"query_index failed for query {query_text!r}: {response.get('message')}")
        return None

    data = response.get("data")
    # queries accepts list[str] | str; be defensive about whether a single query still comes back
    # wrapped in a list.
    if isinstance(data, dict):
        data = [data]
    if not data:
        return []

    return data[0].get("retrieved_tables", [])


def _query_cache_key(
    index_name: str, query_text: str, k: int, n: int, alpha: float, llm_model: str, embed_model: str
) -> str:
    payload = json.dumps(
        {
            "index_name": index_name,
            "query": query_text,
            "k": k,
            "n": n,
            "alpha": alpha,
            "llm_model": llm_model,
            "embed_model": embed_model,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_query_cache(cache_path: Path) -> Dict[str, List[str]]:
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return {}


def save_query_cache(cache_path: Path, cache: Dict[str, List[str]]) -> None:
    with open(cache_path, "w") as f:
        json.dump(cache, f)


def evaluate(
    pneuma: Pneuma,
    index_name: str,
    queries_dataset,
    path_to_ids: Dict[str, Tuple[str, str]],
    top_ks: List[int],
    n: int,
    alpha: float,
    llm_model: str,
    embed_model: str,
    query_cache: Dict[str, List[str]],
    query_cache_path: Path,
    cost_tracker: CostTracker,
    max_cost_usd: Optional[float],
) -> Dict[str, Any]:
    max_k = max(top_ks)
    per_query_results = []
    cache_hits = 0

    for query_row in tqdm(queries_dataset, desc="Evaluating queries"):
        if max_cost_usd is not None and cost_tracker.total_cost_usd >= max_cost_usd:
            logger.warning(
                f"Stopping: estimated spend ${cost_tracker.total_cost_usd:.4f} reached --max-cost-usd "
                f"cap ${max_cost_usd:.2f}. Evaluated {len(per_query_results)}/{len(queries_dataset)} "
                f"queries; results reflect only those."
            )
            break

        query_text = query_row["query"]
        query_id = query_row["query_id"]
        gt_database_id = query_row["database_id"]
        gt_table_id_list = query_row["table_id"]
        if isinstance(gt_table_id_list, str):
            gt_table_id_list = [gt_table_id_list]
        gold_tables_set = set((gt_database_id, t) for t in gt_table_id_list)

        cache_key = _query_cache_key(index_name, query_text, max_k, n, alpha, llm_model, embed_model)
        if cache_key in query_cache:
            retrieved_paths = query_cache[cache_key]
            cache_hits += 1
        else:
            retrieved_paths = query_pneuma(pneuma, index_name, query_text, k=max_k, n=n, alpha=alpha)
            if retrieved_paths is None:
                logger.warning(f"Query failed, not caching: {query_text!r}")
                retrieved_paths = []
            else:
                # Persisted after every query (not just at the end) so a crash mid-run doesn't
                # throw away already-paid-for LLM re-ranking calls.
                query_cache[cache_key] = retrieved_paths
                save_query_cache(query_cache_path, query_cache)

        hits = []
        for rank, table_path in enumerate(retrieved_paths, start=1):
            ids = path_to_ids.get(table_path)
            if ids is None:
                logger.warning(f"Pneuma returned unknown table path '{table_path}', skipping.")
                continue
            db_id, tbl_id = ids
            # Pneuma doesn't expose a per-hit score, only a rank-ordered list; metrics below are
            # rank-based (MRR/MAP/Recall/Precision), so a placeholder monotonic score is fine.
            hits.append(RetrievalHit(database_id=db_id, table_id=tbl_id, score=1.0 / rank))

        processed = process_search_results(hits, gold_tables_set, top_ks)

        per_query_results.append({
            "query_id": query_id,
            "query": query_text,
            "ground_truth": {"database_id": gt_database_id, "table_ids": gt_table_id_list},
            "retrieved": processed["retrieved_items_full_list"],
            "metrics": {
                "gold_tables_count": processed["gold_tables_count"],
                "metrics_per_k": processed["metrics_per_k"],
            },
        })

    logger.info(f"Query cache: {cache_hits}/{len(per_query_results)} queries served from cache")

    summary_metrics = calculate_summary_metrics(per_query_results, top_ks)
    return {"summary_metrics": summary_metrics, "per_query_results": per_query_results}


def save_results(
    output_dir: Path,
    dataset_name: str,
    flattened_metrics: Dict[str, Any],
    full_results: Dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {"task": TASK_NAME, "dataset": dataset_name, "approach": APPROACH_NAME}
    results.update(flattened_metrics)

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(output_dir / "full_results.json", "w") as f:
        json.dump(full_results, f, indent=2)

    logger.info(f"Saved results to {output_dir}")


@monitor_resources()
def run_pneuma_pipeline(
    pneuma: Pneuma,
    index_name: str,
    corpus,
    queries,
    csv_dir: Path,
    manifest_path: Path,
    already_indexed: bool,
    table_row_limit: int,
    max_cell_length: int,
    top_ks: List[int],
    n: int,
    alpha: float,
    llm_model: str,
    embed_model: str,
    query_cache: Dict[str, List[str]],
    query_cache_path: Path,
    cost_tracker: CostTracker,
    max_cost_usd: Optional[float],
) -> Dict[str, Any]:
    """Indexing (if not cached) + evaluation, wrapped together so @monitor_resources times/profiles
    the whole task the same way benchmark_src/tasks/*.py wrap their task_inference functions -
    produces a resource_metrics_task.json in the exact schema gather_results.py already reads.
    """
    if not already_indexed:
        logger.info("Materializing corpus tables as CSVs for Pneuma...")
        path_to_ids = materialize_corpus_csvs(
            corpus, csv_dir, table_row_limit=table_row_limit, max_cell_length=max_cell_length
        )

        logger.info("Registering tables with Pneuma...")
        pneuma.add_tables(path=str(csv_dir), creator="tembed")

        logger.info("Summarizing tables (LLM calls)...")
        pneuma.summarize()

        logger.info(f"Generating index '{index_name}'...")
        pneuma.generate_index(index_name=index_name)

        with open(manifest_path, "w") as f:
            json.dump(path_to_ids, f)

        logger.info(f"Estimated spend after indexing: ${cost_tracker.total_cost_usd:.4f}")
        if max_cost_usd is not None and cost_tracker.total_cost_usd >= max_cost_usd:
            raise CostCapExceeded(
                f"Estimated spend ${cost_tracker.total_cost_usd:.4f} already reached --max-cost-usd "
                f"cap ${max_cost_usd:.2f} during indexing, before any queries ran. The index itself "
                f"is built and cached, so re-running (without --force-reindex) will resume from here."
            )
    else:
        logger.info(f"Reusing existing Pneuma index at {manifest_path.parent} (pass --force-reindex to rebuild).")
        with open(manifest_path) as f:
            path_to_ids = {path: tuple(ids) for path, ids in json.load(f).items()}

    logger.info(f"Evaluating {len(queries)} queries (LLM calls)...")
    return evaluate(
        pneuma, index_name, queries, path_to_ids, top_ks,
        n=n, alpha=alpha, llm_model=llm_model, embed_model=embed_model,
        query_cache=query_cache, query_cache_path=query_cache_path,
        cost_tracker=cost_tracker, max_cost_usd=max_cost_usd,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="table_retrieval dataset name, e.g. bird-validation")
    parser.add_argument("--results-dir", required=True, help="Root results dir, e.g. results/main_experiments or a fresh results_testing/<timestamp> dir")
    parser.add_argument("--llm-model", default="gpt-4o-mini", help="OpenAI model used for table summarization and query judgment")
    parser.add_argument("--embed-model", default="text-embedding-3-small", help="OpenAI embedding model for the vector half of Pneuma's hybrid index (with use_local_model=False, embeddings go through the OpenAI API too, not a local HF model)")
    parser.add_argument("--alpha", type=float, default=0.5, help="BM25 vs. vector weighting in Pneuma's hybrid fusion (0=vector only, 1=BM25 only)")
    parser.add_argument("--n", type=int, default=5, help="Pneuma's retrieval-pool expansion factor: k*n candidates are fused/re-ranked before truncating to k")
    parser.add_argument("--test-case-limit", type=int, default=None, help="Limit corpus/queries for a smoke test")
    parser.add_argument("--force-reindex", action="store_true", help="Rebuild Pneuma's summaries/index even if a cached one exists for this dataset")
    parser.add_argument("--table-row-limit", type=int, default=100, help="Max rows per table written for Pneuma (matches the table_row_limit convention used by other approaches in this repo); <=0 disables")
    parser.add_argument("--max-cell-length", type=int, default=500, help="Max characters per cell written for Pneuma, to avoid a single oversized cell blowing the embedding model's per-input token limit; <=0 disables")
    parser.add_argument("--max-cost-usd", type=float, default=10.0, help="Hard stop once estimated real OpenAI spend (from response usage, not a guess) reaches this; <=0 disables")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Run standalone_comparisons/pneuma/setup.sh first.")

    max_cost_usd = args.max_cost_usd if args.max_cost_usd > 0 else None
    cost_tracker = CostTracker()
    _install_openai_cost_tracking(cost_tracker)

    top_ks = load_top_ks()

    logger.info(f"Loading dataset '{args.dataset}'...")
    dataset_bundle = get_target_dataset_by_name(args.dataset)
    corpus, queries = dataset_bundle.corpus, dataset_bundle.queries

    if args.test_case_limit:
        # Only truncate queries, not the corpus: gold tables for the kept queries could otherwise
        # end up excluded from the truncated corpus (they're independent slices of the dataset),
        # which would guarantee zero recall regardless of whether retrieval actually works.
        # Summarization cost scales with corpus size (cheap regardless, ~fixed per table) while
        # re-ranking cost scales with query count, so this still keeps a smoke test cheap.
        limit = args.test_case_limit
        queries = queries.select(range(min(limit, len(queries))))
        logger.info(f"test_case_limit={limit}: using full {len(corpus)}-row corpus, {len(queries)} query rows")

    logger.info(f"Dataset '{args.dataset}': {len(corpus)} corpus rows, {len(queries)} query rows")

    pneuma_storage_dir = PROJECT_ROOT / "cache" / "pneuma_storage" / args.dataset
    csv_dir = pneuma_storage_dir / "csv_tables"
    manifest_path = pneuma_storage_dir / "table_id_manifest.json"
    query_cache_path = pneuma_storage_dir / "query_cache.json"
    index_name = f"pneuma_{args.dataset}"

    if args.force_reindex and pneuma_storage_dir.exists():
        # Wipes the query cache too: its results are tied to the index content being rebuilt, not
        # just the index_name, so they can't be trusted to still be valid.
        shutil.rmtree(pneuma_storage_dir)

    already_indexed = manifest_path.exists()

    pneuma = Pneuma(
        out_path=str(pneuma_storage_dir),
        use_local_model=False,
        openai_api_key=os.environ["OPENAI_API_KEY"],
        llm_path=args.llm_model,
        embed_path=args.embed_model,
    )
    pneuma.setup()

    query_cache = load_query_cache(query_cache_path)

    evaluation_results, resource_metrics_task = run_pneuma_pipeline(
        pneuma, index_name, corpus, queries, csv_dir, manifest_path, already_indexed,
        args.table_row_limit, args.max_cell_length, top_ks, args.n, args.alpha,
        args.llm_model, args.embed_model, query_cache, query_cache_path,
        cost_tracker, max_cost_usd,
    )

    logger.info(f"Total estimated spend this run: ${cost_tracker.total_cost_usd:.4f}")

    flattened_metrics = flatten_summary_metrics(evaluation_results["summary_metrics"])
    flattened_metrics["estimated_cost_usd"] = round(cost_tracker.total_cost_usd, 4)

    output_dir = PROJECT_ROOT / args.results_dir / APPROACH_NAME / TASK_NAME / args.dataset
    save_results(output_dir, args.dataset, flattened_metrics, evaluation_results)

    resource_metrics_task["function"] = "task_inference"
    with open(output_dir / "resource_metrics_task.json", "w") as f:
        json.dump(resource_metrics_task, f, indent=4)
    logger.info(f"Saved resource metrics to {output_dir / 'resource_metrics_task.json'}")


if __name__ == "__main__":
    main()
