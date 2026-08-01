# Pneuma (table_retrieval comparison)

Runs [Pneuma](https://github.com/TheDataStation/pneuma) (Balaka et al., SIGMOD 2025,
[arXiv:2504.09207](https://arxiv.org/abs/2504.09207)) against this benchmark's `table_retrieval`
datasets, for a direct comparison against GritLM on that task: a retrieval-purpose-built tabular
model against a generic embedding model, on the task Pneuma was designed for.

## Why this isn't an `approaches/` plugin

Pneuma runs its own end-to-end hybrid retrieval (full-text BM25 + vector search + an LLM
relevance-judgment re-ranking pass) over its own internal index, and only exposes a ranked list of
table names per query — there's no fixed-size per-table/per-query embedding to hand to
`TableEmbeddingInterface.create_table_embedding`/`create_query_embedding` the way every other
approach in this benchmark does. See `standalone_comparisons/README.md` for the general rationale.

## Backend

Configured for the **OpenAI API** backend (not Pneuma's local Qwen2.5-7B-Instruct option), via
`Pneuma(..., use_local_model=False, openai_api_key=..., llm_path=<openai model id>)`. Default model
is `gpt-4o-mini` (`--llm-model` to override) — summarization and per-query LLM re-ranking both call
this model, so cost scales with corpus size (once, at index build time) and query count (every run
against a built index).

`use_local_model=False` also routes the embedding step through the OpenAI API, not a local HF model
— `src/pneuma/query_processor/query_processor.py` calls `prompt_openai_embed(self.embedding_model,
...)` whenever the embedding model is an OpenAI client. `--embed-model` must therefore be a real
OpenAI embeddings model (default `text-embedding-3-small`); Pneuma's own default,
`BAAI/bge-base-en-v1.5`, is only valid for the local-model backend and will 400 against OpenAI's
`/v1/embeddings` endpoint. Either way, nothing runs locally with this backend, so no GPU is needed.

## Setup

```bash
bash standalone_comparisons/pneuma/setup.sh
```

Creates the `benchmark_env_pneuma` conda env (Python 3.12, per Pneuma's own requirement), installs
`benchmark_src` (for dataset loading and `benchmark_src/utils/retrieval_metrics.py` — this doesn't
use the `approaches` plugin package at all), and `pip install`s `pneuma`. Prompts for
`OPENAI_API_KEY` into the repo-root `.env` if not already present.

## Running

```bash
conda activate benchmark_env_pneuma
python standalone_comparisons/pneuma/run_pneuma_table_retrieval.py \
    --dataset bird-validation \
    --results-dir results_testing/pneuma_smoke_test \
    --test-case-limit 10   # smoke test first -- see cost note below
```

Drop `--test-case-limit` for a full run. `--dataset` must be one of
`configs/global_datasets.yaml`'s `table_retrieval_datasets`. Results land at
`<results-dir>/pneuma/table_retrieval/<dataset>/results.json`, in the same schema
`benchmark_src/utils/result_utils.py::save_results` produces, so
`python benchmark_src/results_processing/gather_results.py <results-dir>` picks it up alongside
other approaches with no further changes.

### Dataset scope for this comparison

`spider-train` and `tabfact` are excluded from full runs — they're the two largest query sets
(6,997 and 12,779 queries), and cost scales with query count since re-ranking is one LLM call per
candidate per query. The remaining datasets keep the comparison affordable while still covering
both SQL-schema-derived and open-domain question-answering query styles:

```bash
for dataset in bird-validation spider-validation spider-test ottqa fetaqa; do
    python standalone_comparisons/pneuma/run_pneuma_table_retrieval.py \
        --dataset "$dataset" \
        --results-dir results/main_experiments
done
```

Indexing (CSV materialization, summarization, index generation) is cached per dataset under
`cache/pneuma_storage/<dataset>/` and reused on subsequent runs (including with a different
`--results-dir`, e.g. to re-run just the query/evaluation half after a metrics change) — pass
`--force-reindex` to rebuild from scratch (this also clears the query cache below, since cached
results are tied to the index content, not just its name).

Query-time results (query embedding + LLM re-ranking, the recurring cost — re-charged on every run
otherwise, even for an identical query against an identical index) are also cached, per dataset, in
`cache/pneuma_storage/<dataset>/query_cache.json`, keyed on `(index_name, query, k, n, alpha,
llm_model, embed_model)` and written after every query (not just at the end), so a crash mid-run
doesn't throw away already-paid-for calls. Only successful queries are cached; failures are retried
on the next run. There's no flag to bypass it deliberately (e.g. to resample) — delete the file to
reset.

### Cost cap

`--max-cost-usd` (default `10.0`; `<=0` disables) hard-stops the run once estimated spend reaches
it. Spend is tracked from the **real** `usage` field on every OpenAI response (via a monkey-patch of
`openai.OpenAI.__init__` installed at the top of `main()`, so it catches the client instances Pneuma
constructs internally too — see `CostTracker`/`_install_openai_cost_tracking`), not an estimate from
call/token counts, and priced against `CHAT_PRICING_PER_1M`/`EMBEDDING_PRICING_PER_1M` in the script
(extend these if you pass a different `--llm-model`/`--embed-model`).

The cap is enforced at two checkpoints the script controls — right after indexing completes, and
before each query in `evaluate()`'s loop — rather than by raising from inside the patched client
call itself: Pneuma does not reliably surface underlying API errors from its internal client calls
(see the embedding-batch-size limitation below), so an exception raised that deep can't be trusted
to propagate out. Hitting the cap mid-run stops cleanly with whatever queries were already
evaluated; already-completed indexing and cached query results are unaffected and picked up
normally on a re-run.

## Known caveats

- Corpus tables are truncated to `--table-row-limit` rows (default 100, matching the
  `table_row_limit` convention other approaches in this repo use) with each cell capped at
  `--max-cell-length` characters (default 500) before being written as CSVs. Pneuma embeds an entire
  small corpus' documents in a single OpenAI request, so without this cap a single oversized cell
  (e.g. raw Stack-Exchange post-revision text, tens of thousands of characters) can exceed
  `text-embedding-3-small`'s 8,191-token-per-input limit and fail the whole batch with a 400, which
  aborts `generate_index()` before it persists the fulltext (BM25) index — the resulting
  `FileNotFoundError` on `indexes/fulltext/<index_name>/params.index.json` at query time is a
  downstream symptom of that aborted indexing run, not a separate bug.
- Pneuma's own indexing step batches embedding requests well past the OpenAI `/v1/embeddings`
  endpoint's hard 2,048-item-per-request limit. On corpora that produce enough documents to exceed
  it (e.g. `fetaqa`'s ~2,000 tables, each contributing multiple documents), index generation fails
  outright and the dataset cannot be indexed with this backend as published. This is a limitation of
  Pneuma's current release rather than something this script works around, since patching around it
  would mean evaluating a modified Pneuma rather than the published system.
- `retrieved_tables` in `query_index`'s response is each match's full registered CSV path with its
  internal `_SEP_contents_SEP_<doc-type>-<n>` per-document suffix stripped (`table.split("_SEP_")[0]`
  in `src/pneuma/query_processor/query_processor.py`) — not the bare filename stem.
  `materialize_corpus_csvs` therefore maps results back to `(database_id, table_id)` by full path,
  not by filename stem.
- `--test-case-limit` only truncates queries, not the corpus. Corpus and queries are independent
  slices of the dataset, so truncating both by row index can drop the gold table(s) for the kept
  queries out of the index entirely, guaranteeing zero recall regardless of whether retrieval itself
  works — use `--test-case-limit` for cost/latency smoke tests only, not for validating recall.
  ChromaDB (Pneuma's vector index backend) periodically logs "failed to send telemetry event";
  this is unrelated and harmless.
- Pneuma doesn't return a per-hit relevance score, only a rank-ordered table list — `results.json`'s
  metrics (MRR/MAP/Recall/Precision) are all rank-based so this doesn't affect scoring, but the
  `score` field in `full_results.json` is a synthetic `1/rank` placeholder, not a real Pneuma score.
- Resource metrics: `run_pneuma_pipeline` (indexing + evaluation together) is wrapped in the same
  `benchmark_src.utils.resource_monitoring.monitor_resources` decorator integrated approaches use,
  writing `resource_metrics_task.json` in the same schema so `gather_results.py` picks it up with no
  changes. Expect this to look unremarkable on CPU/memory/GPU (Pneuma barely uses local compute) but
  very large on `execution_time (s)` — Pneuma reranks with one sequential OpenAI call per candidate
  (no batching or concurrency), so a full `bird-validation` run (1,534 queries) took **~11.2 hours**
  end-to-end versus GritLM's ~200 seconds for the same task. `results.json` also carries an
  `estimated_cost_usd` field (from the real, tracked spend — see "Cost cap" above), which isn't a
  column any other approach has, but flows through to `all_results.csv` as an extra column harmlessly
  (blank for other approaches' rows).
