# Standalone comparisons

Approaches under `approaches/benchmark_approaches_src/` all plug into the shared framework via
`BaseTabularEmbeddingApproach` and the interfaces in `benchmark_src/approach_interfaces/`: they
implement one of a fixed set of capabilities (row/column/cell/table embedding, predictive ML, ...)
and the task runners in `benchmark_src/tasks/` own everything else — indexing, search, scoring.

`standalone_comparisons/` is for approaches that don't fit that model: they run their own
end-to-end pipeline (their own indexing, retrieval, and ranking logic) and only expose "give me the
final answer for this query," not a comparable fixed-size embedding the framework can index and
search itself. Rather than growing the shared interfaces (and the task runners every other approach
depends on) to accommodate each such approach individually, each one gets its own standalone script
here. 

Each script:

- Loads the same benchmark datasets as the main framework (e.g.
  `benchmark_src.dataset_creation.target.collect_all_target_datasets.get_target_dataset_by_name`),
  so results are comparable to the approaches in `approaches/`.
- Runs the approach's own pipeline directly (no `BaseTabularEmbeddingApproach` subclass).
- Scores with the same metric functions the relevant task runner uses (e.g.
  `benchmark_src/utils/retrieval_metrics.py` for retrieval-style tasks)
- Writes a `results.json` in the same schema and folder layout
  (`<results_dir>/<approach_name>/<task_name>/<dataset_name>/results.json`) that
  `benchmark_src/utils/result_utils.py::save_results` produces, so
  `benchmark_src/results_processing/gather_results.py` picks it up with no changes — it appears
  in the same `all_results.csv` and paper tables as regularly-integrated approaches.

Each approach gets its own subfolder with its own `setup.sh` (own conda env, own dependencies) and
its own README documenting the model/backend used and any caveats. These setup scripts are run
manually and are not wired into the root `setup_benchmark.sh`, since they're explicitly outside the
integrated-approach plugin system that script manages.
