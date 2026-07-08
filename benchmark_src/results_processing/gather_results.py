from pathlib import Path
import logging
from typing import Annotated

import typer

from benchmark_src.results_processing import aggregate, create_plots, resources

logger = logging.getLogger(__name__)


def run(results_folder_name: str) -> None:
    """Aggregate all results.json files under results_folder_name into summary CSVs."""
    results_folder = Path(results_folder_name)
    assert results_folder.exists(), f"Could not find results folder at {results_folder}"

    detailed_results_folder = results_folder / "results_per_task"
    detailed_results_folder.mkdir(parents=True, exist_ok=True)

    logger.info(f"Gathering results from {results_folder}")
    aggregate.gather_results(results_folder, detailed_results_folder)

    logger.info(f"Gathering resource metrics from {results_folder}")
    resources.gather_resources(results_folder, detailed_results_folder)

    logger.info(f"Creating general plots for {results_folder}")
    create_plots.run(results_folder_name)


def main(
    results_folder_name: Annotated[str, typer.Argument(help="Path to the results folder")],
) -> None:
    run(results_folder_name)


if __name__ == "__main__":
    typer.run(main)
