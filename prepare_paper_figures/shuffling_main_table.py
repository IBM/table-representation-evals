"""T8 + T8-BCS: Table Shuffling main tables — accuracy and BCS, v0 only."""

import pandas as pd
from pathlib import Path
import config_helpers as h


def _prepare_v0_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to v0 variation (hi-pos/hi-neg, perturbation=both, default window)."""
    filtered = df.copy()
    filtered = filtered[filtered['dataset'].str.endswith('@@v0')]
    # Strip @@v0 suffix for cleaner display
    filtered['dataset'] = filtered['dataset'].str.replace(r'@@v0$', '', regex=True)
    return filtered


def create_accuracy_table(df: pd.DataFrame, plots_folder: Path):
    """T8: Triplet accuracy per dataset at v0."""
    filtered = _prepare_v0_data(df)

    metric_col = 'TripletAccuracy'
    if metric_col not in filtered.columns:
        print(f"WARNING: {metric_col} not found in shuffling data")
        return

    pivoted = filtered.pivot_table(
        index='dataset',
        columns='chart_name',
        values=metric_col,
        aggfunc='mean',
    ).sort_index()

    h.write_latex_table(
        pivoted,
        plots_folder,
        filename='shuffling_accuracy_table.tex',
        caption='Table Shuffling accuracy per dataset (v0: hi-pos/hi-neg, both row+col perturbation, '
                'default window). Best per dataset in bold, second-best underlined.',
        label='tab:shuffling_accuracy',
        float_fmt='.4f',
    )


def create_bcs_table(df: pd.DataFrame, plots_folder: Path):
    """T8-BCS: Bounded Contrastive Score per dataset at v0."""
    filtered = _prepare_v0_data(df)

    metric_col = 'Bounded Contrastive Score'
    if metric_col not in filtered.columns:
        print(f"WARNING: {metric_col} not found in shuffling data")
        return

    pivoted = filtered.pivot_table(
        index='dataset',
        columns='chart_name',
        values=metric_col,
        aggfunc='mean',
    ).sort_index()

    h.write_latex_table(
        pivoted,
        plots_folder,
        filename='shuffling_bcs_table.tex',
        caption='Table Shuffling Bounded Contrastive Score per dataset '
                '(v0: hi-pos/hi-neg, both row+col perturbation, default window). '
                'Lower is better. Best per dataset in bold, second-best underlined.',
        label='tab:shuffling_bcs',
        float_fmt='.4f',
        higher_better=False,
    )
