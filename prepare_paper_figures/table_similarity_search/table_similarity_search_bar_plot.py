import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

import config_helpers as h

VARIANT_HATCHES = {"Just Schema": None, "Schema + 100 rows": "//"}

# approaches whose table embedding is purely a mean over row embeddings, with no
# independent header/schema-only encoding path — they can never produce a "Just
# Schema" (0-row) data point, unlike e.g. tabbie's earlier crash (a fixable bug)
NO_SCHEMA_ONLY_APPROACHES = {"tabdpt", "tarte"}


def create_barplot(df: pd.DataFrame, plots_folder: Path):
    """Bar plot comparing MRR/MAP on GitTables schema-only vs. schema+100-rows, per approach."""
    for metric in ("MRR", "MAP"):
        if metric not in df.columns:
            print(f"WARNING: {metric} not found in table_similarity_search data, skipping")
            continue

        df_filtered = df[["Approach", "chart_name", "Configuration", "color", metric]].copy()
        df_filtered["chart_name"] = df_filtered["chart_name"].str.replace("*", "", regex=False)
        df_filtered["variant"] = df_filtered["Configuration"].apply(
            lambda x: "Just Schema" if "table_row_limit=0" in str(x) else "Schema + 100 rows"
        )

        variants = list(VARIANT_HATCHES)
        methods = sorted(df_filtered["chart_name"].unique(), key=h.chart_name_sort_key)
        chart_to_approach = df_filtered.drop_duplicates("chart_name").set_index("chart_name")["Approach"]

        fig, ax = plt.subplots(figsize=(max(12, 1.4 * len(methods)), 4.8))

        n_variants = len(variants)
        gap = 0.025
        bar_width = (0.8 - gap * (n_variants - 1)) / n_variants

        x = np.arange(len(methods))
        total_group_width = n_variants * bar_width + (n_variants - 1) * gap
        group_centers = x + total_group_width / 2

        for i, variant in enumerate(variants):
            variant_df = (
                df_filtered[df_filtered["variant"] == variant]
                .set_index("chart_name")
                .reindex(methods)
                .reset_index()
            )
            # reindex introduces NaN rows for approaches without this variant (e.g. tabdpt/tarte
            # have no schema-only mode); "none" keeps the (height-less) bar invisible instead of
            # crashing ax.bar's color parsing
            variant_df["color"] = variant_df["color"].fillna("none")

            positions = x + i * (bar_width + gap)

            bars = ax.bar(
                positions,
                variant_df[metric],
                width=bar_width,
                color=variant_df["color"],
                hatch=VARIANT_HATCHES[variant],
                edgecolor="white",
            )

            for method, bar in zip(methods, bars):
                height = bar.get_height()
                if np.isnan(height):
                    if variant == "Just Schema" and chart_to_approach[method] in NO_SCHEMA_ONLY_APPROACHES:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            0.02,
                            "×",  # marks approaches with no schema-only representation
                            ha="center",
                            va="bottom",
                            fontsize=13,
                            color="#555555",
                        )
                    continue
                label = "0" if height == 0 else f"{height:.2f}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.015,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    rotation=0,
                )

        ax.set_xticks(group_centers)
        ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=13)

        ax.set_ylabel(metric, fontsize=16)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="y", labelsize=14)

        ax.grid(axis="y", linestyle="--", alpha=0.3)

        # legend distinguishes variant via hatch, not color (color already encodes approach)
        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", hatch=VARIANT_HATCHES[v])
            for v in variants
        ]
        ax.legend(
            legend_handles,
            variants,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=2,
            frameon=False,
            fontsize=13,
        )

        plt.tight_layout()

        fig.savefig(plots_folder / f"table_similarity_search_barchart_{metric.lower()}.pdf", bbox_inches="tight")
        plt.close(fig)
