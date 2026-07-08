import pandas as pd

from benchmark_src.utils import cfg_utils


def get_metric_domain(values: pd.Series, metric_name: str) -> tuple[float, float]:
    """
    Domain (min, max) to plot metric_name over, so charts always cover the full
    metric range instead of a data-driven one. Falls back to values' own min/max
    for any side of the range that configs/metric_information.yaml marks as "∞"
    (unbounded), and for metrics missing from that file entirely.
    """
    metric_ranges = cfg_utils.load_metric_ranges()

    domain = None
    for key, value in metric_ranges.items():
        if metric_name.startswith(key):
            domain = value
            break

    if domain is None:
        return values.min(), values.max()

    domain_min, domain_max = domain
    if domain_min == "∞":
        domain_min = values.min()
    if domain_max == "∞":
        domain_max = values.max()

    return domain_min, domain_max
