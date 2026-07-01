"""
Analyze and compare fuzzy cell matching benchmark files.

Run on the old synthetic version:
    python benchmark_src/dataset_creation/bird/analyze_fuzzy_benchmark.py \
        benchmark_src/dataset_creation/bird/fuzzy_cell_matching.json

Run on the new semantic version:
    python benchmark_src/dataset_creation/bird/analyze_fuzzy_benchmark.py \
        benchmark_src/dataset_creation/bird/semantic_fuzzy_matching.json

Compare both:
    python benchmark_src/dataset_creation/bird/analyze_fuzzy_benchmark.py \
        benchmark_src/dataset_creation/bird/fuzzy_cell_matching.json \
        benchmark_src/dataset_creation/bird/semantic_fuzzy_matching.json
"""

import json
import sys
from collections import Counter
from pathlib import Path


def edit_distance(a: str, b: str) -> int:
    a, b = a.lower(), b.lower()
    if len(a) > len(b):
        a, b = b, a
    row = list(range(len(a) + 1))
    for c2 in b:
        new_row = [row[0] + 1]
        for j, c1 in enumerate(a):
            new_row.append(min(new_row[-1] + 1, row[j + 1] + 1, row[j] + (c1 != c2)))
        row = new_row
    return row[-1]


def analyze(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)

    nl_phrases = [e["extracted_values_from_NL"][0] for e in data if e.get("extracted_values_from_NL")]
    db_values  = [e["matched_values"][0]           for e in data if e.get("matched_values")]
    pairs = list(zip(nl_phrases, db_values))

    fuzzy_types   = Counter(e.get("fuzzy_type", e.get("category", "unknown")) for e in data)
    db_ids        = Counter(e["db_id"] for e in data)
    columns       = Counter(
        f"{g['table']}.{g['column']}"
        for e in data
        for g in e.get("gold_columns", [])
    )

    nl_lengths  = [len(nl) for nl in nl_phrases]
    db_lengths  = [len(db) for db in db_values]
    edit_dists  = [edit_distance(nl, db) for nl, db in pairs]
    # normalised edit distance relative to the longer string
    norm_dists  = [
        ed / max(len(nl), len(db), 1)
        for (nl, db), ed in zip(pairs, edit_dists)
    ]

    def percentile(lst, p):
        if not lst:
            return 0
        s = sorted(lst)
        idx = int(len(s) * p / 100)
        return s[min(idx, len(s) - 1)]

    def mean(lst):
        return sum(lst) / len(lst) if lst else 0

    # Edit distance bucket distribution
    ed_buckets = Counter()
    for d in edit_dists:
        if d == 0:
            ed_buckets["0 (identical)"] += 1
        elif d <= 2:
            ed_buckets["1-2"] += 1
        elif d <= 5:
            ed_buckets["3-5"] += 1
        elif d <= 10:
            ed_buckets["6-10"] += 1
        else:
            ed_buckets[">10"] += 1

    return {
        "path": path,
        "n": len(data),
        "fuzzy_types": fuzzy_types,
        "n_databases": len(db_ids),
        "top_databases": db_ids.most_common(10),
        "n_columns": len(columns),
        "top_columns": columns.most_common(10),
        "nl_len": {"mean": mean(nl_lengths), "p50": percentile(nl_lengths, 50),
                   "p90": percentile(nl_lengths, 90), "max": max(nl_lengths, default=0)},
        "db_len": {"mean": mean(db_lengths), "p50": percentile(db_lengths, 50),
                   "p90": percentile(db_lengths, 90), "max": max(db_lengths, default=0)},
        "edit_dist": {"mean": mean(edit_dists), "p50": percentile(edit_dists, 50),
                      "p90": percentile(edit_dists, 90), "max": max(edit_dists, default=0)},
        "norm_edit_dist": {"mean": mean(norm_dists), "p50": percentile(norm_dists, 50),
                           "p90": percentile(norm_dists, 90)},
        "ed_buckets": ed_buckets,
        "samples_by_type": {
            t: [(e["extracted_values_from_NL"][0], e["matched_values"][0], e.get("db_id", ""))
                for e in data
                if e.get("fuzzy_type", e.get("category", "unknown")) == t][:3]
            for t in fuzzy_types
        },
    }


def print_report(r: dict):
    print(f"\n{'='*70}")
    print(f"  {r['path'].name}  ({r['n']} entries)")
    print(f"{'='*70}")

    print(f"\nType distribution:")
    for t, n in sorted(r["fuzzy_types"].items(), key=lambda x: -x[1]):
        bar = "█" * (n * 30 // max(r["fuzzy_types"].values(), default=1))
        print(f"  {t:<30s}  {n:4d}  {bar}")

    print(f"\nCoverage:")
    print(f"  Databases:  {r['n_databases']}")
    print(f"  Columns:    {r['n_columns']}")

    print(f"\nTop databases:")
    for db, n in r["top_databases"]:
        print(f"  {db:<40s}  {n}")

    print(f"\nTop columns:")
    for col, n in r["top_columns"]:
        print(f"  {col:<45s}  {n}")

    print(f"\nNL phrase length (chars):  mean={r['nl_len']['mean']:.1f}  "
          f"p50={r['nl_len']['p50']}  p90={r['nl_len']['p90']}  max={r['nl_len']['max']}")
    print(f"DB value length (chars):   mean={r['db_len']['mean']:.1f}  "
          f"p50={r['db_len']['p50']}  p90={r['db_len']['p90']}  max={r['db_len']['max']}")
    print(f"Edit distance:             mean={r['edit_dist']['mean']:.1f}  "
          f"p50={r['edit_dist']['p50']}  p90={r['edit_dist']['p90']}  max={r['edit_dist']['max']}")
    print(f"Normalised edit distance:  mean={r['norm_edit_dist']['mean']:.2f}  "
          f"p50={r['norm_edit_dist']['p50']:.2f}  p90={r['norm_edit_dist']['p90']:.2f}")

    print(f"\nEdit distance buckets:")
    for bucket in ["0 (identical)", "1-2", "3-5", "6-10", ">10"]:
        n = r["ed_buckets"].get(bucket, 0)
        pct = n / r["n"] * 100 if r["n"] else 0
        print(f"  {bucket:<20s}  {n:4d}  ({pct:.1f}%)")

    print(f"\nSamples per type (NL → DB):")
    for t, samples in sorted(r["samples_by_type"].items()):
        print(f"  [{t}]")
        for nl, db, db_id in samples:
            print(f"    {repr(nl):<45s}  →  {repr(db)}")


def print_comparison(r1: dict, r2: dict):
    print(f"\n{'='*70}")
    print(f"  COMPARISON")
    print(f"  A: {r1['path'].name}  ({r1['n']} entries)")
    print(f"  B: {r2['path'].name}  ({r2['n']} entries)")
    print(f"{'='*70}")

    all_types = sorted(set(r1["fuzzy_types"]) | set(r2["fuzzy_types"]))
    print(f"\nType distribution:")
    print(f"  {'Type':<30s}  {'A':>6}  {'B':>6}")
    print(f"  {'-'*30}  {'------'}  {'------'}")
    for t in all_types:
        a = r1["fuzzy_types"].get(t, 0)
        b = r2["fuzzy_types"].get(t, 0)
        print(f"  {t:<30s}  {a:6d}  {b:6d}")

    print(f"\n{'Metric':<35s}  {'A':>10}  {'B':>10}")
    print(f"  {'-'*35}  {'----------'}  {'----------'}")
    metrics = [
        ("Total entries",          r1["n"],                          r2["n"]),
        ("Databases covered",      r1["n_databases"],                r2["n_databases"]),
        ("Columns covered",        r1["n_columns"],                  r2["n_columns"]),
        ("NL length mean",         f"{r1['nl_len']['mean']:.1f}",    f"{r2['nl_len']['mean']:.1f}"),
        ("NL length p90",          r1["nl_len"]["p90"],              r2["nl_len"]["p90"]),
        ("DB length mean",         f"{r1['db_len']['mean']:.1f}",    f"{r2['db_len']['mean']:.1f}"),
        ("Edit dist mean",         f"{r1['edit_dist']['mean']:.1f}", f"{r2['edit_dist']['mean']:.1f}"),
        ("Edit dist p90",          r1["edit_dist"]["p90"],           r2["edit_dist"]["p90"]),
        ("Norm edit dist mean",    f"{r1['norm_edit_dist']['mean']:.2f}", f"{r2['norm_edit_dist']['mean']:.2f}"),
    ]
    for label, a, b in metrics:
        print(f"  {label:<35s}  {str(a):>10}  {str(b):>10}")

    # NL/DB pair overlap
    pairs_a = {(e["extracted_values_from_NL"][0].lower(), e["matched_values"][0].lower())
               for e in json.load(open(r1["path"])) if e.get("extracted_values_from_NL") and e.get("matched_values")}
    pairs_b = {(e["extracted_values_from_NL"][0].lower(), e["matched_values"][0].lower())
               for e in json.load(open(r2["path"])) if e.get("extracted_values_from_NL") and e.get("matched_values")}
    overlap = pairs_a & pairs_b
    print(f"\nPair overlap (NL, DB): {len(overlap)} shared pairs out of {len(pairs_a)} A / {len(pairs_b)} B")
    if overlap:
        print("  Examples:")
        for nl, db in list(overlap)[:5]:
            print(f"    {repr(nl):<40s}  →  {repr(db)}")


def main():
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print(__doc__)
        sys.exit(1)

    for p in paths:
        if not p.exists():
            print(f"File not found: {p}", file=sys.stderr)
            sys.exit(1)

    reports = [analyze(p) for p in paths]

    for r in reports:
        print_report(r)

    if len(reports) == 2:
        print_comparison(*reports)


if __name__ == "__main__":
    main()
