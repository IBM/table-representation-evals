def compute_testcase_metrics(gt_row_ids, predicted_row_ids, k_values=(1, 3, 5, 10, 50)):
    """
    Computes MRR, MAP, and Recall@k for a single query against its ground-truth block.

    A query can have multiple ground-truth rows (e.g. a cluster of duplicate/matching
    rows). All three metrics are defined per query, not per ground-truth row, so that
    queries with larger ground-truth blocks aren't implicitly weighted more heavily
    when averaged across queries later.
    """
    gt_row_id_set = set(gt_row_ids)
    positions = [predicted_row_ids.index(row_id) + 1 for row_id in gt_row_ids if row_id in predicted_row_ids]

    mrr = 1 / min(positions) if positions else 0.0

    num_hits = 0
    average_precision = 0.0
    for rank, row_id in enumerate(predicted_row_ids, start=1):
        if row_id in gt_row_id_set:
            num_hits += 1
            average_precision += num_hits / rank
    average_precision /= len(gt_row_ids)

    recall_at_k = {k: sum(1 for pos in positions if pos <= k) / len(gt_row_ids) for k in k_values}

    return {"mrr": mrr, "map": average_precision, "recall_at_k": recall_at_k}

def aggregate_testcase_metrics(testcase_metrics_list):
    """
    Averages per-query MRR, MAP, and Recall@k across all test cases of a dataset.
    """
    num_testcases = len(testcase_metrics_list)
    mean_mrr = sum(m["mrr"] for m in testcase_metrics_list) / num_testcases
    mean_map = sum(m["map"] for m in testcase_metrics_list) / num_testcases

    k_values = testcase_metrics_list[0]["recall_at_k"].keys()
    mean_recall_at_k = {
        k: sum(m["recall_at_k"][k] for m in testcase_metrics_list) / num_testcases
        for k in k_values
    }

    results = {f"In top-{k} [%]": mean_recall_at_k[k] * 100 for k in k_values}
    results["MRR"] = mean_mrr
    results["MAP"] = mean_map

    print(f"Out of {num_testcases} test cases: MRR={mean_mrr:.4f}, MAP={mean_map:.4f}")
    for k, recall in mean_recall_at_k.items():
        print(f"Recall@{k}: {recall * 100:.2f}%")

    return results
