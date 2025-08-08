import numpy as np

def sort_with_nones(item):
    if item is None:
        return(1, item)
    else:
        return(0, item)

def replace_ranges(position_list):
    """
    Replace ranges with the lowest of the positions
    e.g. [1, 2, 5, 7, 8] --> [1, 1, 5, 7, 7]
    """
    position_list = sorted(position_list, key=sort_with_nones)
    result_list = position_list.copy()
    start_of_range = None
    for i, position_value in enumerate(position_list):
        if position_value is not None:
            if start_of_range is None:
                # try to start a new range
                start_of_range = position_value
            else:
                # If range does not continue, try to start new range
                if position_value != position_list[i-1] +1: 
                    start_of_range = position_value
            # overwrite value with start of range value
            if start_of_range is not None:
                 result_list[i] = start_of_range
        else:
            # do not modify None values
            start_of_range = None

    #if result_list != position_list :
    #    print(f"Before: {position_list}, after: {result_list}")
    return result_list

def get_position_of_gt(gt_row_ids, predicted_row_ids):
    ##### find position(s) of gt rows in the predicted df (order doesn't matter)
    positions = {}
    position_list = []
    for gt_row_id in gt_row_ids:
        if gt_row_id in predicted_row_ids:
            positions[gt_row_id] = predicted_row_ids.index(gt_row_id) + 1 # start counting positions at 1
            position_list.append(predicted_row_ids.index(gt_row_id) + 1)
        else:
            positions[gt_row_id] = None
            position_list.append(None)

    # for clusters, replace consecutive positions with the lower one
    num_none = sum([1 for x in position_list if x is None])
    if len(position_list) - num_none > 1:
        position_list = replace_ranges(position_list)

    return position_list

def found_at(k, position_list):
    """
    Compute percentage of cases where the required item is among the top-k retrieved items
    """
    num_found = 0
    for pos in position_list:
        if not pos is None:
            if pos <= k:
                num_found += 1
    percent_found = num_found / len(position_list) * 100
    print(f"{percent_found:.2f}% in top-{k} ({num_found}/{len(position_list)})")
    return percent_found

def compute_mrr(position_list, max: int):
    """
    Computes the mean reciprocal rank, given a list of positions
    """
    if len(position_list) == 0:
        return 0
    position_list = [x if x is not None else 2*max for x in position_list]
    reciprocal_ranks = [1/rank for rank in position_list]
    assert len(reciprocal_ranks) == len(position_list)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    return mrr

def compute_all_metrics(all_positions):
    all_positions_without_none = [x for x in all_positions if x is not None]
    print(f"Out of {len(all_positions)} working test cases, {len(all_positions_without_none)} were found in the top-10")
    #print(f"Mean {np.mean(all_positions_without_none)} - Median {np.median(all_positions_without_none)}")
    mrr = compute_mrr(all_positions,  10) # TODO: parameterize
    print(f"MRR: {mrr}")
    results = {
        "In top-1 [%]": found_at(1, all_positions),
        "In top-3 [%]": found_at(3, all_positions),
        "In top-5 [%]": found_at(5, all_positions),
        "In top-10 [%]": found_at(10, all_positions),
        "MRR": mrr,
    }

    return results