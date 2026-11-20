"""For Prompt -> Image retrieval: Precision, Recall, and F1"""

def outcome_counts(
    retrieved: set[str],
    positives: set[str], 
    all_tiles: set[str], # Cleaned to exclude tiles w/o labels
) -> tuple[int, int, int ,int]:  
    """
    Compute TP, FP, TN, FN (tuple returned in that order)

    Currently ignores tiles that don't have any labels
    """
    TP, FP, TN, FN = 0, 0, 0, 0 
    for tile in all_tiles:
        if tile in retrieved:
            if tile in positives:
                TP += 1
            else:
                FP += 1
        else: # tile not retrieved
            if tile not in positives:
                TN += 1
            else:
                FN += 1

    return TP, FP, TN, FN

def metrics(
    outcomes: tuple[int, int, int, int],
) -> tuple[float, float, float]:
    """
    Compute Precision, Recall, F1 (tuple returned in that order)

    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 =  2TP / (2TP + FP + FN)
    """
    TP, FP, TN, FN = outcomes
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * TP / (2 * TP + FP + FN)

    return precision, recall, f1
