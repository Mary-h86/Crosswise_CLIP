import numpy as np

def average_precision(
    rel_sorted: np.ndarray
) -> float:
    """rel_sorted is an ordered binary array, 1 where that tile is correct"""
    n_rel = rel_sorted.sum()

    # If nothing is correct
    if n_rel == 0:
        return float("nan")

    # How many predictions we got at each rank
    hits = np.cumsum(rel_sorted) 

    # Rankings are 1-indexed
    ranks = np.arange(1, len(rel_sorted) + 1)

    # Caclulate precision at each rank
    rank_precision = hits / ranks

    # Comptue average of the precisions
    return float((rank_precision * rel_sorted).sum() / n_rel)

def text_to_image_metrics(
    scores,
    labels,
    type_names,
    ks=(5,10)
) -> dict:
    relevance = labels.T # (T types, N tiles) => relevance[i,j] => tile j has type i
    per_type = {}
    """Recall@K, Precision@K, AP per type, and mAP across all types"""

    for i, name in enumerate(type_names):
        order = np.argsort(-scores[i], kind="stable") # indices sorted from highest score -> lowest
        
        rel = relevance[i][order] # arrange ground-truth in same order

        n_rel = int(rel.sum())

        entry = {"n_relevant": n_rel, "AP": average_precision(rel)}

        # Compute Precision and Recall at each of the provided k-values
        for k in ks:
            found = rel[:k].sum() # correct tiles found in top K
            entry[f"precision@{k}"] = float(found / min(k, len(rel)))
            entry[f"recall@{k}"] = float (found / n_rel) if n_rel else float("nan")

        per_type[name] = entry

    # Compute mAP by averaging across all intersection types
    mAP = float(np.nanmean([e["AP"] for e in per_type.values()]))
    return {"per_type": per_type, "mAP": mAP}

def image_to_text_recall_at_1(
    scores,
    labels,
) -> float:
    pred = scores.argmax(axis=0) # best scoring type for each tile
    hit = labels[np.arange(labels.shape[0]), pred] == 1
    return float(hit.mean())

def evaluate(
    scores,
    labels,
    type_names,
    ks=(5, 10)
) -> dict:
    T, N = scores.shape
    out = text_to_image_metrics(scores, labels, type_names, ks)

    # Add img->txtrecall@1 and the number of tiles 
    out["image_to_text_recall@1"] = image_to_text_recall_at_1(scores, labels)
    out["n_tiles"] = N

    # Out contains: Per type (Recall@K, Precision@K, AP), mAP, image_to_text_recall@1, N_tiles 
    return out


