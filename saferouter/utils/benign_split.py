"""The benign train/test split, grouped by duplicate query text: R2Bench repeats
texts under different question_ids, so an index split leaks test into train.
Every caller must pass the same qid list.
"""
import numpy as np

# Every trainer must split identically, or the comparison is meaningless.
SPLIT_SEED = 42


def group_split_indices(qids, qid_to_text, test_frac=0.15, seed=SPLIT_SEED):
    """Group-aware 85/15 split -> (train_idx, test_idx), sorted int arrays.
    `qids` is the row order of the data matrices; rows missing text form
    singleton groups, i.e. behave as unique."""
    groups = {}
    for i, q in enumerate(qids):
        text = (qid_to_text.get(str(q)) or "").strip()
        key = text if text else f"__qid__{q}"
        groups.setdefault(key, []).append(i)

    keys = sorted(groups)  # deterministic base order before shuffling
    rng = np.random.RandomState(seed)
    rng.shuffle(keys)

    n_test_target = int(test_frac * len(qids))
    test_idx = []
    for k in keys:
        if len(test_idx) >= n_test_target:
            break
        test_idx.extend(groups[k])

    test_idx = np.array(sorted(test_idx), dtype=int)
    train_idx = np.setdiff1d(np.arange(len(qids)), test_idx)

    n_dup_groups = sum(1 for v in groups.values() if len(v) > 1)
    print(f"  Group split: {len(groups)} unique texts ({n_dup_groups} dup groups), "
          f"train={len(train_idx)}, test={len(test_idx)} (no group straddles)")
    return train_idx, test_idx


def canonicalize_goldens(rows, query_field="query",
                         golden_fields=("ground_truth", "golden_answer")):
    """One golden answer per unique query text. Mutates rows in place: a query
    with >1 distinct golden gets the most frequent (ties -> lexicographically
    smallest). Idempotent; returns the number of rows rewritten."""
    from collections import Counter

    def get_golden(r):
        for f in golden_fields:
            v = r.get(f)
            if v is not None and str(v).strip() and str(v).strip().lower() != "nan":
                return str(v)
        return None

    by_text = {}
    for r in rows:
        text = (r.get(query_field) or "").strip()
        g = get_golden(r)
        if text and g is not None:
            by_text.setdefault(text, Counter())[g] += 1

    canonical = {}
    for text, counts in by_text.items():
        if len(counts) > 1:
            top = max(counts.values())
            canonical[text] = sorted(g for g, c in counts.items() if c == top)[0]

    n_fixed = 0
    for r in rows:
        text = (r.get(query_field) or "").strip()
        if text in canonical and get_golden(r) != canonical[text]:
            for f in golden_fields:
                if r.get(f) is not None:
                    r[f] = canonical[text]
            n_fixed += 1
    if canonical:
        print(f"  Golden canonicalization: {len(canonical)} conflicting query "
              f"texts, {n_fixed} rows rewritten")
    return n_fixed
