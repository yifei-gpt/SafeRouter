"""CARROT: k-NN over query embeddings for quality and cost, blended by lambda."""


def carrot_route(carrot_bundle, X, lam=0.0):
    """argmax_m [(1-λ)·quality − λ·zcost], blended in Z-SCORE space so λ sweeps a
    smooth frontier. Do NOT de-normalize cost to dollars: benign costs are
    ~1000x smaller than quality, which leaves λ inert until it collapses at
    λ→1."""
    quality_pred = carrot_bundle["knn_quality"].predict(X)
    if lam == 0.0:
        return quality_pred.argmax(axis=1)
    z_cost_pred = carrot_bundle["knn_cost"].predict(X)   # already standardized
    scores = (1 - lam) * quality_pred - lam * z_cost_pred
    return scores.argmax(axis=1)
