"""CARROT: k-NN regressors for quality and for z-scored cost, each with tuned k."""
import joblib
import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score

from cost import N_MODELS
from training import CORRECT_THR


def tune_n_neighbors(X_train, Y_train,
                     n_neighbors_range=(2, 4, 8, 16, 32, 64, 128, 256, 512),
                     metric='cosine', cv=5):
    """Tune k via cross-validation (original CARROT logic)."""
    best_score = -float('inf')
    best_k = n_neighbors_range[0]
    for k in n_neighbors_range:
        if k > len(X_train):
            continue
        knn = KNeighborsRegressor(n_neighbors=k, metric=metric)
        scores = cross_val_score(knn, X_train, Y_train, cv=cv, scoring='r2')
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score = mean_score
            best_k = k
    return int(best_k), best_score


def train_carrot_knn(data, *, save_path):
    """Train CARROT with quality KNN + cost KNN (matches original two-branch design).

    Original routing: model_idx = ((1-λ) * quality_pred - λ * cost_pred).argmax()
    We save both KNNs; at evaluation time λ controls the quality-cost tradeoff.
    """
    print(f"\n{'='*60}\nTraining CARROT (KNN, cosine)\n{'='*60}")
    X_train = data["emb"][data["train_idx"]]
    Y_train = data["score"][data["train_idx"]]
    X_test = data["emb"][data["test_idx"]]
    Y_test = data["score"][data["test_idx"]]
    C_train = data["cost"][data["train_idx"]]
    C_test = data["cost"][data["test_idx"]]

    # ---- Quality branch ----
    print("  [Quality branch]")
    best_k, cv_score = tune_n_neighbors(X_train, Y_train)
    print(f"    Best k={best_k}  CV R²={cv_score:.4f}")

    knn_quality = KNeighborsRegressor(n_neighbors=best_k, metric='cosine')
    knn_quality.fit(X_train, Y_train)

    Y_pred = knn_quality.predict(X_test)
    B_true = (Y_test >= CORRECT_THR).astype(int)
    aucs = []
    for i in range(N_MODELS):
        try:
            aucs.append(roc_auc_score(B_true[:, i], Y_pred[:, i]))
        except ValueError:   # only one class present for this model → AUC undefined
            aucs.append(0.5)
    print(f"    Test mean_auc={np.mean(aucs):.4f}")

    # ---- Cost branch (Z-score normalized, matching original) ----
    print("  [Cost branch]")
    cost_mu = C_train.mean(axis=0, keepdims=True)
    cost_std = C_train.std(axis=0, keepdims=True) + 1e-8
    Z_train = (C_train - cost_mu) / cost_std

    best_k_c, cv_score_c = tune_n_neighbors(X_train, Z_train)
    print(f"    Best k={best_k_c}  CV R²={cv_score_c:.4f}")

    knn_cost = KNeighborsRegressor(n_neighbors=best_k_c, metric='cosine')
    knn_cost.fit(X_train, Z_train)

    Z_pred = knn_cost.predict(X_test)
    C_pred = cost_mu + cost_std * Z_pred
    cost_rmse = float(np.sqrt(mean_squared_error(C_test, C_pred)))
    print(f"    Test cost_rmse={cost_rmse:.6f}")

    # Save both branches + normalization params
    carrot_bundle = {
        "knn_quality": knn_quality,
        "knn_cost": knn_cost,
        "cost_mu": cost_mu,
        "cost_std": cost_std,
    }
    joblib.dump(carrot_bundle, save_path)
    print(f"  Saved → {save_path}")
