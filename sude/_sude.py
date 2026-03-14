from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_is_fitted

from .clle import clle
from .init_pca import init_pca
from .learning_l import learning_l
from .learning_s import learning_s
from .opt_scale import opt_scale
from .pps import pps


_INIT_ALIASES = {
    "spectral": "le",
    "le": "le",
    "pca": "pca",
    "mds": "mds",
}


def _resolve_init(init: str) -> str:
    try:
        return _INIT_ALIASES[init]
    except KeyError as exc:
        raise ValueError(
            "init must be one of {'spectral', 'le', 'pca', 'mds'}"
        ) from exc


def _validate_parameters(
    n_samples: int,
    n_features: int,
    n_components: int,
    n_neighbors: int,
    normalize: bool,
    large: bool,
    init: str,
    agg_coef: float,
    max_iter: int,
) -> str:
    if not isinstance(n_components, int) or n_components <= 0:
        raise ValueError("n_components must be a positive integer")
    if n_components >= n_features:
        raise ValueError(
            "n_components must be smaller than the number of input features "
            f"(n_features={n_features})"
        )
    if n_components >= n_samples:
        raise ValueError(
            "n_components must be smaller than the number of input samples "
            f"(n_samples={n_samples})"
        )
    if not isinstance(n_neighbors, int) or n_neighbors < 0:
        raise ValueError("n_neighbors must be a non-negative integer")
    if n_neighbors >= n_samples:
        raise ValueError(
            "n_neighbors must be smaller than the number of input samples "
            f"(n_samples={n_samples})"
        )
    if not isinstance(normalize, bool):
        raise ValueError("normalize must be a boolean")
    if not isinstance(large, bool):
        raise ValueError("large must be a boolean")
    if agg_coef < 0:
        raise ValueError("agg_coef must be non-negative")
    if not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")
    return _resolve_init(init)


def _prepare_training_data(
    X: np.ndarray,
    normalize: bool,
):
    X_unique, inverse_indices = np.unique(X, axis=0, return_inverse=True)
    scaler = None
    if normalize:
        scaler = MinMaxScaler().fit(X_unique)
        X_unique = scaler.transform(X_unique)
    return X_unique, inverse_indices, scaler


def _compute_landmarks(
    X: np.ndarray,
    n_components: int,
    n_neighbors: int,
):
    n_samples, n_features = X.shape
    if n_neighbors == 0:
        return [], [], list(range(n_samples))

    if n_samples > 5000 and n_features > 50:
        reduced = init_pca(X, n_components, 0.8)
        get_knn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(
            reduced
        ).kneighbors(reduced, return_distance=False)
    else:
        get_knn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(
            X
        ).kneighbors(X, return_distance=False)
    _, rnn = np.unique(get_knn, return_counts=True)
    id_samp = pps(get_knn, rnn, 1)
    return get_knn, rnn, id_samp


def _embed_with_landmarks(
    X: np.ndarray,
    X_landmarks: np.ndarray,
    Y_landmarks: np.ndarray,
    scale: np.ndarray,
    n_components: int,
):
    if X.shape[0] == 0:
        return np.empty((0, n_components))

    top_k = min(n_components + 1, X_landmarks.shape[0])
    near_dis, near_samp = NearestNeighbors(n_neighbors=top_k).fit(
        X_landmarks
    ).kneighbors(X)

    embedding = np.zeros((X.shape[0], n_components))
    for i in range(X.shape[0]):
        near_top_k = near_samp[i]
        top_X = X_landmarks[near_top_k]
        top_Y = Y_landmarks[near_top_k]
        nearest_scale = float(scale[near_top_k[0]])
        n_dis = near_dis[i, 0] * nearest_scale
        embedding[i] = np.asarray(clle(top_X, top_Y, X[i], n_dis)).reshape(-1)
    return embedding


def _fit_embedding(
    X: np.ndarray,
    n_components: int,
    n_neighbors: int,
    normalize: bool,
    large: bool,
    init: str,
    agg_coef: float,
    max_iter: int,
):
    X_unique, inverse_indices, scaler = _prepare_training_data(X, normalize)
    n_samples, n_features = X_unique.shape
    resolved_init = _validate_parameters(
        n_samples=n_samples,
        n_features=n_features,
        n_components=n_components,
        n_neighbors=n_neighbors,
        normalize=normalize,
        large=large,
        init=init,
        agg_coef=agg_coef,
        max_iter=max_iter,
    )

    get_knn, rnn, id_samp = _compute_landmarks(X_unique, n_components, n_neighbors)
    X_landmarks = X_unique[id_samp]

    learning_fn = learning_l if large else learning_s
    Y_landmarks, k2 = learning_fn(
        X_landmarks,
        n_neighbors,
        get_knn,
        rnn,
        id_samp,
        n_components,
        resolved_init,
        agg_coef,
        max_iter,
    )

    scale_neighbors = min(k2, max(1, X_landmarks.shape[0] - 1))
    scale = opt_scale(X_landmarks, Y_landmarks, scale_neighbors)

    if n_neighbors > 0:
        id_rest = np.setdiff1d(range(n_samples), id_samp)
        Y_unique = np.zeros((n_samples, n_components))
        Y_unique[id_samp] = Y_landmarks
        Y_unique[id_rest] = _embed_with_landmarks(
            X_unique[id_rest],
            X_landmarks,
            Y_landmarks,
            scale,
            n_components,
        )
    else:
        Y_unique = Y_landmarks

    return {
        "embedding": Y_unique[inverse_indices],
        "landmarks": X_landmarks,
        "landmark_embedding": Y_landmarks,
        "landmark_scale": scale,
        "scaler": scaler,
        "resolved_init": resolved_init,
    }


class SUDE(TransformerMixin, BaseEstimator):
    """Scalable manifold learning estimator with a scikit-learn style API."""

    def __init__(
        self,
        n_components: int = 2,
        *,
        n_neighbors: int = 20,
        normalize: bool = True,
        large: bool = False,
        init: Literal["spectral", "le", "pca", "mds"] = "spectral",
        agg_coef: float = 1.2,
        max_iter: int = 50,
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.normalize = normalize
        self.large = large
        self.init = init
        self.agg_coef = agg_coef
        self.max_iter = max_iter

    def fit(self, X, y=None):
        """Fit the SUDE embedding on X."""
        X = self._validate_data(X, ensure_2d=True, dtype=np.float64)
        fit_result = _fit_embedding(
            X=X,
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            normalize=self.normalize,
            large=self.large,
            init=self.init,
            agg_coef=self.agg_coef,
            max_iter=self.max_iter,
        )

        self.embedding_ = fit_result["embedding"]
        self.X_landmarks_ = fit_result["landmarks"]
        self.Y_landmarks_ = fit_result["landmark_embedding"]
        self.landmark_scale_ = fit_result["landmark_scale"]
        self.scaler_ = fit_result["scaler"]
        self.X_fit_ = np.array(X, copy=True)
        self.init_ = fit_result["resolved_init"]
        self.n_landmarks_ = self.X_landmarks_.shape[0]
        self.n_iter_ = self.max_iter
        return self

    def fit_transform(self, X, y=None):
        """Fit the model on X and return the learned embedding."""
        self.fit(X, y=y)
        return self.embedding_

    def transform(self, X):
        """Embed new samples using the fitted SUDE landmarks."""
        check_is_fitted(
            self,
            attributes=[
                "embedding_",
                "X_landmarks_",
                "Y_landmarks_",
                "landmark_scale_",
            ],
        )
        X = self._validate_data(X, reset=False, ensure_2d=True, dtype=np.float64)
        if X.shape == self.X_fit_.shape and np.array_equal(X, self.X_fit_):
            return np.array(self.embedding_, copy=True)
        X_unique, inverse_indices = np.unique(X, axis=0, return_inverse=True)
        if self.scaler_ is not None:
            X_unique = self.scaler_.transform(X_unique)
        embedding = _embed_with_landmarks(
            X_unique,
            self.X_landmarks_,
            self.Y_landmarks_,
            self.landmark_scale_,
            self.n_components,
        )
        return embedding[inverse_indices]

    def get_feature_names_out(self, input_features=None):
        """Return output feature names for the embedding coordinates."""
        return np.asarray(
            [f"sude{i}" for i in range(self.n_components)],
            dtype=object,
        )


def sude(
    X: np.ndarray,
    no_dims: int = 2,
    k1: int = 20,
    normalize: bool = True,
    large: bool = False,
    initialize: Literal["le", "pca", "mds"] = "le",
    agg_coef: float = 1.2,
    T_epoch: int = 50,
):
    """
    Backward-compatible function interface for computing a SUDE embedding.

    This wrapper preserves the original parameter names while delegating the
    computation to the sklearn-style SUDE estimator.
    """
    init = "spectral" if initialize == "le" else initialize
    estimator = SUDE(
        n_components=no_dims,
        n_neighbors=k1,
        normalize=normalize,
        large=large,
        init=init,
        agg_coef=agg_coef,
        max_iter=T_epoch,
    )
    return estimator.fit_transform(X)
