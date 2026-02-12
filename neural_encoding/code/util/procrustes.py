import numpy as np
import torch
from scipy.linalg import orthogonal_procrustes
from sklearn.base import BaseEstimator, TransformerMixin

torch.set_grad_enabled(False)


def orthogonal_procrustes_torch(A, B, check_finite=True):
    u, w, vt = torch.linalg.svd((B.T @ torch.conj(A)).T)
    R = u @ vt
    scale = w.sum()
    return R, scale


def procrustes(data1, data2, device="cuda"):
    # https://discuss.pytorch.org/t/is-there-an-orthogonal-procrustes-for-pytorch/131365
    # https://gist.github.com/mkocabas/54ea2ff3b03260e3fedf8ad22536f427
    mtx1 = np.array(data1, dtype=np.double, copy=True)
    mtx2 = np.array(data2, dtype=np.double, copy=True)

    if mtx1.ndim != 2 or mtx2.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")
    if mtx1.shape != mtx2.shape:
        raise ValueError("Input matrices must be of same shape")
    if mtx1.size == 0:
        raise ValueError("Input matrices must be >0 rows and >0 cols")

    # translate all the data to the origin
    mtx1 -= np.mean(mtx1, 0)
    mtx2 -= np.mean(mtx2, 0)

    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")

    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    mtx1 /= norm1
    mtx2 /= norm2

    # transform mtx2 to minimize disparity
    # R, s = orthogonal_procrustes(mtx1, mtx2)
    R, s = orthogonal_procrustes_torch(
        torch.tensor(mtx1, device=device), torch.tensor(mtx2, device=device)
    )
    R = R.numpy(force=True)
    s = s.numpy(force=True)

    # mtx2 = np.dot(mtx2, R.T) * s    # HERE, the projected mtx2 is estimated.
    # measure the dissimilarity between the two datasets
    # disparity = np.sum(np.square(mtx1 - mtx2))

    return R, s


class ProcrustesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.R_ = None
        self.s_ = None
        self.disparity_ = None

    def fit(self, X, Y):
        """
        Fit the model to the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The first dataset.
        Y : array-like, shape (n_samples, n_features)
            The second dataset.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = np.array(X, dtype=np.double, copy=True)
        Y = np.array(Y, dtype=np.double, copy=True)

        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError("Input matrices must be two-dimensional")
        if X.shape != Y.shape:
            raise ValueError("Input matrices must be of same shape")
        if X.size == 0:
            raise ValueError("Input matrices must be >0 rows and >0 cols")

        # Translate all the data to the origin
        X -= np.mean(X, 0)
        Y -= np.mean(Y, 0)

        norm1 = np.linalg.norm(X)
        norm2 = np.linalg.norm(Y)

        if norm1 == 0 or norm2 == 0:
            raise ValueError("Input matrices must contain >1 unique points")

        # Change scaling of data (in rows) such that trace(X*X') = 1
        X /= norm1
        Y /= norm2

        # Transform Y to minimize disparity
        self.R_, self.s_ = orthogonal_procrustes(X, Y)
        Y_transformed = np.dot(Y, self.R_.T) * self.s_

        # Measure the dissimilarity between the two datasets
        self.disparity_ = np.sum(np.square(X - Y_transformed))

        return self

    def transform(self, Y):
        """
        Transform the second dataset to align with the first dataset.

        Parameters
        ----------
        Y : array-like, shape (n_samples, n_features)
            The second dataset.

        Returns
        -------
        Y_transformed : array-like, shape (n_samples, n_features)
            The transformed second dataset.
        """
        if self.R_ is None or self.s_ is None:
            raise ValueError("The model has not been fitted yet.")

        Y = np.array(Y, dtype=np.double, copy=True)
        Y -= np.mean(Y, 0)
        Y /= np.linalg.norm(Y)
        Y_transformed = np.dot(Y, self.R_.T) * self.s_

        return Y_transformed
