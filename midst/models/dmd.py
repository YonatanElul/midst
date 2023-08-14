# This code is a python implementation of the Measure-Preserving Extended Dynamic Mode Decomposition algorithm,
# based on the official MATLAB implementation by the authors, which can be found here:
# https://github.com/MColbrook/Measure-preserving-Extended-Dynamic-Mode-Decomposition.
# The paper can be found here:
# "The mpEDMD Algorithm for Data-Driven Computations of Measure-Preserving Dynamical Systems",
# Matthew J. Colbrook, SIAM, 2023
# https://epubs.siam.org/doi/full/10.1137/22M1521407
from abc import abstractmethod
from typing import Tuple, Union, Optional
from sklearn.preprocessing import PolynomialFeatures

import numpy as np
import scipy as sp


class Transform:
    """
    Abstract class for feature transformation for the DMD models
    """

    def __init__(self):
        pass

    @abstractmethod
    def transform(
            self,
            X: np.ndarray,
            *args,
            **kwargs,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def inverse_transform(
            self,
            embedding: np.ndarray,
            initial_condition: Optional[np.ndarray] = None,
            *args,
            **kwargs,
    ) -> np.ndarray:
        pass


class TimeDelayTransform(Transform):
    """
    Implements a delay embedding transformation (i.e., Takens transformation)
    """
    def __init__(
            self,
            k_delays: int = 1,
    ):
        super().__init__()
        self._k = k_delays

    def transform(
            self,
            X: np.ndarray,
            *args,
            **kwargs,
    ) -> np.ndarray:
        # Embed
        embeddings = [
            X[k:] - X[:-k]
            for k in range(1, self._k + 1)
        ]
        min_embedding_size = min([emb.shape[0] for emb in embeddings])
        embeddings = [
            emb[:min_embedding_size]
            for emb in embeddings
        ]
        embeddings = np.concatenate(
            embeddings,
            axis=1,
        )

        return embeddings

    def inverse_transform(
            self,
            embedding: np.ndarray,
            initial_condition: Optional[np.ndarray] = None,
            *args,
            **kwargs,
    ) -> np.ndarray:
        assert initial_condition is not None
        X = np.cumsum(
            np.concatenate([initial_condition, embedding], axis=0),
            axis=0,
        )

        return X


class PolynomialFeaturesTransform(Transform):
    """
    Implements a polynomial embeddings transformation
    """

    def __init__(
            self,
            degree: int = 2,
    ):
        super().__init__()
        self._degree = degree
        self._poly = PolynomialFeatures(
            degree=degree,
            interaction_only=False,
            include_bias=True,
        )

    def transform(
            self,
            X: np.ndarray,
            *args,
            **kwargs,
    ) -> np.ndarray:
        self._poly.fit(X)
        embeddings = self._poly.transform(X)
        return embeddings

    def inverse_transform(
            self,
            embedding: np.ndarray,
            initial_condition: Optional[np.ndarray] = None,
            *args,
            **kwargs,
    ) -> np.ndarray:
        assert initial_condition is not None
        X = embedding[:, 1:(initial_condition.shape[1] + 1)]
        return X


class SVDTransform(Transform):
    """
    Implements a feature transformation which linearly projects the input into the
    left singular vector matrix of the data.
    """

    def __init__(self):
        super().__init__()
        self._u = None
        self._v = None

    def transform(
            self,
            X: np.ndarray,
            *args,
            **kwargs,
    ) -> np.ndarray:
        # Embed
        U, S, Vh = np.linalg.svd(X)
        self._u = U
        self._v = Vh.T

        embeddings = X @ self._v
        return embeddings

    def inverse_transform(
            self,
            embedding: np.ndarray,
            initial_condition: Optional[np.ndarray] = None,
            *args,
            **kwargs,
    ) -> np.ndarray:
        X = embedding @ self._v.T
        return X


class PolarTransform(Transform):
    """
    Implements a polar-coordinates transformation
    """

    def __init__(self):
        super().__init__()

    def transform(
            self,
            X: np.ndarray,
            *args,
            **kwargs,
    ) -> np.ndarray:
        time = kwargs['times']
        time = np.tile(time[:, None], (1, X.shape[1]))
        r = np.sqrt(np.power(time, 2) + np.power(X, 2))
        theta = np.arctan2(X, time)
        embeddings = np.concatenate([r, theta], axis=1)
        return embeddings

    def inverse_transform(
            self,
            embedding: np.ndarray,
            initial_condition: Optional[np.ndarray] = None,
            *args,
            **kwargs,
    ) -> np.ndarray:
        # X is the y in the (x, y) -> (r, theta) coordinate transformation
        n = initial_condition.shape[1]
        r = embedding[:, :n]
        theta = embedding[:, n:]
        X = r * np.sin(theta)
        return X


class BaseDMD:
    """
    An abstract class for DMD-based models
    """
    def __init__(self):
        self._koopman = None
        self._modes = None
        self._spectrum = None
        self._gamma = None
        self._inv = None

    @property
    def koopman(self) -> Union[None, np.ndarray]:
        return self._koopman

    @property
    def modes(self) -> Union[None, np.ndarray]:
        return self._modes

    @property
    def spectrum(self) -> Union[None, np.ndarray]:
        return self._spectrum

    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass


class EDMD(BaseDMD):
    """
    An implementation of the Extended Dynamic Mode Decomposition (EDMD) algorithm
    """
    def __init__(
            self,
            eigvals_cutoff_threshold: float = 1e-2,
    ):
        super(EDMD, self).__init__()
        self._eigvals_cutoff_threshold = eigvals_cutoff_threshold

    @staticmethod
    def _get_Psi(X: np.ndarray) -> Tuple[np.ndarray]:
        psi_x = X[:-1]
        psi_y = X[1:]

        return psi_x, psi_y

    @staticmethod
    def _get_DMD_matrices(
            psi_x: np.ndarray,
            psi_y: np.ndarray,
    ) -> Tuple[np.ndarray]:
        m = psi_x.shape[0]
        G = (1 / m) * psi_x.T @ psi_x
        A = (1 / m) * psi_x.T @ psi_y

        return G, A

    def fit(self, X: np.ndarray) -> None:
        psi_x, psi_y = self._get_Psi(X=X)

        # Get the DMD matrices
        G, A = self._get_DMD_matrices(
            psi_x=psi_x,
            psi_y=psi_y
        )

        # Get the Koopman operator
        koopman, _, _, _ = np.linalg.lstsq(G, A)
        eigvals, eigvec = np.linalg.eig(koopman)

        self._koopman = koopman
        self._modes = eigvec
        self._spectrum = eigvals
        self._gamma = np.diag(eigvals)
        self._inv = sp.linalg.pinv(self._modes)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Transform X if relevant
        predictions = np.linalg.multi_dot(
            [self._modes, self._gamma, self._inv, X.T]
        )
        predictions = np.real(predictions.T)
        return predictions


class mpEDMD(EDMD):
    """
    An implementation of the Measure Preserving Extended Dynamic Mode Decomposition (mpEDMD) algorithm
    """
    @staticmethod
    def _get_mpEDMD(
            G: np.ndarray,
            A: np.ndarray,
            eigvals_cutoff_threshold: float = 1e-2,
    ) -> Tuple[np.ndarray]:
        # Make sure G is Hermitian
        G = (G + G.T) / 2
        eigvals, eigvecs = np.linalg.eigh(G)

        # Cut-off eigenvalues and their respective eigenvectors, whose magnitude is negligible,
        # to avoid NaNs in the computation of Gsqrt and GsqrtI
        valid_inds = eigvals.real > eigvals_cutoff_threshold
        valid_eigvals = eigvals[valid_inds]
        valid_eigvecs = eigvecs[:, valid_inds]

        # G^{1/2}
        Gsqrt = valid_eigvecs @ np.diag(np.sqrt(valid_eigvals)) @ valid_eigvecs.T

        # G^{-1/2}
        GsqrtI = valid_eigvecs @ np.diag(np.sqrt((1. / valid_eigvals))) @ valid_eigvecs.T

        arg = GsqrtI @ A.T @ GsqrtI
        U, Sv, Vh = np.linalg.svd(arg)

        # Get the Koopman operator
        mpK = GsqrtI @ Vh.T @ U.T @ Gsqrt

        # schur usedto ensure orthonormal basis
        mpV, mpD = sp.linalg.schur((Vh.T @ U.T), output='complex')
        mpV = GsqrtI @ mpV
        mpD = np.diag(np.diag(mpD))

        return mpK, mpV, mpD

    def fit(self, X: np.ndarray) -> None:
        psi_x, psi_y = self._get_Psi(X=X)

        # Get the DMD matrices
        G, A = self._get_DMD_matrices(
            psi_x=psi_x,
            psi_y=psi_y
        )

        # Get the Koopman operator
        mpK, mpV, mpD = self._get_mpEDMD(
            G=G,
            A=A,
            eigvals_cutoff_threshold=self._eigvals_cutoff_threshold,
        )

        self._koopman = mpK
        self._modes = mpV
        self._spectrum = np.diag(mpD)
        self._gamma = mpD
        self._inv = sp.linalg.pinv(mpV)
