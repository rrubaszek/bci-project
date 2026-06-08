from src.models.unsupervised.base import BaseModelPipeline
from src.preprocessing.unsupervised.extract.base import BaseExtractor
import mne
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


class GmmPipeline(BaseModelPipeline):
    """Potok dla modelu Mieszanin Gaussowskich (GMM)."""

    def __init__(
        self, extractor: BaseExtractor, n_clusters: int = 2, pca_components: int = 3
    ) -> None:
        super().__init__(extractor)
        self.n_clusters: int = n_clusters
        self.pca_components: int = pca_components

        self._scaler: StandardScaler = StandardScaler()
        self._pca: PCA = PCA(n_components=self.pca_components, random_state=42)
        self._model: GaussianMixture = GaussianMixture(
            n_components=self.n_clusters,
            covariance_type="full",
            random_state=42,
            max_iter=100,
        )

    def _scale_features(self, features: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            return self._scaler.fit_transform(features)
        return self._scaler.transform(features)

    def _reduce_dimensions(
        self, scaled_data: np.ndarray, fit: bool = False
    ) -> np.ndarray:
        if fit:
            return self._pca.fit_transform(scaled_data)
        return self._pca.transform(scaled_data)

    def fit(self, raw_data: mne.io.Raw | list[mne.io.Raw]) -> None:
        features: np.ndarray = self._extract_and_prepare(raw_data)
        scaled: np.ndarray = self._scale_features(features, fit=True)
        reduced: np.ndarray = self._reduce_dimensions(scaled, fit=True)
        self._model.fit(reduced)

    def predict(self, raw_data: mne.io.Raw | list[mne.io.Raw]) -> np.ndarray:
        features: np.ndarray = self._extract_and_prepare(raw_data)
        scaled: np.ndarray = self._scale_features(features, fit=False)
        reduced: np.ndarray = self._reduce_dimensions(scaled, fit=False)
        return self._model.predict(reduced)
