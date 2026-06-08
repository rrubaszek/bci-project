from src.models.unsupervised.base import BaseModelPipeline
from src.preprocessing.unsupervised.extract.base import BaseExtractor
import mne
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class HmmPipeline(BaseModelPipeline):
    """Potok dla Ukrytych Modeli Markowa (HMM)."""

    def __init__(
        self, extractor: BaseExtractor, n_states: int = 2, pca_components: int = 1
    ) -> None:
        super().__init__(extractor)
        self.n_states: int = n_states
        self.pca_components: int = pca_components

        self._scaler: StandardScaler = StandardScaler()
        self._pca: PCA = PCA(n_components=self.pca_components, random_state=42)

        from hmmlearn.hmm import GaussianHMM

        self._model: GaussianHMM = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            random_state=42,
            n_iter=100,
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
