from src.models.unsupervised.base import BaseModelPipeline
from src.preprocessing.unsupervised.extract.base import BaseExtractor
import mne
import numpy as np
from sklearn.decomposition import PCA
from hmmlearn.hmm import GaussianHMM


class HmmPipeline(BaseModelPipeline):
    """Potok dla Ukrytych Modeli Markowa (HMM)."""

    def __init__(
        self,
        extractor: BaseExtractor,
        n_states: int = 2,
        pca_components: int | float = 0.9,
    ) -> None:
        super().__init__(extractor)
        self.n_states: int = n_states
        self.pca_components: int | float = pca_components

        self._pca: PCA = PCA(n_components=self.pca_components, random_state=42)

        self._model: GaussianHMM = GaussianHMM(
            n_components=self.n_states,
            covariance_type="diag",
            min_covar=0.001,
            random_state=42,
            n_iter=100,
        )

    def _reduce_dimensions(
        self, scaled_data: np.ndarray, fit: bool = False
    ) -> np.ndarray:
        if fit:
            return self._pca.fit_transform(scaled_data)
        return self._pca.transform(scaled_data)

    def fit(self, raw_data: mne.io.Raw | list[mne.io.Raw]) -> None:
        features: np.ndarray = self._extract_and_prepare(raw_data)
        reduced: np.ndarray = self._reduce_dimensions(features, fit=True)
        self._model.fit(reduced)

    def predict(self, raw_data: mne.io.Raw | list[mne.io.Raw]) -> np.ndarray:
        features: np.ndarray = self._extract_and_prepare(raw_data)
        reduced: np.ndarray = self._reduce_dimensions(features, fit=False)
        return self._model.predict(reduced)
