from src.preprocessing.unsupervised.extract.base import BaseExtractor
from abc import ABC, abstractmethod
import mne
import numpy as np


class BaseModelPipeline(ABC):
    """
    Abstrakcyjna klasa bazowa dla potoków modeli.
    Posiada zintegrowany mechanizm ekstrakcji cech z obiektów mne.io.Raw.
    """

    def __init__(self, extractor: BaseExtractor) -> None:
        self.extractor: BaseExtractor = extractor

    def _extract_and_prepare(
        self, raw_data: mne.io.Raw | list[mne.io.Raw]
    ) -> np.ndarray:
        """
        Uniwersalnie przygotowuje cechy. Jeśli przekazano listę plików (np. dla GMM),
        ekstrahuje cechy dla każdego z nich i łączy je w jedną dużą macierz.
        """
        if isinstance(raw_data, list):
            features_list: list[np.ndarray] = [
                self.extractor.extract(raw) for raw in raw_data
            ]
            return np.vstack(features_list)

        return self.extractor.extract(raw_data)

    @abstractmethod
    def fit(self, raw_data: mne.io.Raw | list[mne.io.Raw]) -> None:
        """Ekstrahuje cechy i trenuje komponenty transformujące oraz model."""
        pass

    @abstractmethod
    def predict(self, raw_data: mne.io.Raw | list[mne.io.Raw]) -> np.ndarray:
        """Ekstrahuje cechy i generuje predykcje (klastry/stany) dla podanych danych."""
        pass
