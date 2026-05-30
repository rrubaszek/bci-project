from src.preprocessing.unsupervised.extract.extractors.ica_extractor import IcaExtractor
from src.preprocessing.unsupervised.extract.extractors.psd_extractor import PsdExtractor
from src.preprocessing.unsupervised.extract.base import BaseExtractor
from enum import StrEnum


class ExtractyorType(StrEnum):
    PSD = "psd"
    ICA = "ica"


class ExtractorFactory:
    """Fabryka odpowiedzialna za powoływanie odpowiednich obiektów ekstraktorów."""

    @staticmethod
    def create_extractor(
        extractor_type: ExtractyorType,
        window_duration: float | int,
        window_overlap: float | int,
        **kwargs: float | int,
    ) -> BaseExtractor:

        if extractor_type is ExtractyorType.PSD:
            return PsdExtractor(
                window_duration=window_duration, window_overlap=window_overlap
            )

        elif extractor_type is ExtractyorType.ICA:
            n_comp: int = int(kwargs.get("n_components", 10))
            return IcaExtractor(
                n_components=n_comp,
                window_duration=window_duration,
                window_overlap=window_overlap,
            )
