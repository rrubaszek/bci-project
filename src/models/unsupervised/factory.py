from enum import StrEnum
from src.models.unsupervised.model_pipelines.hmm import HmmPipeline
from src.models.unsupervised.model_pipelines.gmm import GmmPipeline
from src.models.unsupervised.base import BaseModelPipeline
from src.preprocessing.unsupervised.extract.base import BaseExtractor


class ModelTypes(StrEnum):
    HMM = "HMM"
    GMM = "GMM"


class ModelFactory:
    """Fabryka odpowiedzialna za bezpieczne i jednolite powoływanie potoków E2E."""

    @staticmethod
    def create_pipeline(
        model_type: ModelTypes, extractor: BaseExtractor, **kwargs: int
    ) -> BaseModelPipeline:
        """
        Tworzy potok z wbudowanym ekstraktorem.

        Parametry:
        - model_type: 'gmm' lub 'hmm'.
        - extractor: instancja obiektu dziedziczącego po BaseExtractor (z extract.py).
        - kwargs: parametry modeli (n_clusters, n_states, pca_components).
        """

        if model_type is ModelTypes.GMM:
            return GmmPipeline(
                extractor=extractor,
                n_clusters=kwargs.get("n_clusters", 2),
                pca_components=kwargs.get("pca_components", 3),
            )
        elif model_type is ModelTypes.HMM:
            return HmmPipeline(
                extractor=extractor,
                n_states=kwargs.get("n_states", 2),
                pca_components=kwargs.get("pca_components", 1),
            )
