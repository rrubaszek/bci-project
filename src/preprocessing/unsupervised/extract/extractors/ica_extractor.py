from src.preprocessing.unsupervised.extract.base import BaseExtractor
import mne
import numpy as np
from mne.preprocessing import ICA


class IcaExtractor(BaseExtractor):
    def __init__(
        self,
        n_components: int = 10,
        window_duration: float = 0.5,
        window_overlap: float = 0.4,
    ) -> None:
        # Wywołujemy konstruktor klasy bazowej
        super().__init__(window_duration, window_overlap)
        self.n_components: int = n_components

    def extract(self, raw_filtered_data: mne.io.Raw) -> np.ndarray:
        ica: ICA = ICA(
            n_components=self.n_components,
            random_state=42,
            max_iter="auto",
            method="infomax",
        )
        ica.fit(raw_filtered_data, verbose=False)
        sources: mne.io.Raw = ica.get_sources(raw_filtered_data)

        # Generowanie okien na źródłach ICA i obliczenie mocy
        epochs: mne.Epochs = self._create_sliding_windows(sources)
        return self._compute_band_power(epochs)
