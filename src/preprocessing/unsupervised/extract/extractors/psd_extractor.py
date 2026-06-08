from src.preprocessing.unsupervised.extract.base import BaseExtractor
import mne
import numpy as np


class PsdExtractor(BaseExtractor):
    def extract(self, raw_filtered_data: mne.io.Raw) -> np.ndarray:
        epochs: mne.Epochs = self._create_sliding_windows(raw_filtered_data)
        return self._compute_band_power(epochs)
