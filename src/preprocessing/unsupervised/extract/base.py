from abc import ABC, abstractmethod
import mne
import numpy as np


class BaseExtractor(ABC):
    """Abstrakcyjna klasa bazowa (Interfejs) dla wszystkich metod ekstrakcji cech."""

    def __init__(self, window_duration: float, window_overlap: float) -> None:
        self.window_duration: float = window_duration
        self.window_overlap: float = window_overlap

    def _create_sliding_windows(self, raw: mne.io.Raw) -> mne.Epochs:
        """Prywatna metoda pomocnicza dzieląca sygnał na okna czasowe (epoki)."""
        return mne.make_fixed_length_epochs(
            raw,
            duration=self.window_duration,
            overlap=self.window_overlap,
            preload=True,
            verbose=False,
        )

    def _compute_band_power(self, epochs: mne.Epochs) -> np.ndarray:
        """Prywatna metoda pomocnicza licząca średnią moc pasm Mu i Beta."""
        spectrum = epochs.compute_psd(
            method="welch", fmin=8.0, fmax=30.0, verbose=False
        )
        psds, freqs = spectrum.get_data(return_freqs=True)

        mu_mask: np.ndarray = (freqs >= 8.0) & (freqs <= 12.0)
        beta_mask: np.ndarray = (freqs >= 13.0) & (freqs <= 30.0)

        mu_power: np.ndarray = psds[..., mu_mask].mean(axis=-1)
        beta_power: np.ndarray = psds[..., beta_mask].mean(axis=-1)

        return np.hstack((mu_power, beta_power))

    @abstractmethod
    def extract(self, raw_filtered_data: mne.io.Raw) -> np.ndarray:
        """Główna metoda ekstrahująca macierz cech z przefiltrowanego sygnału."""
        pass
