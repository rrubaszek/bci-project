from src.preprocessing.unsupervised.constants import EEG_CHANNELS, EMOTIV_MONTAGE
import mne


def preprocess_eeg(
    raw_data: mne.io.Raw,
    bad_channels: list[str] | None = None,
    l_freq: float = 8.0,
    h_freq: float = 30.0,
    notch_freq: float = 50.0,
) -> mne.io.Raw | None:
    """
    Filtruje i przygotowuje surowy sygnał EEG z kasku Emotiv.

    Parametry:
    - raw_data: obiekt mne.io.Raw (wczytany plik .edf).
    - bad_channels: lista nazw niedziałających kanałów, np. ['T7', 'O1'].
    - l_freq: dolna granica filtru pasmowoprzepustowego w Hz (domyślnie 8.0).
    - h_freq: górna granica filtru pasmowoprzepustowego w Hz (domyślnie 30.0).
    - notch_freq: częstotliwość filtru Notch do usunięcia szumu z sieci elektrycznej.

    Zwraca:
    - preprocessed_signal: przefiltrowany i naprawiony obiekt mne.io.Raw lub None w przypadku błędu.
    """

    preprocessed_signal: mne.io.Raw = raw_data.copy()

    if not _pick_eeg_channels(preprocessed_signal):
        return None

    _setup_montage_and_types(preprocessed_signal)

    _mark_bad_channels(preprocessed_signal, bad_channels)

    _apply_filters(preprocessed_signal, l_freq, h_freq, notch_freq)

    _interpolate_bad_channels(preprocessed_signal, bad_channels)

    return preprocessed_signal


def _pick_eeg_channels(raw: mne.io.Raw) -> bool:
    try:
        raw.pick_channels(EEG_CHANNELS)
        return True
    except ValueError as e:
        print(
            f"Błąd wyboru kanałów. Upewnij się, że nazwy w pliku .edf są poprawne: {e}"
        )
        return False


def _setup_montage_and_types(raw: mne.io.Raw) -> None:
    channel_types: dict[str, str] = {ch: "eeg" for ch in EEG_CHANNELS}
    raw.set_channel_types(channel_types)

    montage: mne.channels.DigMontage = mne.channels.make_standard_montage(
        EMOTIV_MONTAGE
    )
    raw.set_montage(montage)


def _mark_bad_channels(raw: mne.io.Raw, bad_channels: list[str] | None) -> None:
    if bad_channels is not None:
        raw.info["bads"] = bad_channels


def _apply_filters(
    raw: mne.io.Raw, l_freq: float, h_freq: float, notch_freq: float
) -> None:
    raw.notch_filter(freqs=notch_freq, verbose=False)
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin", verbose=False)


def _interpolate_bad_channels(raw: mne.io.Raw, bad_channels: list[str] | None) -> None:
    if bad_channels is not None and len(bad_channels) > 0:
        raw.interpolate_bads(reset_bads=True, verbose=False)
