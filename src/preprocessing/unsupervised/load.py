from src.preprocessing.unsupervised.filter import preprocess_eeg
import mne

from src.paths import EMOTIV_CLEANED, EMOTIV_RAW


def load_raw_data() -> dict[str, mne.io.Raw]:
    data: dict[str, mne.io.Raw] = {}
    for edf_file in EMOTIV_RAW.glob("*.edf"):
        if edf_file.name.endswith(".md.edf"):
            continue

        print(f"Processing {edf_file.name}")

        raw: mne.io.Raw = mne.io.read_raw_edf(edf_file, preload=True)
        data[edf_file.name.split(".")[0]] = raw

    return data


def run():
    raw = load_raw_data()
    for name, r in raw.items():
        cleaned_r = preprocess_eeg(r, bad_channels=None)

        if cleaned_r is None:
            raise ValueError("Nie udało się przetworzyć danych")

        cleaned_r.resample(128)
        cleaned_r.save(EMOTIV_CLEANED / f"{name}.fif", overwrite=True)
        print(f"Pomyślnie przetworzono i zapisano: {name}")


if __name__ == "__main__":
    run()
