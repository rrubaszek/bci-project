from pathlib import Path
import mne
import numpy as np

from src.paths import EMOTIV_CLEANED, EMOTIV_RAW

TARGET_SFREQ = 250  # BCI IV 2b

def main():
    for edf_file in EMOTIV_RAW.glob("*.edf"):
        if edf_file.name.endswith(".md.edf"):
            continue

        print(f"Processing {edf_file.name}")

        raw = mne.io.read_raw_edf(edf_file, preload=True)

        if not {"FC5", "FC6"}.issubset(raw.ch_names):
            print(f"Skipping {edf_file.name}: FC5/FC6 missing")
            continue

        data = raw.get_data(
            picks=["FC5", "FC6"]
        )

        fc5 = data[0]
        fc6 = data[1]
        cz = (fc5 + fc6) / 2

        mapped = np.vstack([
            fc5,   # C3 proxy
            cz,    # Cz proxy
            fc6,   # C4 proxy
        ])

        info = mne.create_info(
            ch_names=["C3", "Cz", "C4"],
            sfreq=raw.info["sfreq"],
            ch_types="eeg",
        )

        new_raw = mne.io.RawArray(mapped, info)

        if new_raw.info["sfreq"] != TARGET_SFREQ:
            new_raw.resample(TARGET_SFREQ)

        out_file = EMOTIV_CLEANED / f"{edf_file.stem}_bciciv2b.edf"

        mne.export.export_raw(
            out_file,
            new_raw,
            fmt="edf",
            overwrite=True,
        )

        print(f"Saved -> {out_file}")
    
if __name__ == "__main__":
    main()