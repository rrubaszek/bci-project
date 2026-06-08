import mne
import numpy as np

from src.paths import EMOTIV_CLEANED, EMOTIV_RAW

def load_raw_data():
    data: list[mne.io.Raw] = []
    for edf_file in EMOTIV_RAW.glob("*.edf"):
        if edf_file.name.endswith(".md.edf"):
            continue

        print(f"Processing {edf_file.name}")

        raw: mne.io.Raw = mne.io.read_raw_edf(edf_file, preload=True)
        data.append(raw)
        
    return data

def run():
    raw = load_raw_data()
    for r in raw:
        r.filter(4.0, 40.0, fir_design="firwin", skip_by_annotation="edge")
        r.resample(128)
        r.save(EMOTIV_CLEANED / f"{r.info['subject_info']['id']}.fif", overwrite=True)
    
if __name__ == "__main__":
    run()