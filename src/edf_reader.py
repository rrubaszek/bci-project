"""Minimal EDF reader (no mne dependency) — reads Emotiv EDF, returns physical
signals for selected channels + sampling rate."""
import numpy as np

def read_edf(path, channels):
    with open(path, 'rb') as f:
        header = f.read(256)
        ns = int(header[252:256])
        n_records = int(header[236:244])
        record_dur = float(header[244:252])

        labels = [header[16*i:16*(i+1)].decode('ascii').strip()
                  for i, header in enumerate([f.read(0)]*0)]  # placeholder, real read below

    with open(path, 'rb') as f:
        f.seek(256)
        labels = [f.read(16).decode('ascii').strip() for _ in range(ns)]
        transducer = [f.read(80) for _ in range(ns)]
        units = [f.read(8).decode('ascii').strip() for _ in range(ns)]
        phys_min = [float(f.read(8)) for _ in range(ns)]
        phys_max = [float(f.read(8)) for _ in range(ns)]
        dig_min = [int(f.read(8)) for _ in range(ns)]
        dig_max = [int(f.read(8)) for _ in range(ns)]
        prefilter = [f.read(80) for _ in range(ns)]
        nr_samples = [int(f.read(8)) for _ in range(ns)]
        reserved = [f.read(32) for _ in range(ns)]

        data = {ch: [] for ch in channels}
        idx_map = {lbl: i for i, lbl in enumerate(labels)}

        for rec in range(n_records):
            for i in range(ns):
                n = nr_samples[i]
                raw = np.frombuffer(f.read(n*2), dtype='<i2')
                lbl = labels[i]
                if lbl in data:
                    scale = (phys_max[i] - phys_min[i]) / (dig_max[i] - dig_min[i])
                    offset = phys_min[i] - dig_min[i]*scale
                    data[lbl].append(raw.astype(np.float64)*scale + offset)

    sfreq = nr_samples[idx_map[channels[0]]] / record_dur
    out = np.vstack([np.concatenate(data[ch]) for ch in channels])  # (n_ch, n_samples)
    return out, sfreq
