"""
Analiza istotności kanałów (ANOVA) na cechach Welch PSD (pasma Mu/Beta) per kanał

Etykiety stanu (niska/wysoka aktywność) wynikają z protokołu eksperymentu:
  0.0 - 2.5 s : spoczynek (baseline)
  2.5 - 5.0 s : zadanie ruchowe (wysoka aktywność)
  5.0 - 7.5+s : spoczynek (post-baseline)

Cechy: Welch PSD, okno 1.0 s, nakładanie 0.5 s (jak w sekcji 5.2 artykułu),
pasmo Mu (8-12 Hz) i Beta (13-30 Hz), per kanał (14 kanałów Emotiv Epoc+).

Test: scipy.stats.f_oneway (jednoczynnikowa ANOVA, 2 grupy: rest vs task) per
kolumna cechy (kanał x pasmo). Korekta wielokrotnych porówań: Holm-Bonferroni
(28 testów: 14 kanałów x 2 pasma).
"""
import sys
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt, iirnotch, welch
from scipy.stats import f_oneway

from edf_reader import read_edf

EEG_CHANNELS = ["AF3","F7","F3","FC5","T7","P7","O1","O2","P8","T8",
                "FC6","F4","F8","AF4"]

SESSIONS = ["lewy1", "lewy2", "prawy1", "prawy2"]

# Użyto 'r' przed stringiem (raw string) zapobiega to błędom znaków ucieczki w Windows
DATA_DIR = Path(r"F:\bci-project\data\emotiv\raw")

WINDOW_S = 1.0
OVERLAP_S = 0.5
MU_BAND = (8.0, 12.0)
BETA_BAND = (13.0, 30.0)
TASK_START, TASK_END = 2.5, 5.0


def preprocess(sig, sfreq):
    """Notch 50Hz + bandpass 8-30Hz, jak w src/preprocessing/unsupervised/filter.py."""
    b_notch, a_notch = iirnotch(50.0, Q=30.0, fs=sfreq)
    sig = filtfilt(b_notch, a_notch, sig, axis=1)
    b_bp, a_bp = butter(4, [8.0/(sfreq/2), 30.0/(sfreq/2)], btype="band")
    sig = filtfilt(b_bp, a_bp, sig, axis=1)
    return sig


def make_windows(n_samples, sfreq, window_s, overlap_s):
    win = int(round(window_s * sfreq))
    step = int(round((window_s - overlap_s) * sfreq))
    starts = list(range(0, n_samples - win + 1, step))
    return starts, win


def band_power(psd, freqs, band):
    mask = (freqs >= band[0]) & (freqs <= band[1])
    return psd[..., mask].mean(axis=-1)


def extract_features_and_labels(sig, sfreq):
    n_ch, n_samples = sig.shape
    starts, win = make_windows(n_samples, sfreq, WINDOW_S, OVERLAP_S)
    mu_feats, beta_feats, labels = [], [], []
    for s in starts:
        seg = sig[:, s:s + win]
        freqs, psd = welch(seg, fs=sfreq, nperseg=win, axis=1)
        mu_feats.append(band_power(psd, freqs, MU_BAND))
        beta_feats.append(band_power(psd, freqs, BETA_BAND))
        t_center = (s + win / 2) / sfreq
        labels.append(1 if TASK_START <= t_center < TASK_END else 0)
    return np.array(mu_feats), np.array(beta_feats), np.array(labels)


def holm_bonferroni(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    prev = 0.0
    for rank, idx in enumerate(order):
        corrected = (m - rank) * pvals[idx]
        corrected = max(corrected, prev)
        corrected = min(corrected, 1.0)
        adj[idx] = corrected
        prev = corrected
    return adj


def main():
    all_mu, all_beta, all_labels = [], [], []
    per_session = {}
    for sess in SESSIONS:
        path = DATA_DIR / f"{sess}.edf"
        # Zamiana Path na string by upewnić się, że funkcja wczytująca poprawnie przetworzy ścieżkę
        sig, sfreq = read_edf(str(path), EEG_CHANNELS)
        sig = preprocess(sig, sfreq)
        mu, beta, labels = extract_features_and_labels(sig, sfreq)
        per_session[sess] = (mu, beta, labels)
        all_mu.append(mu)
        all_beta.append(beta)
        all_labels.append(labels)
        print(f"{sess}: {len(labels)} okien, rest={np.sum(labels==0)}, task={np.sum(labels==1)}")

    all_mu = np.vstack(all_mu)
    all_beta = np.vstack(all_beta)
    all_labels = np.concatenate(all_labels)

    rows = []
    pvals = []
    for band_name, feats in [("Mu (8-12Hz)", all_mu), ("Beta (13-30Hz)", all_beta)]:
        for ci, ch in enumerate(EEG_CHANNELS):
            rest_vals = feats[all_labels == 0, ci]
            task_vals = feats[all_labels == 1, ci]
            F, p = f_oneway(rest_vals, task_vals)
            rows.append([band_name, ch, F, p, rest_vals.mean(), task_vals.mean()])
            pvals.append(p)

    pvals = np.array(pvals)
    adj_p = holm_bonferroni(pvals)

    print("\n=== ANOVA (f_oneway) per kanał x pasmo, rest vs task (zbiór pooled, n=%d) ===" % len(all_labels))
    print(f"{'Pasmo':15s} {'Kanał':6s} {'F':>10s} {'p':>12s} {'p_holm':>12s} {'rest mean':>14s} {'task mean':>14s}")
    for row, ap in zip(rows, adj_p):
        band_name, ch, F, p, rm, tm = row
        sig_mark = "*" if ap < 0.05 else ""
        print(f"{band_name:15s} {ch:6s} {F:10.3f} {p:12.4g} {ap:12.4g} {rm:14.6g} {tm:14.6g} {sig_mark}")

    # Rank channels by significance (lowest Holm-adjusted p, both bands combined -> take min)
    ch_best_p = {}
    for row, ap in zip(rows, adj_p):
        band_name, ch, F, p, rm, tm = row
        if ch not in ch_best_p or ap < ch_best_p[ch][0]:
            ch_best_p[ch] = (ap, band_name, F)

    print("\n=== Ranking kanałów (najlepszy wynik z dwóch pasm, po korekcie Holm-Bonferroni) ===")
    for ch, (ap, band_name, F) in sorted(ch_best_p.items(), key=lambda kv: kv[1][0]):
        print(f"{ch:6s}  best_p_holm={ap:10.4g}  pasmo={band_name:15s} F={F:8.3f}")


if __name__ == "__main__":
    main()