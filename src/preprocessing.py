"""
BCI Competition IV - Dataset 2b
Full Exploratory Analysis Script
=================================
Dataset:  9 subjects, 3 EEG channels (C3, Cz, C4), binary motor imagery
          Left hand (class 1) vs Right hand (class 2)
Format:   GDF files, 250 Hz sampling rate
Sessions: 5 sessions per subject (sessions 1-2: without feedback, 3-5: with feedback)

Usage
-----
1. Download Dataset 2b from:
   https://www.bbci.de/competition/iv/#dataset2b
   Place all .gdf files in a directory: ../data/
2. uv run preprocessing.py --data_dir ./data/ --out_dir ./results

Directory structure produced
-----------------------------
results/
  figures/          - .png figures
  exports/          - CSV tables for article
  logs/             - per-subject text logs

Figures produced
----------------
  01_dataset_overview.png          - epoch counts, class balance, session distribution
  02_raw_psd_per_subject.png       - Welch PSD overlaid per channel, both classes
  03_grand_avg_psd.png             - Grand-average PSD across subjects, mu + beta bands shaded
  04_erd_ers_timecourse.png        - Event-related desynchronization/synchronization (mu + beta)
  05_erd_ers_per_subject.png       - ERD/ERS time course, one panel per subject
  06_channel_correlation.png       - Inter-channel correlation matrices per class
  07_stationarity_summary.png      - ADF p-values heatmap (subjects x channels)
  08_autocorrelation.png           - ACF/PACF for each channel, both classes
  09_coherence.png                 - Inter-channel coherence (C3-C4) per class
  10_csp_patterns.png              - Top 4 CSP spatial patterns (as bar charts, 3-channel)
  11_csp_scatter.png               - Trial scatter in CSP feature space
  12_snr_per_subject.png           - Band-limited SNR per subject, per class
  13_trial_variability.png         - Within-subject trial-to-trial variance over sessions
  14_artifact_summary.png          - Rejected epochs overview
"""

import os
import sys
import argparse
import warnings
import logging
from pathlib import Path
from itertools import combinations

from paths import SRC_DIR, DEFAULT_DATA_DIR, DEFAULT_OUT_DIR

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import seaborn as sns
from scipy import signal, stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import mne
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("WARNING")

matplotlib.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.2,
})

FS = 250                  # sampling rate (Hz)
T_MIN, T_MAX = 0.5, 2.5   # imagery window relative to cue onset (s)
BASELINE = (-0.5, 0.0)    # baseline correction window (s)
MU_BAND = (8, 12)         # mu rhythm
BETA_BAND = (13, 30)      # beta rhythm
BANDS = {"mu": MU_BAND, "beta": BETA_BAND}
CHANNELS = ["C3", "Cz", "C4"]   # Dataset 2b channels
CLASS_IDS = [1, 2]               # 1=left hand, 2=right hand
CLASS_NAMES = {1: "Left hand", 2: "Right hand"}
STANDARD_EVENT_ID = {
    "left_hand": 1,
    "right_hand": 2
}
CLASS_COLORS = {1: "#2166ac", 2: "#d6604d"}
N_SUBJECTS = 9
N_SESSIONS = 5
REJECT_THRESH = 100e-6  

# Standard BCI Comp IV 2b event codes
EVENT_LEFT = 769    # left hand cue
EVENT_RIGHT = 770   # right hand cue


def setup_dirs(out_dir: Path):
    for sub in ["figures", "exports", "logs"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

def setup_logger(out_dir: Path) -> logging.Logger:
    logger = logging.getLogger("bci2b")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(out_dir / "logs" / "analysis.log")
    ch = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s", "%H:%M:%S")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger

def savefig(fig, path: Path, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def band_power(data, fs, band):
    """Mean bandpower via Welch, data shape (n_channels, n_samples)."""
    freqs, psd = signal.welch(data, fs=fs, nperseg=fs * 2, noverlap=fs)
    idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    return psd[:, idx].mean(axis=1)  # (n_channels,)

"""
══════════════════════════════════════════════════════════════════════════════
SECTION 1 · DATA LOADING
══════════════════════════════════════════════════════════════════════════════
"""
def find_gdf_files(data_dir: Path) -> dict:
    """
    Scan data_dir for Dataset 2b GDF files.
    Expected naming: B0{subject}0{session}T.gdf (training) / B0{subject}0{session}E.gdf (eval)
    Returns dict: {subject_id: [list of (session, path) tuples]}
    """
    files = {}
    for gdf in sorted(data_dir.glob("*.gdf")):
        stem = gdf.stem  # e.g. B0101T or B0101E
        if len(stem) >= 5 and stem[0] == "B" and stem[-1] == "T":
            try:
                subj = int(stem[1:3])
                sess = int(stem[3:5])
                files.setdefault(subj, []).append((sess, gdf))
            except ValueError:
                continue
    return files

def load_subject_epochs(subject_id: int, session_files: list, logger) -> tuple:
    """
    Load all sessions for one subject, extract left/right hand epochs.
    Returns (epochs_left, epochs_right, metadata_df)
    """
    all_epochs = {1: [], 2: []}
    meta_rows = []

    for sess_idx, (sess_num, fpath) in enumerate(sorted(session_files)):
        logger.info(f"  Subject {subject_id:02d} | session {sess_num} | {fpath.name}")
        try:
            raw = mne.io.read_raw_gdf(str(fpath), preload=True, verbose=False)
        except Exception as e:
            logger.warning(f"    Could not load {fpath.name}: {e}")
            continue

        ch_names = raw.ch_names
        # Map whatever names are in the file to C3/Cz/C4
        eeg_chs = [c for c in ch_names if "EEG" in c or c in CHANNELS]
        if len(eeg_chs) >= 3:
            rename_map = {eeg_chs[0]: "C3", eeg_chs[1]: "Cz", eeg_chs[2]: "C4"}
            raw.rename_channels(rename_map)
        raw.pick_channels(["C3", "Cz", "C4"], ordered=True)
        raw.set_channel_types({"C3": "eeg", "Cz": "eeg", "C4": "eeg"})
        raw.set_montage("standard_1020", on_missing="ignore")

        raw.filter(1.0, 40.0, fir_window="hamming", verbose=False)
        raw.notch_filter([50.0], verbose=False)   # EU power line

        events, event_id = mne.events_from_annotations(raw, verbose=False)

        # Map event codes robustly
        id_map = {}
        for code, name in event_id.items():
            # Handle both numeric strings and descriptive names
            code_int = None
            try:
                code_int = int(float(code))
            except (ValueError, TypeError):
                if "769" in str(name) or "left" in str(name).lower():
                    code_int = 769
                elif "770" in str(name) or "right" in str(name).lower():
                    code_int = 770
            if code_int == EVENT_LEFT:
                id_map["left_hand"] = event_id[code]
            elif code_int == EVENT_RIGHT:
                id_map["right_hand"] = event_id[code]
                

        # Fallback: use first two event types as left/right
        if len(id_map) < 2 and len(event_id) >= 2:
            keys = sorted(event_id.keys())
            id_map = {"left_hand": event_id[keys[0]], "right_hand": event_id[keys[1]]}
            logger.warning(f"    Fallback event mapping used for {fpath.name}")

        if not id_map:
            logger.warning(f"    No usable events in {fpath.name}, skipping")
            continue
    
        events_fixed = []
        for ev in events:
            if ev[2] == id_map.get("left_hand"):
                events_fixed.append([ev[0], ev[1], 1])
            elif ev[2] == id_map.get("right_hand"):
                events_fixed.append([ev[0], ev[1], 2])

        events_fixed = np.array(events_fixed)
        epochs = mne.Epochs(
            raw, events_fixed,
            event_id=STANDARD_EVENT_ID,
            tmin=-1.0,
            tmax=T_MAX,
            baseline=None,
            reject={"eeg": REJECT_THRESH},
            preload=True,
            verbose=False,
        )

        n_total = len(epochs.events)
        n_kept = len(epochs)
        n_rejected = n_total - n_kept

        for cls_name, cls_id in [("left_hand", 1), ("right_hand", 2)]:
            try:
                ep_cls = epochs[cls_name]
                all_epochs[cls_id].append(ep_cls)
                meta_rows.append({
                    "subject": subject_id,
                    "session": sess_num,
                    "class": cls_id,
                    "n_epochs": len(ep_cls),
                    "n_rejected": n_rejected,
                    "feedback": sess_num >= 3,
                })
            except KeyError:
                pass

    # Concatenate across sessions
    result = {}
    for cls_id in [1, 2]:
        if all_epochs[cls_id]:
            result[cls_id] = mne.concatenate_epochs(all_epochs[cls_id], verbose=False)
        else:
            result[cls_id] = None

    meta_df = pd.DataFrame(meta_rows)
    return result[1], result[2], meta_df


def load_all_subjects(data_dir: Path, logger) -> dict:
    """Load all subjects. Returns dict with per-subject data."""
    gdf_map = find_gdf_files(data_dir)
    if not gdf_map:
        logger.error(f"No GDF files found in {data_dir}. Check the path and file naming.")
        sys.exit(1)

    logger.info(f"Found {len(gdf_map)} subjects: {sorted(gdf_map.keys())}")

    subjects = {}
    all_meta = []

    for subj_id in sorted(gdf_map.keys()):
        logger.info(f"Loading subject {subj_id:02d}...")
        ep_left, ep_right, meta = load_subject_epochs(subj_id, gdf_map[subj_id], logger)
        subjects[subj_id] = {
            1: ep_left,
            2: ep_right
        }
        all_meta.append(meta)

    meta_df = pd.concat(all_meta, ignore_index=True)
    return subjects, meta_df

"""
══════════════════════════════════════════════════════════════════════════════
SECTION 2 · DATASET OVERVIEW  →  Figure 01
══════════════════════════════════════════════════════════════════════════════
"""
def fig_dataset_overview(meta_df: pd.DataFrame, out_dir: Path, logger):
    logger.info("Figure 01: Dataset overview")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # (a) Epoch counts per subject per class
    ax = axes[0]
    counts = meta_df.groupby(["subject", "class"])["n_epochs"].sum().unstack(fill_value=0)
    x = np.arange(len(counts))
    w = 0.35
    ax.bar(x - w/2, counts.get(1, 0), w, label="Left hand", color=CLASS_COLORS[1], alpha=0.85)
    ax.bar(x + w/2, counts.get(2, 0), w, label="Right hand", color=CLASS_COLORS[2], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([f"S{s}" for s in counts.index])
    ax.set_xlabel("Subject"); ax.set_ylabel("Epochs (accepted)")
    ax.set_title("(a) Accepted epochs per subject")
    ax.legend()

    # (b) Rejection rate per subject
    ax = axes[1]
    rej = meta_df.groupby("subject").agg(
        total=("n_epochs", "sum"), rejected=("n_rejected", "sum")
    )
    rej["pct"] = 100 * rej["rejected"] / (rej["total"] + rej["rejected"])
    ax.bar(np.arange(len(rej)), rej["pct"], color="#636363", alpha=0.8)
    ax.set_xticks(np.arange(len(rej)))
    ax.set_xticklabels([f"S{s}" for s in rej.index])
    ax.set_xlabel("Subject"); ax.set_ylabel("Rejected epochs (%)")
    ax.set_title("(b) Artifact rejection rate")
    ax.axhline(rej["pct"].mean(), color="crimson", ls="--", lw=1, label=f"Mean={rej['pct'].mean():.1f}%")
    ax.legend()

    # (c) Session breakdown (feedback vs no-feedback)
    ax = axes[2]
    sess_counts = meta_df.groupby(["session", "class"])["n_epochs"].sum().unstack(fill_value=0)
    x = np.arange(len(sess_counts))
    ax.bar(x - w/2, sess_counts.get(1, 0), w, color=CLASS_COLORS[1], alpha=0.85, label="Left")
    ax.bar(x + w/2, sess_counts.get(2, 0), w, color=CLASS_COLORS[2], alpha=0.85, label="Right")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Sess {s}\n{'(fb)' if s >= 3 else '(no fb)'}" for s in sess_counts.index])
    ax.set_xlabel("Session"); ax.set_ylabel("Epochs")
    ax.set_title("(c) Epochs per session")
    ax.legend()

    fig.suptitle("BCI Competition IV Dataset 2b — Dataset Overview", fontweight="bold")
    savefig(fig, out_dir / "figures" / "01_dataset_overview.png")

"""
══════════════════════════════════════════════════════════════════════════════
SECTION 3 · SIGNAL QUALITY CHECKS  →  Figure 14
══════════════════════════════════════════════════════════════════════════════
"""
def fig_artifact_summary(meta_df: pd.DataFrame, out_dir: Path, logger):
    logger.info("Figure 14: Artifact summary")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Rejection per session
    ax = axes[0]
    sess_rej = meta_df.groupby("session").agg(
        total=("n_epochs", "sum"), rejected=("n_rejected", "sum")
    )
    sess_rej["pct"] = 100 * sess_rej["rejected"] / (sess_rej["total"] + sess_rej["rejected"]).clip(lower=1)
    colors = ["#969696" if s < 3 else "#3182bd" for s in sess_rej.index]
    ax.bar(sess_rej.index, sess_rej["pct"], color=colors, alpha=0.85)
    ax.set_xlabel("Session"); ax.set_ylabel("Rejected epochs (%)")
    ax.set_title("Rejection rate by session")
    legend_els = [Line2D([0], [0], color="#969696", lw=6, label="No feedback"),
                  Line2D([0], [0], color="#3182bd", lw=6, label="Feedback")]
    ax.legend(handles=legend_els)

    # Heatmap: rejected epochs per subject x session
    ax = axes[1]
    pivot = meta_df.groupby(["subject", "session"])["n_rejected"].sum().unstack(fill_value=0)
    sns.heatmap(pivot, ax=ax, cmap="YlOrRd", annot=True, fmt="d",
                linewidths=0.4, cbar_kws={"label": "Rejected trials"},
                yticklabels=[f"S{i}" for i in pivot.index])
    ax.set_title("Rejected epochs heatmap (subject × session)")
    ax.set_xlabel("Session"); ax.set_ylabel("Subject")

    fig.suptitle("Artifact Rejection Summary", fontweight="bold")
    savefig(fig, out_dir / "figures" / "14_artifact_summary.png")

"""
══════════════════════════════════════════════════════════════════════════════
SECTION 4 · RAW PSD PER SUBJECT  →  Figure 02
══════════════════════════════════════════════════════════════════════════════
"""
def fig_raw_psd_per_subject(subjects: dict, out_dir: Path, logger):
    logger.info("Figure 02: Raw PSD per subject")

    subj_ids = sorted(subjects.keys())
    n = len(subj_ids)
    fig, axes = plt.subplots(n, 3, figsize=(12, 2.2 * n), sharex=True, sharey=False)
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, sid in enumerate(subj_ids):
        for col, ch in enumerate(CHANNELS):
            ax = axes[row, col]
            for cls_id in CLASS_IDS:
                ep = subjects[sid].get(CLASS_NAMES[cls_id].lower().replace(" ", "_"))
                ep = subjects[sid].get(cls_id) if ep is None else ep
                # Use dict keys 1/2
                ep = subjects[sid][cls_id] if cls_id in subjects[sid] else None
                if ep is None:
                    continue
                ch_idx = ep.ch_names.index(ch) if ch in ep.ch_names else 0
                data = ep.get_data()[:, ch_idx, :]   # (n_trials, n_times)
                data_concat = data.reshape(-1)
                freqs, psd = signal.welch(data_concat, fs=FS, nperseg=FS * 2)
                mask = freqs <= 40
                ax.semilogy(freqs[mask], psd[mask],
                            color=CLASS_COLORS[cls_id], alpha=0.85,
                            label=CLASS_NAMES[cls_id] if row == 0 else None)

            for band_name, band in BANDS.items():
                ax.axvspan(band[0], band[1], alpha=0.08,
                           color="green" if band_name == "mu" else "orange")

            if col == 0:
                ax.set_ylabel(f"S{sid}\nPSD (V²/Hz)")
            if row == 0:
                ax.set_title(ch)
            if row == n - 1:
                ax.set_xlabel("Frequency (Hz)")

    handles = [Line2D([0], [0], color=CLASS_COLORS[1], label="Left hand"),
               Line2D([0], [0], color=CLASS_COLORS[2], label="Right hand"),
               plt.Rectangle((0, 0), 1, 1, fc="green", alpha=0.15, label="Mu band"),
               plt.Rectangle((0, 0), 1, 1, fc="orange", alpha=0.15, label="Beta band")]
    fig.legend(handles=handles, loc="upper right", ncol=2)
    fig.suptitle("Welch PSD per Subject per Channel", fontweight="bold", y=1.01)
    savefig(fig, out_dir / "figures" / "02_raw_psd_per_subject.png")

"""
══════════════════════════════════════════════════════════════════════════════
SECTION 5 · GRAND-AVERAGE PSD  →  Figure 03
══════════════════════════════════════════════════════════════════════════════
"""
def fig_grand_avg_psd(subjects: dict, out_dir: Path, logger):
    logger.info("Figure 03: Grand-average PSD")

    freqs_ref = None
    psd_acc = {cls_id: {ch: [] for ch in CHANNELS} for cls_id in CLASS_IDS}

    for sid in subjects:
        for cls_id in CLASS_IDS:
            ep = subjects[sid][cls_id]
            if ep is None:
                continue
            for ch in CHANNELS:
                if ch not in ep.ch_names:
                    continue
                ch_idx = ep.ch_names.index(ch)
                data = ep.get_data()[:, ch_idx, :].reshape(-1)
                freqs, psd = signal.welch(data, fs=FS, nperseg=FS * 2)
                freqs_ref = freqs
                psd_acc[cls_id][ch].append(psd)

    if freqs_ref is None:
        logger.warning("  No data for grand-average PSD")
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)

    for col, ch in enumerate(CHANNELS):
        ax = axes[col]
        for cls_id in CLASS_IDS:
            psds = np.array(psd_acc[cls_id][ch])  # (n_subjects, n_freqs)
            if psds.ndim < 2 or len(psds) == 0:
                continue
            mean = psds.mean(axis=0)
            sem = psds.std(axis=0) / np.sqrt(len(psds))
            mask = freqs_ref <= 40
            ax.semilogy(freqs_ref[mask], mean[mask],
                        color=CLASS_COLORS[cls_id], label=CLASS_NAMES[cls_id])
            ax.fill_between(freqs_ref[mask],
                            mean[mask] - sem[mask],
                            mean[mask] + sem[mask],
                            alpha=0.2, color=CLASS_COLORS[cls_id])

        for band_name, (f0, f1) in BANDS.items():
            mask_b = (freqs_ref >= f0) & (freqs_ref <= f1) & (freqs_ref <= 40)
            ax.axvspan(f0, f1, alpha=0.10,
                       color="green" if band_name == "mu" else "orange",
                       label=f"{band_name.capitalize()} band")

        ax.set_title(f"Channel {ch}")
        ax.set_xlabel("Frequency (Hz)")
        if col == 0:
            ax.set_ylabel("PSD (V²/Hz)")
        ax.legend(fontsize=7)

    fig.suptitle("Grand-Average Welch PSD (mean ± SEM across subjects)", fontweight="bold")
    savefig(fig, out_dir / "figures" / "03_grand_avg_psd.png")

"""
══════════════════════════════════════════════════════════════════════════════
SECTION 6 · ERD / ERS  →  Figures 04 & 05
══════════════════════════════════════════════════════════════════════════════
"""
def compute_erd_ers(epochs, band, fs=FS, baseline=BASELINE):
    """
    Compute ERD/ERS time course relative to baseline.
    Returns (times, erd_matrix) where erd_matrix shape = (n_channels, n_times).
    ERD/ERS formula: (A - R) / R * 100  where R = baseline power, A = time-varying power.
    """
    data = epochs.get_data()   # (n_trials, n_ch, n_times)
    times = epochs.times

    # Band-pass filter
    b, a = signal.butter(4, band, btype="band", fs=fs)
    data_filt = signal.filtfilt(b, a, data, axis=-1)

    # Instantaneous power (squared envelope via Hilbert)
    analytic = signal.hilbert(data_filt, axis=-1)
    power = np.abs(analytic) ** 2   # (n_trials, n_ch, n_times)

    # Baseline indices
    bl_idx = np.where((times >= baseline[0]) & (times <= baseline[1]))[0]
    ref_power = power[:, :, bl_idx].mean(axis=(0, 2))  # (n_ch,)

    # ERD/ERS: average across trials, then normalize
    mean_power = power.mean(axis=0)  # (n_ch, n_times)
    erd = (mean_power - ref_power[:, None]) / ref_power[:, None] * 100

    return times, erd


def fig_erd_ers_grand(subjects: dict, out_dir: Path, logger):
    logger.info("Figure 04: Grand-average ERD/ERS")

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey="row", sharex=True)
    times_ref = None

    for row, (band_name, band) in enumerate(BANDS.items()):
        erd_acc = {cls_id: {ch: [] for ch in CHANNELS} for cls_id in CLASS_IDS}

        for sid in subjects:
            for cls_id in CLASS_IDS:
                ep = subjects[sid][cls_id]
                if ep is None or len(ep) < 5:
                    continue
                times, erd = compute_erd_ers(ep, band)
                times_ref = times
                for c_idx, ch in enumerate(CHANNELS):
                    if c_idx < erd.shape[0]:
                        erd_acc[cls_id][ch].append(erd[c_idx])

        for col, ch in enumerate(CHANNELS):
            ax = axes[row, col]
            for cls_id in CLASS_IDS:
                erds = np.array(erd_acc[cls_id][ch])
                if len(erds) == 0:
                    continue
                mean = erds.mean(axis=0)
                sem = erds.std(axis=0) / np.sqrt(len(erds))
                ax.plot(times_ref, mean, color=CLASS_COLORS[cls_id],
                        label=CLASS_NAMES[cls_id])
                ax.fill_between(times_ref, mean - sem, mean + sem,
                                alpha=0.2, color=CLASS_COLORS[cls_id])

            ax.axhline(0, color="black", lw=0.7, ls="--")
            ax.axvline(0, color="gray", lw=0.7, ls=":")
            ax.axvspan(T_MIN, T_MAX, alpha=0.07, color="blue", label="Imagery" if col == 0 else None)
            ax.set_title(f"{ch} — {band_name.capitalize()} ({band[0]}–{band[1]} Hz)")
            if col == 0:
                ax.set_ylabel("ERD/ERS (%)\n← ERD  |  ERS →")
            if row == 1:
                ax.set_xlabel("Time (s) relative to cue")
            if col == 0 and row == 0:
                ax.legend(fontsize=7)

    fig.suptitle("Grand-Average ERD/ERS — Motor Imagery Window", fontweight="bold")
    savefig(fig, out_dir / "figures" / "04_erd_ers_timecourse.png")


def fig_erd_ers_per_subject(subjects: dict, out_dir: Path, logger):
    logger.info("Figure 05: ERD/ERS per subject")

    subj_ids = sorted(subjects.keys())
    n = len(subj_ids)
    # One row per subject, columns = channels, 2 figures (mu / beta)

    for band_name, band in BANDS.items():
        fig, axes = plt.subplots(n, 3, figsize=(12, 2.0 * n),
                                 sharex=True, sharey=False)
        if n == 1:
            axes = axes[np.newaxis, :]

        for row, sid in enumerate(subj_ids):
            for col, ch in enumerate(CHANNELS):
                ax = axes[row, col]
                for cls_id in CLASS_IDS:
                    ep = subjects[sid][cls_id]
                    if ep is None or len(ep) < 5:
                        continue
                    times, erd = compute_erd_ers(ep, band)
                    c_idx = CHANNELS.index(ch) if ch in CHANNELS else col
                    if c_idx < erd.shape[0]:
                        ax.plot(times, erd[c_idx], color=CLASS_COLORS[cls_id],
                                lw=1.0, label=CLASS_NAMES[cls_id])
                ax.axhline(0, color="black", lw=0.5, ls="--")
                ax.axvline(0, color="gray", lw=0.5, ls=":")
                ax.axvspan(T_MIN, T_MAX, alpha=0.07, color="blue")
                if col == 0:
                    ax.set_ylabel(f"S{sid}\nERD/ERS (%)")
                if row == 0:
                    ax.set_title(ch)
                if row == n - 1:
                    ax.set_xlabel("Time (s)")

        handles = [Line2D([0], [0], color=CLASS_COLORS[c], label=CLASS_NAMES[c]) for c in CLASS_IDS]
        fig.legend(handles=handles, loc="upper right")
        fig.suptitle(f"ERD/ERS per Subject — {band_name.capitalize()} band "
                     f"({band[0]}–{band[1]} Hz)", fontweight="bold", y=1.005)
        fname = f"05_erd_ers_per_subject_{band_name}.png"
        savefig(fig, out_dir / "figures" / fname)

"""
══════════════════════════════════════════════════════════════════════════════
SECTION 7 · CHANNEL CORRELATION  →  Figure 06
══════════════════════════════════════════════════════════════════════════════
"""
def fig_channel_correlation(subjects: dict, out_dir: Path, logger):
    logger.info("Figure 06: Channel correlation matrices")

    corr_acc = {cls_id: [] for cls_id in CLASS_IDS}

    for sid in subjects:
        for cls_id in CLASS_IDS:
            ep = subjects[sid][cls_id]
            if ep is None:
                continue
            data = ep.get_data()   # (n_trials, n_ch, n_times)
            for trial in data:
                corr_acc[cls_id].append(np.corrcoef(trial))  # (3, 3)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

    corr_cls = {}
    for cls_id in CLASS_IDS:
        corr_cls[cls_id] = np.stack(corr_acc[cls_id]).mean(axis=0)

    # Difference matrix
    diff = corr_cls[1] - corr_cls[2]

    for ax, (title, mat) in zip(axes, [
        (CLASS_NAMES[1], corr_cls[1]),
        (CLASS_NAMES[2], corr_cls[2]),
        ("Difference (Left − Right)", diff),
    ]):
        vmax = 1.0 if title != "Difference (Left − Right)" else 0.2
        vmin = -1.0 if title != "Difference (Left − Right)" else -0.2
        cmap = "RdBu_r" if "Diff" in title else "coolwarm"
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(3)); ax.set_xticklabels(CHANNELS)
        ax.set_yticks(range(3)); ax.set_yticklabels(CHANNELS)
        ax.set_title(title)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color="black")
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Mean Inter-Channel Correlation (grand average across trials & subjects)",
                 fontweight="bold")
    savefig(fig, out_dir / "figures" / "06_channel_correlation.png")

"""
══════════════════════════════════════════════════════════════════════════════
SECTION 8 · STATIONARITY  →  Figure 07
══════════════════════════════════════════════════════════════════════════════
"""
def fig_stationarity(subjects: dict, out_dir: Path, logger):
    logger.info("Figure 07: Stationarity (ADF) analysis")

    # ADF test per subject, channel, class — on mean epoch
    results = []
    for sid in sorted(subjects.keys()):
        for cls_id in CLASS_IDS:
            ep = subjects[sid][cls_id]
            if ep is None:
                continue
            data = ep.get_data()   # (n_trials, n_ch, n_times)
            for c_idx, ch in enumerate(CHANNELS):
                # Test each trial individually, report mean p-value
                p_vals = []
                for trial in data[:, c_idx, :]:
                    try:
                        adf_res = adfuller(trial, autolag="AIC")
                        p_vals.append(adf_res[1])
                    except Exception:
                        pass
                if p_vals:
                    results.append({
                        "subject": sid,
                        "channel": ch,
                        "class": CLASS_NAMES[cls_id],
                        "adf_p_mean": np.mean(p_vals),
                        "adf_p_median": np.median(p_vals),
                        "pct_stationary": 100 * np.mean(np.array(p_vals) < 0.05),
                    })

    res_df = pd.DataFrame(results)
    if res_df.empty:
        logger.warning("  No ADF results computed")
        return

    # Export
    res_df.to_csv(out_dir / "exports" / "stationarity_adf.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, cls_name in zip(axes, [CLASS_NAMES[1], CLASS_NAMES[2]]):
        sub_df = res_df[res_df["class"] == cls_name]
        pivot = sub_df.pivot(index="subject", columns="channel", values="pct_stationary")
        sns.heatmap(pivot, ax=ax, cmap="RdYlGn", vmin=0, vmax=100,
                    annot=True, fmt=".0f", linewidths=0.4,
                    cbar_kws={"label": "Stationary epochs (%)"},
                    yticklabels=[f"S{i}" for i in pivot.index])
        ax.set_title(f"{cls_name} — ADF p < 0.05")
        ax.set_xlabel("Channel"); ax.set_ylabel("Subject")

    fig.suptitle("Epoch-Level Stationarity (ADF Test)\n"
                 "Values = % of epochs classified as stationary (p < 0.05)",
                 fontweight="bold")
    savefig(fig, out_dir / "figures" / "07_stationarity_summary.png")
    logger.info(f"  Note: EEG epochs are expected to be largely stationary within short windows.")

"""
══════════════════════════════════════════════════════════════════════════════
SECTION 9 · AUTOCORRELATION  →  Figure 08
══════════════════════════════════════════════════════════════════════════════
"""
def fig_autocorrelation(subjects: dict, out_dir: Path, logger):
    """
    ACF/PACF for each channel and class, averaged across subjects.
    Reveals dominant oscillation periods (mu ~25 ms lag, beta ~50 ms lag).
    NOT used for forecasting — used for oscillation characterization.
    """
    logger.info("Figure 08: Autocorrelation / PACF")

    import statsmodels.api as sm

    n_lags = 60   # 60 samples = 240 ms at 250 Hz
    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    # rows = channels, cols = [ACF left, ACF right, ACF difference]

    for row, ch in enumerate(CHANNELS):
        acf_by_cls = {}
        for cls_id in CLASS_IDS:
            acfs = []
            for sid in subjects:
                ep = subjects[sid][cls_id]
                if ep is None:
                    continue
                c_idx = CHANNELS.index(ch)
                data = ep.get_data()[:, c_idx, :]  # (n_trials, n_times)
                # Compute ACF for each trial then average
                for trial in data[:20]:  # limit to 20 trials for speed
                    try:
                        acf_vals = sm.tsa.acf(trial, nlags=n_lags, fft=True)
                        acfs.append(acf_vals)
                    except Exception:
                        pass
            acf_by_cls[cls_id] = np.array(acfs).mean(axis=0) if acfs else np.zeros(n_lags + 1)

        lags = np.arange(n_lags + 1) / FS * 1000  # convert to ms

        for col, (title, data) in enumerate([
            (CLASS_NAMES[1], acf_by_cls.get(1, np.zeros(n_lags + 1))),
            (CLASS_NAMES[2], acf_by_cls.get(2, np.zeros(n_lags + 1))),
            ("Difference", acf_by_cls.get(1, np.zeros(n_lags + 1)) -
                           acf_by_cls.get(2, np.zeros(n_lags + 1))),
        ]):
            ax = axes[row, col]
            color = CLASS_COLORS.get(1 if col == 0 else 2, "#555555")
            ax.stem(lags[1:], data[1:], linefmt=color, markerfmt=" ",
                    basefmt="k-")
            ax.axhline(0, color="black", lw=0.5)
            # Significance bounds (±1.96/sqrt(N))
            n_approx = 500
            bound = 1.96 / np.sqrt(n_approx)
            ax.axhline(bound, color="red", ls="--", lw=0.8, alpha=0.7)
            ax.axhline(-bound, color="red", ls="--", lw=0.8, alpha=0.7)
            ax.set_title(f"{ch} — {title}")
            ax.set_xlabel("Lag (ms)")
            ax.set_ylabel("ACF")
            ax.set_ylim(-0.5, 1.0)

    fig.suptitle("Mean Autocorrelation Function (ACF) per Channel and Class\n"
                 "Peaks at ~100 ms → mu rhythm; ~50 ms → beta rhythm",
                 fontweight="bold")
    savefig(fig, out_dir / "figures" / "08_autocorrelation.png")

"""
══════════════════════════════════════════════════════════════════════════════
SECTION 10 · COHERENCE  →  Figure 09
══════════════════════════════════════════════════════════════════════════════
"""
def fig_coherence(subjects: dict, out_dir: Path, logger):
    logger.info("Figure 09: Inter-channel coherence")

    ch_pairs = list(combinations(range(len(CHANNELS)), 2))  # (0,1),(0,2),(1,2)
    pair_names = [f"{CHANNELS[a]}–{CHANNELS[b]}" for a, b in ch_pairs]

    fig, axes = plt.subplots(1, len(ch_pairs), figsize=(13, 4), sharey=True)

    for ax_idx, ((ca, cb), pname) in enumerate(zip(ch_pairs, pair_names)):
        ax = axes[ax_idx]
        for cls_id in CLASS_IDS:
            cohs = []
            for sid in subjects:
                ep = subjects[sid][cls_id]
                if ep is None:
                    continue
                data = ep.get_data()  # (n_trials, n_ch, n_times)
                for trial in data:
                    f, cxy = signal.coherence(trial[ca], trial[cb], fs=FS,
                                              nperseg=FS // 2)
                    cohs.append(cxy)

            if cohs:
                cohs = np.array(cohs)
                mean = cohs.mean(axis=0)
                sem = cohs.std(axis=0) / np.sqrt(len(cohs))
                mask = f <= 40
                ax.plot(f[mask], mean[mask], color=CLASS_COLORS[cls_id],
                        label=CLASS_NAMES[cls_id])
                ax.fill_between(f[mask], mean[mask] - sem[mask],
                                mean[mask] + sem[mask],
                                alpha=0.2, color=CLASS_COLORS[cls_id])

        for band_name, band in BANDS.items():
            ax.axvspan(band[0], band[1], alpha=0.1,
                       color="green" if band_name == "mu" else "orange")
        ax.set_title(pname)
        ax.set_xlabel("Frequency (Hz)")
        if ax_idx == 0:
            ax.set_ylabel("Coherence")
        ax.legend(fontsize=7)
        ax.set_ylim(0, 1)

    fig.suptitle("Inter-Channel Coherence (grand average across subjects)",
                 fontweight="bold")
    savefig(fig, out_dir / "figures" / "09_coherence.png")

"""
══════════════════════════════════════════════════════════════════════════════
SECTION 11 · CSP  →  Figures 10 & 11
══════════════════════════════════════════════════════════════════════════════
"""
def fig_csp(subjects: dict, out_dir: Path, logger):
    logger.info("Figures 10–11: CSP patterns and feature scatter")

    # Grand-level CSP across all subjects
    all_data, all_labels = [], []
    for sid in sorted(subjects.keys()):
        for cls_id in CLASS_IDS:
            ep = subjects[sid][cls_id]
            if ep is None:
                continue
            # Use imagery window only
            ep_crop = ep.copy().crop(T_MIN, T_MAX)
            data = ep_crop.get_data()
            all_data.append(data)
            all_labels.extend([cls_id] * len(data))

    if not all_data:
        logger.warning("  No data for CSP analysis")
        return

    X = np.concatenate(all_data, axis=0)  # (n_total, 3, n_times)
    y = np.array(all_labels)
    y_bin = (y == 2).astype(int)  # binary: 0=left, 1=right

    csp = CSP(n_components=3, reg=None, log=True, norm_trace=False)
    try:
        csp.fit(X, y_bin)
    except Exception as e:
        logger.warning(f"  CSP fit failed: {e}")
        return

    # Figure 10: Spatial patterns (bar charts for 3 channels)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    for i in range(3):
        ax = axes[i]
        pattern = csp.patterns_[i] if hasattr(csp, "patterns_") else csp.filters_[i]
        colors = [CLASS_COLORS[1] if v > 0 else CLASS_COLORS[2] for v in pattern]
        ax.bar(CHANNELS, pattern, color=colors, alpha=0.85)
        ax.axhline(0, color="black", lw=0.7)
        ax.set_title(f"CSP component {i+1}")
        ax.set_ylabel("Weight")

    fig.suptitle("CSP Spatial Patterns (C3, Cz, C4)\n"
                 "Blue = positive weight, Red = negative weight",
                 fontweight="bold")
    savefig(fig, out_dir / "figures" / "10_csp_patterns.png")

    # Figure 11: Feature scatter in CSP space
    X_csp = csp.transform(X)  # (n_trials, n_components)
    fig, ax = plt.subplots(figsize=(6, 5))
    for cls_id, label in CLASS_NAMES.items():
        idx = y == cls_id
        ax.scatter(X_csp[idx, 0], X_csp[idx, 1],
                   c=CLASS_COLORS[cls_id], label=label, alpha=0.4, s=15)
    ax.set_xlabel("CSP component 1 (log-variance)")
    ax.set_ylabel("CSP component 2 (log-variance)")
    ax.set_title("Trial distribution in CSP feature space")
    ax.legend()
    savefig(fig, out_dir / "figures" / "11_csp_scatter.png")

    # Cross-validated LDA accuracy with CSP
    pipeline = Pipeline([("csp", CSP(n_components=3, reg=None, log=True)),
                         ("lda", LinearDiscriminantAnalysis())])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    try:
        scores = cross_val_score(pipeline, X, y_bin, cv=skf, scoring="accuracy")
        logger.info(f"  CSP+LDA cross-val accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    except Exception as e:
        logger.warning(f"  CSP+LDA cross-val failed: {e}")
        scores = np.array([])

    return scores

"""
══════════════════════════════════════════════════════════════════════════════
SECTION 12 · SNR  →  Figure 12
══════════════════════════════════════════════════════════════════════════════
"""
def fig_snr(subjects: dict, out_dir: Path, logger):
    logger.info("Figure 12: SNR per subject")

    records = []
    for sid in sorted(subjects.keys()):
        for cls_id in CLASS_IDS:
            ep = subjects[sid][cls_id]
            if ep is None:
                continue
            for c_idx, ch in enumerate(CHANNELS):
                data = ep.get_data()[:, c_idx, :]  # (n_trials, n_times)
                for band_name, band in BANDS.items():
                    bp = band_power(data.mean(axis=0)[None, :], FS, band)[0]
                    # Noise = power outside the band
                    noise_mask = np.ones(FS * 2 // 2 + 1, dtype=bool)
                    freqs = np.fft.rfftfreq(FS * 2, 1 / FS)
                    in_band = (freqs >= band[0]) & (freqs <= band[1])
                    freqs2, psd = signal.welch(data.mean(axis=0), fs=FS, nperseg=FS * 2)
                    noise_psd = psd[~((freqs2 >= band[0]) & (freqs2 <= band[1]) |
                                      (freqs2 > 35))].mean()
                    snr = 10 * np.log10(bp / (noise_psd + 1e-30))
                    records.append({
                        "subject": sid,
                        "class": CLASS_NAMES[cls_id],
                        "channel": ch,
                        "band": band_name,
                        "snr_db": snr,
                    })

    snr_df = pd.DataFrame(records)
    snr_df.to_csv(out_dir / "exports" / "snr_per_subject.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, band_name in zip(axes, BANDS.keys()):
        sub = snr_df[snr_df["band"] == band_name]
        pivot = sub.groupby(["subject", "class"])["snr_db"].mean().unstack()
        x = np.arange(len(pivot))
        w = 0.35
        for i, cls_name in enumerate(CLASS_NAMES.values()):
            if cls_name in pivot.columns:
                ax.bar(x + (i - 0.5) * w, pivot[cls_name], w,
                       color=list(CLASS_COLORS.values())[i], alpha=0.85,
                       label=cls_name)
        ax.set_xticks(x)
        ax.set_xticklabels([f"S{s}" for s in pivot.index])
        ax.set_xlabel("Subject")
        ax.set_ylabel("SNR (dB)")
        ax.set_title(f"{band_name.capitalize()} band SNR")
        ax.legend()
        ax.axhline(snr_df[snr_df["band"] == band_name]["snr_db"].mean(),
                   color="crimson", ls="--", lw=1, label="Grand mean")

    fig.suptitle("Band-Limited SNR per Subject", fontweight="bold")
    savefig(fig, out_dir / "figures" / "12_snr_per_subject.png")

"""
══════════════════════════════════════════════════════════════════════════════
SECTION 13 · TRIAL VARIABILITY  →  Figure 13
══════════════════════════════════════════════════════════════════════════════
"""
def fig_trial_variability(subjects: dict, meta_df: pd.DataFrame, out_dir: Path, logger):
    logger.info("Figure 13: Trial-to-trial variability")

    records = []
    for sid in sorted(subjects.keys()):
        for cls_id in CLASS_IDS:
            ep = subjects[sid][cls_id]
            if ep is None:
                continue
            data = ep.get_data()  # (n_trials, n_ch, n_times)
            for c_idx, ch in enumerate(CHANNELS):
                trial_vars = data[:, c_idx, :].var(axis=1)  # per-trial variance
                records.append({
                    "subject": sid,
                    "class": CLASS_NAMES[cls_id],
                    "channel": ch,
                    "mean_var": trial_vars.mean(),
                    "std_var": trial_vars.std(),
                    "cv": trial_vars.std() / (trial_vars.mean() + 1e-30),
                })

    var_df = pd.DataFrame(records)
    var_df.to_csv(out_dir / "exports" / "trial_variability.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    for ax, ch in zip(axes, CHANNELS):
        sub = var_df[var_df["channel"] == ch]
        for cls_id, cls_name in CLASS_NAMES.items():
            sv = sub[sub["class"] == cls_name]
            ax.errorbar(sv["subject"], sv["mean_var"] * 1e12,
                        yerr=sv["std_var"] * 1e12,
                        fmt="o-", color=CLASS_COLORS[cls_id],
                        label=cls_name, capsize=3)
        ax.set_title(f"Channel {ch}")
        ax.set_xlabel("Subject")
        ax.set_ylabel("Trial variance (µV²)")
        ax.legend(fontsize=7)

    fig.suptitle("Trial-to-Trial EEG Variance per Subject", fontweight="bold")
    savefig(fig, out_dir / "figures" / "13_trial_variability.png")

"""
══════════════════════════════════════════════════════════════════════════════
SECTION 14 · SUMMARY STATISTICS TABLE  →  exports/
══════════════════════════════════════════════════════════════════════════════
"""
def export_summary_stats(subjects: dict, meta_df: pd.DataFrame, out_dir: Path, logger):
    logger.info("Exporting summary statistics tables")

    rows = []
    for sid in sorted(subjects.keys()):
        for cls_id in CLASS_IDS:
            ep = subjects[sid][cls_id]
            if ep is None:
                continue
            data = ep.get_data()
            for c_idx, ch in enumerate(CHANNELS):
                ch_data = data[:, c_idx, :]
                for band_name, band in BANDS.items():
                    bp = band_power(ch_data, FS, band)   # per-trial bandpower
                    rows.append({
                        "subject": sid,
                        "class": CLASS_NAMES[cls_id],
                        "channel": ch,
                        "band": band_name,
                        "n_trials": len(ch_data),
                        "bp_mean_uV2": bp.mean() * 1e12,
                        "bp_std_uV2": bp.std() * 1e12,
                        "amp_mean_uV": np.abs(ch_data).mean() * 1e6,
                        "amp_std_uV": ch_data.std(axis=1).mean() * 1e6,
                    })

    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(out_dir / "exports" / "summary_stats.csv", index=False)

    # Grand summary (mean ± std across subjects)
    grand = stats_df.groupby(["class", "channel", "band"]).agg(
        n_subjects=("subject", "nunique"),
        bp_mean=("bp_mean_uV2", "mean"),
        bp_sem=("bp_mean_uV2", lambda x: x.std() / np.sqrt(len(x))),
        amp_mean=("amp_mean_uV", "mean"),
        amp_sem=("amp_mean_uV", lambda x: x.std() / np.sqrt(len(x))),
    ).round(4)
    grand.to_csv(out_dir / "exports" / "grand_summary.csv")

    logger.info(f"  Saved summary_stats.csv ({len(stats_df)} rows)")
    logger.info(f"  Saved grand_summary.csv ({len(grand)} rows)")


def main():
    parser = argparse.ArgumentParser(
        description="BCI Competition IV Dataset 2b — Full EEG Exploration"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Directory containing .gdf files (default: project_root/data)"
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help="Output directory (default: project_root/results)"
    )
    parser.add_argument("--skip_autocorr", action="store_true",
                        help="Skip ACF/PACF (slow for large datasets)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    setup_dirs(out_dir)
    logger = setup_logger(out_dir)

    logger.info("=" * 65)
    logger.info("BCI Competition IV Dataset 2b — Full Exploration")
    logger.info("=" * 65)
    logger.info(f"Data directory : {data_dir}")
    logger.info(f"Output directory: {out_dir}")

    # ── 1. Load data ──────────────────────────────────────────────────────
    logger.info("\n[1/9] Loading all subjects...")
    subjects, meta_df = load_all_subjects(data_dir, logger)
    meta_df.to_csv(out_dir / "exports" / "epoch_metadata.csv", index=False)

    loaded = {sid: subjects[sid] for sid in subjects
              if subjects[sid][1] is not None or subjects[sid][2] is not None}
    logger.info(f"  Successfully loaded {len(loaded)} subjects")

    if not loaded:
        logger.error("No valid data loaded. Exiting.")
        sys.exit(1)

    # ── 2. Overview ───────────────────────────────────────────────────────
    logger.info("\n[2/9] Dataset overview figure...")
    fig_dataset_overview(meta_df, out_dir, logger)
    fig_artifact_summary(meta_df, out_dir, logger)

    # ── 3. PSD ────────────────────────────────────────────────────────────
    logger.info("\n[3/9] Power spectral density figures...")
    fig_raw_psd_per_subject(subjects, out_dir, logger)
    fig_grand_avg_psd(subjects, out_dir, logger)

    # ── 4. ERD/ERS ────────────────────────────────────────────────────────
    logger.info("\n[4/9] ERD/ERS figures...")
    fig_erd_ers_grand(subjects, out_dir, logger)
    fig_erd_ers_per_subject(subjects, out_dir, logger)

    # ── 5. Channel correlation ────────────────────────────────────────────
    logger.info("\n[5/9] Channel correlation figure...")
    fig_channel_correlation(subjects, out_dir, logger)

    # ── 6. Stationarity ───────────────────────────────────────────────────
    logger.info("\n[6/9] Stationarity analysis...")
    fig_stationarity(subjects, out_dir, logger)

    # ── 7. Autocorrelation ────────────────────────────────────────────────
    if not args.skip_autocorr:
        logger.info("\n[7/9] Autocorrelation / PACF...")
        try:
            import statsmodels  # noqa
            fig_autocorrelation(subjects, out_dir, logger)
        except ImportError:
            logger.warning("  statsmodels not installed — skipping ACF. pip install statsmodels")
    else:
        logger.info("\n[7/9] Autocorrelation skipped (--skip_autocorr flag)")

    # ── 8. Coherence ──────────────────────────────────────────────────────
    logger.info("\n[8/9] Inter-channel coherence...")
    fig_coherence(subjects, out_dir, logger)

    # ── 9. CSP + SNR + variability + exports ─────────────────────────────
    logger.info("\n[9/9] CSP, SNR, variability, summary stats...")
    csp_scores = fig_csp(subjects, out_dir, logger)
    fig_snr(subjects, out_dir, logger)
    fig_trial_variability(subjects, meta_df, out_dir, logger)
    export_summary_stats(subjects, meta_df, out_dir, logger)

    # ── Final summary ─────────────────────────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("DONE. Summary of outputs:")
    logger.info(f"  Figures : {out_dir / 'figures'}")
    logger.info(f"  Exports : {out_dir / 'exports'}")
    logger.info(f"  Log     : {out_dir / 'logs' / 'analysis.log'}")

    if csp_scores is not None and len(csp_scores) > 0:
        logger.info(f"\n  CSP+LDA cross-val (grand): "
                    f"{csp_scores.mean():.3f} ± {csp_scores.std():.3f}")

    logger.info("\nKey interpretive notes:")
    logger.info("  · ERD (negative values) in mu/beta = motor imagery activation")
    logger.info("  · C3 ERD → right hand imagery; C4 ERD → left hand imagery")
    logger.info("  · Stationarity: EEG epochs SHOULD be mostly stationary (<2.5s)")
    logger.info("  · Autocorrelation peaks encode oscillation period, NOT forecast structure")
    logger.info("  · CSP maximises variance ratio between classes — ideal for BCI paradigms")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()