"""
Trening klasycznych modeli ML (LDA, SVM, Random Forest, LightGBM) na zbiorze BCI-IV 2b.
Uruchomienie: python train_ml_baselines.py --data-dir ./data
"""

import argparse
import re
import warnings
from collections import defaultdict
from pathlib import Path

import mne
import numpy as np
import lightgbm as lgb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from mne.decoding import CSP

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("WARNING")

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS & DATA LOADING (Identyczne jak w Deep Learning)
# ──────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = ["Left hand", "Right hand"]
SFREQ_TARGET = 250
TMIN, TMAX = 0.5, 2.5
EEG_CHANNELS = ["EEG:C3", "EEG:Cz", "EEG:C4"]


def discover_files(data_dir: Path):
    pattern = re.compile(r"^B(\d{2})\d{2}[TE]\.gdf$", re.IGNORECASE)
    train_files: dict = defaultdict(list)
    eval_files: dict = defaultdict(list)

    for f in sorted(data_dir.glob("*.gdf")):
        m = pattern.match(f.name)
        if not m: continue
        subj = m.group(1)
        if f.stem[-1].upper() == "T":
            train_files[subj].append(f)
        else:
            eval_files[subj].append(f)
    return dict(train_files), dict(eval_files)


def load_gdf(path: Path):
    try:
        raw = mne.io.read_raw_gdf(str(path), preload=True, eog=[])
    except Exception as e:
        return None, None

    raw.filter(4.0, 40.0, fir_design="firwin", skip_by_annotation="edge")
    available = raw.ch_names
    picks = [ch for ch in EEG_CHANNELS if ch in available]
    if len(picks) < 3: picks = available[:3]
    raw.pick_channels(picks)

    if int(raw.info["sfreq"]) != SFREQ_TARGET: raw.resample(SFREQ_TARGET)

    try:
        events, event_id_found = mne.events_from_annotations(raw)
    except Exception:
        return None, None

    MI_KEYS = {"769", "770"}
    MI_VALS = {1, 2, 769, 770}
    keep = {k: v for k, v in event_id_found.items() if k in MI_KEYS}
    if len(keep) < 2: keep = {k: v for k, v in event_id_found.items() if v in MI_VALS}
    if len(keep) < 2: return None, None

    epochs = mne.Epochs(raw, events, event_id=keep, tmin=TMIN, tmax=TMAX,
                        baseline=None, preload=True, reject=None, verbose=False)
    X = epochs.get_data().astype(np.float64)  # CSP woli float64
    y_raw = epochs.events[:, -1]

    unique = sorted(np.unique(y_raw))
    label_map = {old: new for new, old in enumerate(unique)}
    y = np.array([label_map[l] for l in y_raw], dtype=np.int64)
    return X, y


def load_subject(train_paths, eval_paths, subj_id):
    X_tr_list, y_tr_list, X_ev_list, y_ev_list = [], [], [], []
    for p in train_paths:
        X, y = load_gdf(p)
        if X is not None: X_tr_list.append(X); y_tr_list.append(y)
    for p in eval_paths:
        X, y = load_gdf(p)
        if X is not None: X_ev_list.append(X); y_ev_list.append(y)

    if not X_tr_list or not X_ev_list: return None, None, None, None
    return (np.concatenate(X_tr_list, axis=0), np.concatenate(y_tr_list, axis=0),
            np.concatenate(X_ev_list, axis=0), np.concatenate(y_ev_list, axis=0))


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
def main(args):
    data_dir = Path(args.data_dir)
    train_files, eval_files = discover_files(data_dir)
    all_subjects = sorted(set(train_files) | set(eval_files))

    if not all_subjects: raise SystemExit(f"Brak plików w {data_dir}")

    # Definicja naszych 4 modeli klasycznych do przetestowania
    # Używamy CSP (z log-wariancją) jako ekstraktora cech dla wszystkich modeli
    models = {
        "LDA": Pipeline([('csp', CSP(n_components=4, log=True)), ('scaler', StandardScaler()),
                         ('clf', LinearDiscriminantAnalysis())]),
        "SVM": Pipeline(
            [('csp', CSP(n_components=4, log=True)), ('scaler', StandardScaler()), ('clf', SVC(kernel='rbf', C=1.0))]),
        "RandomForest": Pipeline([('csp', CSP(n_components=4, log=True)), ('scaler', StandardScaler()),
                                  ('clf', RandomForestClassifier(n_estimators=200, random_state=42))]),
        "LightGBM": Pipeline([('csp', CSP(n_components=4, log=True)), ('scaler', StandardScaler()),
                              ('clf', lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1))])
    }

    # Słowniki na wyniki
    results = {model_name: {} for model_name in models.keys()}

    for subj in all_subjects:
        t_paths = train_files.get(subj, [])
        e_paths = eval_files.get(subj, [])
        if not t_paths or not e_paths: continue

        X_train, y_train, X_eval, y_eval = load_subject(t_paths, e_paths, subj)
        if X_train is None: continue

        print(f"\n[{subj}] Trenowanie modeli dla pacjenta {subj}...")

        # Testowanie każdego modelu na danych konkretnego pacjenta
        for model_name, pipeline in models.items():
            # CSP wyciąga cechy i model uczy się klasyfikować
            pipeline.fit(X_train, y_train)

            # Ewaluacja na sesjach testowych
            y_pred = pipeline.predict(X_eval)
            acc = accuracy_score(y_eval, y_pred)

            results[model_name][subj] = acc
            print(f"  -> {model_name:15s} Accuracy: {acc:.4f}")

    # =========================================================
    # PODSUMOWANIE I DRUKOWANIE WYNIKÓW (Format do publikacji)
    # =========================================================
    print("\n" + "=" * 80)
    print("WYNIKI KOŃCOWE (Dokładność w %) - Tryb: Zależny od pacjenta (Per-Subject)")
    print("=" * 80)

    header = f"{'Pacjent':^10} | {'LDA':^12} | {'SVM':^12} | {'RandomForest':^14} | {'LightGBM':^12}"
    print(header)
    print("-" * len(header))

    # Obliczanie i drukowanie dla każdego pacjenta
    for subj in sorted(results["LDA"].keys()):
        row = f"{subj:^10} | "
        row += f"{results['LDA'][subj] * 100:^12.2f} | "
        row += f"{results['SVM'][subj] * 100:^12.2f} | "
        row += f"{results['RandomForest'][subj] * 100:^14.2f} | "
        row += f"{results['LightGBM'][subj] * 100:^12.2f}"
        print(row)

    print("-" * len(header))

    # Obliczanie średniej (Pooled/Average)
    avg_row = f"{'ŚREDNIA':^10} | "
    avg_row += f"{np.mean(list(results['LDA'].values())) * 100:^12.2f} | "
    avg_row += f"{np.mean(list(results['SVM'].values())) * 100:^12.2f} | "
    avg_row += f"{np.mean(list(results['RandomForest'].values())) * 100:^14.2f} | "
    avg_row += f"{np.mean(list(results['LightGBM'].values())) * 100:^12.2f}"
    print(avg_row)
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data", help="Katalog z plikami *.gdf")
    args = parser.parse_args()
    main(args)