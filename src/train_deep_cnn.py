"""
Usage
-----
    python bci_eegnet_train.py --data-dir ./data --epochs 150
    python bci_eegnet_train.py --data-dir ./data --epochs 150 --eval-mode per-subj
"""

import argparse
import re
import warnings
from collections import defaultdict
from pathlib import Path

import mne
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("WARNING")

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

CLASS_NAMES  = ["Left hand", "Right hand"]
SFREQ_TARGET = 250        # Hz – resample target
TMIN, TMAX   = 0.5, 2.5  # epoch window (seconds after cue onset)
EEG_CHANNELS = ["EEG:C3", "EEG:Cz", "EEG:C4"]

# ──────────────────────────────────────────────────────────────────────────────
# 1.  FILE DISCOVERY
# ──────────────────────────────────────────────────────────────────────────────

def discover_files(data_dir: Path):
    """
    Scan data_dir for BCI-IV 2b .gdf files.
    Returns two dicts keyed by subject-id string:
        train_files[subj] = sorted list of *T.gdf paths
        eval_files[subj]  = sorted list of *E.gdf paths
    """
    pattern = re.compile(r"^B(\d{2})\d{2}[TE]\.gdf$", re.IGNORECASE)

    train_files: dict = defaultdict(list)
    eval_files:  dict = defaultdict(list)

    for f in sorted(data_dir.glob("*.gdf")):
        m = pattern.match(f.name)
        if not m:
            continue
        subj = m.group(1)                       # e.g. "01", "02", …
        if f.stem[-1].upper() == "T":
            train_files[subj].append(f)
        else:
            eval_files[subj].append(f)

    return dict(train_files), dict(eval_files)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_gdf(path: Path):
    """Load one .gdf file → (X float32, y int64) or (None, None) on failure."""
    try:
        raw = mne.io.read_raw_gdf(str(path), preload=True, eog=[])
    except Exception as e:
        print(f"  [WARN] Cannot open {path.name}: {e}")
        return None, None

    # band-pass filter 4–40 Hz
    raw.filter(4.0, 40.0, fir_design="firwin", skip_by_annotation="edge")

    # pick EEG channels
    available = raw.ch_names
    picks = [ch for ch in EEG_CHANNELS if ch in available]
    if len(picks) < 3:
        picks = available[:3]
    raw.pick_channels(picks)

    # resample if needed
    if int(raw.info["sfreq"]) != SFREQ_TARGET:
        raw.resample(SFREQ_TARGET)

    # parse events from annotations
    try:
        events, event_id_found = mne.events_from_annotations(raw)
    except Exception:
        print(f"  [WARN] No annotations in {path.name}, skipping.")
        return None, None

    # keep left-hand (769) and right-hand (770) events
    MI_KEYS = {"769", "770"}
    MI_VALS = {1, 2, 769, 770}
    keep = {k: v for k, v in event_id_found.items() if k in MI_KEYS}
    if len(keep) < 2:
        keep = {k: v for k, v in event_id_found.items() if v in MI_VALS}
    if len(keep) < 2:
        print(f"  [WARN] <2 MI event types in {path.name} "
              f"(found: {event_id_found}), skipping.")
        return None, None

    epochs = mne.Epochs(
        raw, events, event_id=keep,
        tmin=TMIN, tmax=TMAX,
        baseline=None, preload=True,
        reject=None, verbose=False,
    )
    X = epochs.get_data().astype(np.float32)
    y_raw = epochs.events[:, -1]

    # remap to {0, 1}
    unique = sorted(np.unique(y_raw))
    label_map = {old: new for new, old in enumerate(unique)}
    y = np.array([label_map[l] for l in y_raw], dtype=np.int64)

    return X, y


def load_subject(train_paths, eval_paths, subj_id):
    """Pool all sessions for one subject → (X_tr, y_tr, X_ev, y_ev)."""
    X_tr_list, y_tr_list = [], []
    for p in train_paths:
        X, y = load_gdf(p)
        if X is not None:
            X_tr_list.append(X)
            y_tr_list.append(y)

    X_ev_list, y_ev_list = [], []
    for p in eval_paths:
        X, y = load_gdf(p)
        if X is not None:
            X_ev_list.append(X)
            y_ev_list.append(y)

    if not X_tr_list or not X_ev_list:
        print(f"  [WARN] Subject {subj_id}: missing train or eval data, skipping.")
        return None, None, None, None

    return (
        np.concatenate(X_tr_list, axis=0),
        np.concatenate(y_tr_list, axis=0),
        np.concatenate(X_ev_list, axis=0),
        np.concatenate(y_ev_list, axis=0),
    )


def z_score_fit_transform(X_train, X_test):
    """Fit scaler on train, apply to both. Returns float32 arrays."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(
        X_train.reshape(len(X_train), -1)
    ).reshape(X_train.shape).astype(np.float32)
    X_te = scaler.transform(
        X_test.reshape(len(X_test), -1)
    ).reshape(X_test.shape).astype(np.float32)
    return X_tr, X_te


# ──────────────────────────────────────────────────────────────────────────────
# 3.  MODEL  –  EEGNet
# ──────────────────────────────────────────────────────────────────────────────

class EEGNet(nn.Module):
    """EEGNet (Lawhern et al., 2018).  Input: (B, 1, n_ch, n_times)."""

    def __init__(self, n_classes=2, n_channels=3, n_times=500,
                 F1=8, D=2, F2=16, dropout=0.5):
        super().__init__()

        # Block 1 – temporal conv + depthwise spatial conv
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, kernel_size=(n_channels, 1),
                      groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )
        # Block 2 – separable conv
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16),
                      padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            flat = self.block2(self.block1(dummy)).view(1, -1).shape[1]

        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(flat, n_classes))

    def forward(self, x):
        return self.classifier(self.block2(self.block1(x)))


# ──────────────────────────────────────────────────────────────────────────────
# 4.  TRAINING HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def make_loader(X, y, batch_size, shuffle):
    return DataLoader(
        TensorDataset(torch.tensor(X[:, None, :, :]), torch.tensor(y)),
        batch_size=batch_size, shuffle=shuffle,
    )


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    loss_sum = correct = n = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(y)
        correct  += (logits.argmax(1) == y).sum().item()
        n        += len(y)
    return loss_sum / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = correct = n = 0
    preds_all, labels_all = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss_sum += criterion(logits, y).item() * len(y)
        p = logits.argmax(1)
        correct  += (p == y).sum().item()
        n        += len(y)
        preds_all.append(p.cpu().numpy())
        labels_all.append(y.cpu().numpy())
    return (
        loss_sum / n, correct / n,
        np.concatenate(preds_all), np.concatenate(labels_all),
    )


# ──────────────────────────────────────────────────────────────────────────────
# 5.  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"Data directory not found: {data_dir}")

    # ── discover all .gdf files ───────────────────────────────────────────────
    train_files, eval_files = discover_files(data_dir)
    all_subjects = sorted(set(train_files) | set(eval_files))

    if not all_subjects:
        raise SystemExit(
            f"No BCI-IV 2b .gdf files found in {data_dir}.\n"
            "Expected filenames like B0101T.gdf, B0203T.gdf, B0304E.gdf …"
        )

    print(f"\nFound {len(all_subjects)} subject(s): {', '.join(all_subjects)}")
    print(f"  Train files total : {sum(len(v) for v in train_files.values())}")
    print(f"  Eval  files total : {sum(len(v) for v in eval_files.values())}\n")

    # ── load all subjects ─────────────────────────────────────────────────────
    X_train_all, y_train_all = [], []
    X_eval_per_subj, y_eval_per_subj = {}, {}

    for subj in all_subjects:
        t_paths = train_files.get(subj, [])
        e_paths = eval_files.get(subj,  [])
        if not t_paths or not e_paths:
            print(f"  [SKIP] Subject {subj}: "
                  f"train={len(t_paths)} eval={len(e_paths)} file(s)")
            continue

        print(f"  Loading subject {subj}  "
              f"({len(t_paths)} train, {len(e_paths)} eval session(s)) …")
        X_tr, y_tr, X_ev, y_ev = load_subject(t_paths, e_paths, subj)
        if X_tr is None:
            continue

        print(f"    train: {len(y_tr)} trials  {np.bincount(y_tr).tolist()}")
        print(f"    eval : {len(y_ev)} trials  {np.bincount(y_ev).tolist()}")

        X_train_all.append(X_tr);  y_train_all.append(y_tr)
        X_eval_per_subj[subj] = X_ev
        y_eval_per_subj[subj] = y_ev

    if not X_train_all:
        raise SystemExit("No usable data found — check your .gdf files.")

    # pool
    X_train = np.concatenate(X_train_all, axis=0)
    y_train = np.concatenate(y_train_all, axis=0)
    X_eval  = np.concatenate(list(X_eval_per_subj.values()), axis=0)
    y_eval  = np.concatenate(list(y_eval_per_subj.values()), axis=0)

    print(f"\nPooled train : {X_train.shape}  labels={np.bincount(y_train).tolist()}")
    print(f"Pooled eval  : {X_eval.shape}   labels={np.bincount(y_eval).tolist()}")

    # ── normalise (scaler fitted on train only) ───────────────────────────────
    X_train, X_eval = z_score_fit_transform(X_train, X_eval)

    # keep per-subject eval slices aligned after normalisation
    if args.eval_mode == "per-subj":
        offset = 0
        for subj in list(X_eval_per_subj.keys()):
            n = len(y_eval_per_subj[subj])
            X_eval_per_subj[subj] = X_eval[offset: offset + n]
            offset += n

    # ── data loaders ──────────────────────────────────────────────────────────
    train_loader = make_loader(X_train, y_train, args.batch_size, shuffle=True)
    eval_loader  = make_loader(X_eval,  y_eval,  args.batch_size, shuffle=False)

    # ── model ─────────────────────────────────────────────────────────────────
    n_ch, n_times = X_train.shape[1], X_train.shape[2]
    model = EEGNet(
        n_classes=2, n_channels=n_ch, n_times=n_times,
        dropout=args.dropout,
    ).to(device)
    print(f"\nEEGNet — n_ch={n_ch}  n_times={n_times}  "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # ── training loop ─────────────────────────────────────────────────────────
    best_acc = best_epoch = 0
    best_state = None

    print(f"\nTraining for {args.epochs} epochs …\n")
    hdr = f"{'Epoch':>6}  {'Tr Loss':>8}  {'Tr Acc':>7}  {'Va Loss':>8}  {'Va Acc':>7}"
    print(hdr); print("─" * len(hdr))

    for epoch in tqdm(range(1, args.epochs + 1), desc="Training", ncols=72):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc, _, _ = evaluate(model, eval_loader, criterion, device)
        scheduler.step()

        if va_acc > best_acc:
            best_acc, best_epoch = va_acc, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            tqdm.write(f"{epoch:6d}  {tr_loss:8.4f}  {tr_acc:7.4f}  "
                       f"{va_loss:8.4f}  {va_acc:7.4f}")

    # ── final evaluation ──────────────────────────────────────────────────────
    model.load_state_dict(best_state)

    print(f"\n{'='*56}")
    print(f"FINAL RESULTS  (best checkpoint: epoch {best_epoch})")
    print(f"{'='*56}")

    _, pooled_acc, preds, labels = evaluate(model, eval_loader, criterion, device)
    print(f"\nPooled eval accuracy : {pooled_acc:.4f}\n")
    print(classification_report(labels, preds, target_names=CLASS_NAMES))
    print("Confusion matrix (pooled):")
    print(confusion_matrix(labels, preds))

    if args.eval_mode == "per-subj":
        print(f"\n{'─'*40}")
        print("Per-subject accuracy:")
        print(f"{'─'*40}")
        for subj in sorted(X_eval_per_subj.keys()):
            s_loader = make_loader(
                X_eval_per_subj[subj], y_eval_per_subj[subj],
                args.batch_size, shuffle=False,
            )
            _, acc_s, _, _ = evaluate(model, s_loader, criterion, device)
            n_s = len(y_eval_per_subj[subj])
            print(f"  Subject {subj}:  acc={acc_s:.4f}  (n={n_s} trials)")

    # ── save checkpoint ───────────────────────────────────────────────────────
    if args.save:
        torch.save({
            "model_state": best_state,
            "config": {"n_channels": n_ch, "n_times": n_times,
                       "dropout": args.dropout},
            "subjects_trained": sorted(X_eval_per_subj.keys()),
        }, args.save)
        print(f"\nModel saved → {args.save}")


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train EEGNet on ALL BCI-IV 2b subjects in a directory"
    )
    parser.add_argument(
        "--data-dir", default="./data",
        help="Folder containing all *.gdf files  (default: ./data)"
    )
    parser.add_argument(
        "--eval-mode", choices=["pooled", "per-subj"], default="pooled",
        help="'pooled': single combined test set  |  "
             "'per-subj': also prints per-subject accuracy"
    )
    parser.add_argument("--epochs",       type=int,   default=150)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout",      type=float, default=0.5)
    parser.add_argument("--batch-size",   type=int,   default=32)
    parser.add_argument(
        "--save", default="eegnet_best.pt",
        help="Checkpoint output path ('' to disable)"
    )
    args = parser.parse_args()
    main(args)