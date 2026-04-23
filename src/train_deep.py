"""
Trening modelu Deep ConvNet (Schirrmeister et al., 2017) na zbiorze BCI-IV 2b.
Uruchomienie: python train_deep.py --data-dir ./data --epochs 150 --eval-mode per-subj
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
# CONSTANTS & DATA LOADING (Identyczne jak w Shallow ConvNet dla uczciwego porównania)
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
        print(f"  [Błąd MNE] Nie udało się otworzyć {path.name}: {e}")
        return None, None

    raw.filter(4.0, 40.0, fir_design="firwin", skip_by_annotation="edge")
    available = raw.ch_names
    picks = [ch for ch in EEG_CHANNELS if ch in available]
    if len(picks) < 3: picks = available[:3]
    raw.pick_channels(picks)

    if int(raw.info["sfreq"]) != SFREQ_TARGET:
        raw.resample(SFREQ_TARGET)

    try:
        events, event_id_found = mne.events_from_annotations(raw)
    except Exception:
        print(f"  [Brak Zdarzeń] {path.name} nie ma adnotacji.")
        return None, None

    MI_KEYS = {"769", "770"}
    MI_VALS = {1, 2, 769, 770}
    keep = {k: v for k, v in event_id_found.items() if k in MI_KEYS}
    if len(keep) < 2:
        keep = {k: v for k, v in event_id_found.items() if v in MI_VALS}

    if len(keep) < 2:
        print(f"  [Brak Klas] {path.name} - znalezione etykiety: {event_id_found}")
        return None, None

    epochs = mne.Epochs(raw, events, event_id=keep, tmin=TMIN, tmax=TMAX,
                        baseline=None, preload=True, reject=None, verbose=False)
    X = epochs.get_data().astype(np.float32)
    y_raw = epochs.events[:, -1]

    unique = sorted(np.unique(y_raw))
    label_map = {old: new for new, old in enumerate(unique)}
    y = np.array([label_map[l] for l in y_raw], dtype=np.int64)
    return X, y


def load_subject(train_paths, eval_paths, subj_id):
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

    if not X_tr_list or not X_ev_list: return None, None, None, None
    return (np.concatenate(X_tr_list, axis=0), np.concatenate(y_tr_list, axis=0),
            np.concatenate(X_ev_list, axis=0), np.concatenate(y_ev_list, axis=0))


def z_score_fit_transform(X_train, X_test):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train.reshape(len(X_train), -1)).reshape(X_train.shape).astype(np.float32)
    X_te = scaler.transform(X_test.reshape(len(X_test), -1)).reshape(X_test.shape).astype(np.float32)
    return X_tr, X_te


# ──────────────────────────────────────────────────────────────────────────────
# 2. NOWY MODEL: DEEP CONVNET
# ──────────────────────────────────────────────────────────────────────────────
class DeepConvNet(nn.Module):
    """
    Deep Convolutional Neural Network (Schirrmeister et al., 2017)
    Wejście: (Batch, 1, n_channels, n_times)
    """

    def __init__(self, n_classes=2, n_channels=3, n_times=500, n_filters_time=25,
                 n_filters_spat=25, filter_time_length=10, pool_time_length=3,
                 pool_time_stride=3, dropout=0.5):
        super().__init__()

        # Blok 1: Ekstrakcja cech czasowych i przestrzennych (podobnie jak w Shallow, ale z ELU)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, n_filters_time, kernel_size=(1, filter_time_length), bias=False),
            nn.Conv2d(n_filters_time, n_filters_spat, kernel_size=(n_channels, 1), bias=False),
            nn.BatchNorm2d(n_filters_spat),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, pool_time_length), stride=(1, pool_time_stride)),
            nn.Dropout(dropout)
        )

        # Blok 2: Głębsza ekstrakcja
        self.block2 = nn.Sequential(
            nn.Conv2d(n_filters_spat, n_filters_spat * 2, kernel_size=(1, filter_time_length), bias=False),
            nn.BatchNorm2d(n_filters_spat * 2),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, pool_time_length), stride=(1, pool_time_stride)),
            nn.Dropout(dropout)
        )

        # Blok 3: Jeszcze głębsza ekstrakcja
        self.block3 = nn.Sequential(
            nn.Conv2d(n_filters_spat * 2, n_filters_spat * 4, kernel_size=(1, filter_time_length), bias=False),
            nn.BatchNorm2d(n_filters_spat * 4),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, pool_time_length), stride=(1, pool_time_stride)),
            nn.Dropout(dropout)
        )

        # Blok 4: Ostatnia warstwa konwolucyjna przed klasyfikatorem
        self.block4 = nn.Sequential(
            nn.Conv2d(n_filters_spat * 4, n_filters_spat * 8, kernel_size=(1, filter_time_length), bias=False),
            nn.BatchNorm2d(n_filters_spat * 8),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, pool_time_length), stride=(1, pool_time_stride)),
            nn.Dropout(dropout)
        )

        # Wyliczenie wielkości warstwy spłaszczonej
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            out = self.block4(self.block3(self.block2(self.block1(dummy))))
            flat_size = out.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, n_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.classifier(x)


# ──────────────────────────────────────────────────────────────────────────────
# 3. TRAINING HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def make_loader(X, y, batch_size, shuffle):
    return DataLoader(TensorDataset(torch.tensor(X[:, None, :, :]), torch.tensor(y)), batch_size=batch_size,
                      shuffle=shuffle)


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
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
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
        correct += (p == y).sum().item()
        n += len(y)
        preds_all.append(p.cpu().numpy())
        labels_all.append(y.cpu().numpy())
    return loss_sum / n, correct / n, np.concatenate(preds_all), np.concatenate(labels_all)


# ──────────────────────────────────────────────────────────────────────────────
# 4. MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Urządzenie: {device}")

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir(): raise SystemExit(f"Nie znaleziono: {data_dir}")

    train_files, eval_files = discover_files(data_dir)
    all_subjects = sorted(set(train_files) | set(eval_files))

    if not all_subjects: raise SystemExit(f"Brak plików w {data_dir}")

    X_train_all, y_train_all = [], []
    X_eval_per_subj, y_eval_per_subj = {}, {}

    for subj in all_subjects:
        t_paths = train_files.get(subj, [])
        e_paths = eval_files.get(subj, [])
        if not t_paths or not e_paths:
            print(
                f"Pominięto pacjenta {subj}: Posiada {len(t_paths)} plików Treningowych i {len(e_paths)} Ewaluacyjnych.")
            continue

        print(f"Wczytywanie pacjenta {subj}...")
        X_tr, y_tr, X_ev, y_ev = load_subject(t_paths, e_paths, subj)
        if X_tr is None:
            print(f"Pominięto pacjenta {subj}: Błąd ekstrakcji danych z plików GDF.")
            continue

        X_train_all.append(X_tr)
        y_train_all.append(y_tr)
        X_eval_per_subj[subj] = X_ev
        y_eval_per_subj[subj] = y_ev

    X_train = np.concatenate(X_train_all, axis=0)
    y_train = np.concatenate(y_train_all, axis=0)
    X_eval = np.concatenate(list(X_eval_per_subj.values()), axis=0)
    y_eval = np.concatenate(list(y_eval_per_subj.values()), axis=0)

    X_train, X_eval = z_score_fit_transform(X_train, X_eval)

    if args.eval_mode == "per-subj":
        offset = 0
        for subj in list(X_eval_per_subj.keys()):
            n = len(y_eval_per_subj[subj])
            X_eval_per_subj[subj] = X_eval[offset: offset + n]
            offset += n

    train_loader = make_loader(X_train, y_train, args.batch_size, shuffle=True)
    eval_loader = make_loader(X_eval, y_eval, args.batch_size, shuffle=False)

    n_ch, n_times = X_train.shape[1], X_train.shape[2]

    # Zastąpienie ShallowConvNet modelem DeepConvNet
    model = DeepConvNet(n_classes=2, n_channels=n_ch, n_times=n_times, dropout=args.dropout).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = best_epoch = 0
    best_state = None

    print(f"\nTrening Deep ConvNet przez {args.epochs} epok...\n")
    for epoch in tqdm(range(1, args.epochs + 1), desc="Training", ncols=72):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc, _, _ = evaluate(model, eval_loader, criterion, device)
        scheduler.step()

        if va_acc > best_acc:
            best_acc, best_epoch = va_acc, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    print(f"\n{'=' * 56}")
    print(f"WYNIKI DEEP CONVNET (Najlepsza epoka: {best_epoch})")
    print(f"{'=' * 56}")

    _, pooled_acc, preds, labels = evaluate(model, eval_loader, criterion, device)
    print(f"\nDokładność ogólna (Pooled) : {pooled_acc:.4f}\n")

    if args.eval_mode == "per-subj":
        print(f"\n{'─' * 40}\nDokładność dla poszczególnych badanych:\n{'─' * 40}")
        for subj in sorted(X_eval_per_subj.keys()):
            s_loader = make_loader(X_eval_per_subj[subj], y_eval_per_subj[subj], args.batch_size, shuffle=False)
            _, acc_s, _, _ = evaluate(model, s_loader, criterion, device)
            print(f"  Pacjent {subj}:  acc={acc_s:.4f}")

    if args.save:
        torch.save(best_state, args.save)
        print(f"\nZapisano model -> {args.save}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--eval-mode", choices=["pooled", "per-subj"], default="per-subj")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save", default="deep_best.pt")
    args = parser.parse_args()
    main(args)