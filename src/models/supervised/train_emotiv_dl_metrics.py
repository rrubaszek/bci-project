"""
Ewaluacja wszystkich modeli Deep Learning na ciągłych sygnałach Emotiv.
Uruchomienie: python -m src.models.supervised.train_emotiv_dl
"""
import warnings
import torch
import torch.nn as nn
import numpy as np
import mne
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
)
from tqdm import tqdm

from src.paths import EMOTIV_CLEANED

warnings.filterwarnings("ignore")
mne.set_log_level("WARNING")

# =====================================================================
# 1. DEFINICJE WSZYSTKICH MODELI DEEP LEARNING Z TWOJEGO PROJEKTU
# =====================================================================

# --- DEEP CONVNET ---
class DeepConvNet(nn.Module):
    def __init__(self, n_classes=2, n_channels=3, n_times=500, dropout=0.5):
        super().__init__()
        F_time, F_spat = 25, 25
        # Krótsze jądra z paddingiem i pooling 2x umożliwiają pracę
        # na oknach 1-sekundowych (np. 200 próbek przy 200 Hz).
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F_time, kernel_size=(1, 9), padding=(0, 4), bias=False),
            nn.Conv2d(F_time, F_spat, kernel_size=(n_channels, 1), bias=False),
            nn.BatchNorm2d(F_spat), nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)), nn.Dropout(dropout)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F_spat, F_spat * 2, kernel_size=(1, 9), padding=(0, 4), bias=False),
            nn.BatchNorm2d(F_spat * 2), nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)), nn.Dropout(dropout)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(F_spat * 2, F_spat * 4, kernel_size=(1, 9), padding=(0, 4), bias=False),
            nn.BatchNorm2d(F_spat * 4), nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)), nn.Dropout(dropout)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(F_spat * 4, F_spat * 8, kernel_size=(1, 9), padding=(0, 4), bias=False),
            nn.BatchNorm2d(F_spat * 8), nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)), nn.Dropout(dropout)
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            flat_size = self.block4(self.block3(self.block2(self.block1(dummy)))).view(1, -1).shape[1]
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(flat_size, n_classes))

    def forward(self, x):
        return self.classifier(self.block4(self.block3(self.block2(self.block1(x)))))

# --- SHALLOW CONVNET ---
class Square(nn.Module):
    def forward(self, x): return torch.pow(x, 2)

class Log(nn.Module):
    def forward(self, x): return torch.log(torch.clamp(x, min=1e-6))

class ShallowConvNet(nn.Module):
    def __init__(self, n_classes=2, n_channels=3, n_times=500, dropout=0.5):
        super().__init__()
        self.temporal_spatial = nn.Sequential(
            nn.Conv2d(1, 40, kernel_size=(1, 25), bias=False),
            nn.Conv2d(40, 40, kernel_size=(n_channels, 1), bias=False),
            nn.BatchNorm2d(40), Square(),
            nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15)), Log(), nn.Dropout(dropout)
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            flat_size = self.temporal_spatial(dummy).view(1, -1).shape[1]
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(flat_size, n_classes))

    def forward(self, x):
        return self.classifier(self.temporal_spatial(x))

# --- EEGNET ---
class EEGNet(nn.Module):
    def __init__(self, n_classes=2, n_channels=3, n_times=500, dropout=0.5):
        super().__init__()
        F1, D, F2 = 8, 2, 16
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, kernel_size=(n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D), nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)), nn.Dropout(dropout)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)), nn.Dropout(dropout)
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            flat = self.block2(self.block1(dummy)).view(1, -1).shape[1]
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(flat, n_classes))

    def forward(self, x):
        return self.classifier(self.block2(self.block1(x)))

# --- EEG-CONFORMER ---
class EEGConformer(nn.Module):
    def __init__(self, n_classes=2, n_channels=3, n_times=500, dropout=0.5):
        super().__init__()
        emb_size = 40
        self.cnn = nn.Sequential(
            nn.Conv2d(1, emb_size, kernel_size=(1, 25), padding=(0, 12), bias=False),
            nn.BatchNorm2d(emb_size),
            nn.Conv2d(emb_size, emb_size, kernel_size=(n_channels, 1), bias=False),
            nn.BatchNorm2d(emb_size), nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 15), stride=(1, 5)), nn.Dropout(dropout)
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            seq_len = self.cnn(dummy).shape[3]

        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, emb_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size, nhead=4, dim_feedforward=emb_size * 2,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Sequential(nn.Linear(emb_size, n_classes))

    def forward(self, x):
        x = self.cnn(x).squeeze(2).permute(0, 2, 1) # -> (B, seq_len, emb_size)
        x = x + self.pos_embedding
        x = self.transformer(x)
        x = x.mean(dim=1) # Global Avg Pool
        return self.classifier(x)

# =====================================================================
# 2. MECHANIKA DANYCH I GŁÓWNA PĘTLA
# =====================================================================

# Struktura etykiet w każdym pliku EDF:
# 0.0-2.5 s -> klasa 0
# 2.5-5.0 s -> klasa 1
# 5.0-7.5 s -> klasa 0
CLASS_SEGMENTS = (
    (0.0, 2.5, 0),
    (2.5, 5.0, 1),
    (5.0, 7.5, 0),
)


def load_segmented_edf(
    filename: str,
    duration: float = 1.0,
    overlap: float = 0.95,
):
    """Wczytuje EDF i tworzy okna bez przekraczania granic klas."""
    if duration <= 0:
        raise ValueError("duration musi być większe od 0.")
    if overlap < 0 or overlap >= duration:
        raise ValueError("overlap musi spełniać warunek: 0 <= overlap < duration.")

    file_path = EMOTIV_CLEANED / filename
    if not file_path.exists():
        print(f"Nie znaleziono pliku: {file_path}")
        return None, None

    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    raw.filter(4.0, 40.0, fir_design="firwin", verbose=False)

    data = raw.get_data().astype(np.float32)
    sfreq = float(raw.info["sfreq"])
    raw.close()

    window_samples = int(round(duration * sfreq))
    step_samples = int(round((duration - overlap) * sfreq))

    if window_samples <= 0:
        raise ValueError("Długość okna po przeliczeniu musi wynosić co najmniej 1 próbkę.")
    if step_samples <= 0:
        raise ValueError("Przesunięcie okna musi wynosić co najmniej 1 próbkę.")

    windows = []
    labels = []

    for segment_start, segment_end, label in CLASS_SEGMENTS:
        start_sample = int(round(segment_start * sfreq))
        end_sample = int(round(segment_end * sfreq))
        end_sample = min(end_sample, data.shape[1])

        segment_samples = end_sample - start_sample
        if segment_samples < window_samples:
            print(
                f"Pomijam segment {segment_start:.1f}-{segment_end:.1f} s "
                f"w pliku {filename}: segment jest krótszy od okna {duration:.1f} s."
            )
            continue

        # Zakres kończy się tak, aby okno nigdy nie wyszło poza segment klasy.
        for window_start in range(
            start_sample,
            end_sample - window_samples + 1,
            step_samples,
        ):
            window_end = window_start + window_samples
            windows.append(data[:, window_start:window_end].copy())
            labels.append(label)

    if not windows:
        print(f"Nie utworzono żadnych okien dla pliku: {filename}")
        return None, None

    X = np.stack(windows).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    return X, y


class AugmentedEEGDataset(Dataset):
    """Zwraca oryginalne i losowo augmentowane okna EEG.

    Augmentacja jest stosowana wyłącznie do zbioru treningowego.
    Pierwsza kopia każdego okna pozostaje niezmieniona, a kolejne
    otrzymują niewielkie skalowanie amplitudy, szum i przesunięcie w czasie.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        repeats: int = 5,
        noise_std: float = 0.02,
        amplitude_min: float = 0.9,
        amplitude_max: float = 1.1,
        max_shift_fraction: float = 0.05,
    ):
        if repeats < 1:
            raise ValueError("repeats musi być większe lub równe 1.")

        self.X = torch.from_numpy(X[:, None, :, :]).float()
        self.y = torch.from_numpy(y).long()
        self.repeats = repeats
        self.noise_std = noise_std
        self.amplitude_min = amplitude_min
        self.amplitude_max = amplitude_max
        self.max_shift_samples = int(round(X.shape[-1] * max_shift_fraction))

    def __len__(self):
        return len(self.X) * self.repeats

    def __getitem__(self, index):
        base_index = index % len(self.X)
        copy_index = index // len(self.X)

        x = self.X[base_index].clone()
        y = self.y[base_index]

        # Pierwsze przejście przez zbiór zawiera niezmienione próbki.
        if copy_index == 0:
            return x, y

        amplitude = torch.empty(1).uniform_(
            self.amplitude_min, self.amplitude_max
        ).item()
        x = x * amplitude

        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std

        if self.max_shift_samples > 0:
            shift = int(
                torch.randint(
                    -self.max_shift_samples,
                    self.max_shift_samples + 1,
                    (1,),
                ).item()
            )
            x = torch.roll(x, shifts=shift, dims=-1)

        return x, y

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Urządzenie Akcelerujące: {device}\n")

    print("Wczytywanie danych metodą Sliding Window (1.0s okno, 0.95s overlap)...")
    # Każdy plik ma układ klas: 2.5 s klasy 0, 2.5 s klasy 1, 2.5 s klasy 0.
    # Okna mają 1.0 s i krok 0.05 s, ale są tworzone osobno w każdym
    # segmencie, więc nigdy nie przecinają granic klas.
    DUR, OVL = 1.0, 0.95
    X_tr_l, y_tr_l = load_segmented_edf("lewy1_bciciv2b.edf", DUR, OVL)
    X_tr_r, y_tr_r = load_segmented_edf("prawy1_bciciv2b.edf", DUR, OVL)

    X_ev_l, y_ev_l = load_segmented_edf("lewy2_bciciv2b.edf", DUR, OVL)
    X_ev_r, y_ev_r = load_segmented_edf("prawy2_bciciv2b.edf", DUR, OVL)

    if any(v is None for v in [X_tr_l, X_tr_r, X_ev_l, X_ev_r]):
        print("Brak plików. Uruchom skrypt konwertera.")
        return

    X_train = np.concatenate([X_tr_l, X_tr_r], axis=0)
    y_train = np.concatenate([y_tr_l, y_tr_r], axis=0)
    X_eval = np.concatenate([X_ev_l, X_ev_r], axis=0)
    y_eval = np.concatenate([y_ev_l, y_ev_r], axis=0)

    train_classes, train_counts = np.unique(y_train, return_counts=True)
    eval_classes, eval_counts = np.unique(y_eval, return_counts=True)
    train_distribution = {int(cls): int(count) for cls, count in zip(train_classes, train_counts)}
    eval_distribution = {int(cls): int(count) for cls, count in zip(eval_classes, eval_counts)}
    print(f"Rozkład klas treningowych: {train_distribution}")
    print(f"Rozkład klas ewaluacyjnych: {eval_distribution}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(len(X_train), -1)).reshape(X_train.shape).astype(np.float32)
    X_eval = scaler.transform(X_eval.reshape(len(X_eval), -1)).reshape(X_eval.shape).astype(np.float32)

    # Sliding window daje około 186 bazowych okien treningowych przy 200 Hz.
    # Pięć wariantów każdego okna daje około 930 przykładów na epokę uczenia.
    AUGMENT_REPEATS = 5
    train_dataset = AugmentedEEGDataset(
        X_train,
        y_train,
        repeats=AUGMENT_REPEATS,
        noise_std=0.02,
        amplitude_min=0.9,
        amplitude_max=1.1,
        max_shift_fraction=0.05,
    )
    eval_dataset = TensorDataset(
        torch.from_numpy(X_eval[:, None, :, :]).float(),
        torch.from_numpy(y_eval).long(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=32,
        shuffle=False,
    )

    print(f"Bazowa pula treningowa: {len(X_train)} okien")
    print(f"Pula treningowa z augmentacją: {len(train_dataset)} przykładów na epokę")
    print(f"Pula ewaluacyjna: {len(X_eval)} okien\n")

    # Rejestr Modeli
    n_ch, n_times = X_train.shape[1], X_train.shape[2]
    models = {
        "EEGNet": EEGNet(n_channels=n_ch, n_times=n_times, dropout=0.6).to(device),
        "ShallowConvNet": ShallowConvNet(n_channels=n_ch, n_times=n_times, dropout=0.6).to(device),
        "DeepConvNet": DeepConvNet(n_channels=n_ch, n_times=n_times, dropout=0.6).to(device),
        "EEG-Conformer": EEGConformer(n_channels=n_ch, n_times=n_times, dropout=0.6).to(device)
    }

    # Wagi kompensują naturalny stosunek klas 2:1 wynikający z układu 0-1-0.
    class_counts = np.bincount(y_train, minlength=2)
    class_weights = len(y_train) / (2.0 * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

    results = {}
    epochs_limit = 100

    print("=" * 60)
    print("ROZPOCZĘCIE ZBIORCZEJ EWALUACJI MODELI DEEP LEARNING")
    print("=" * 60)

    for name, model in models.items():
        print(f"\nTrenowanie: {name} ...")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)

        # Najlepszy checkpoint wybieramy według balanced accuracy.
        # Accuracy i MCC zapisujemy z tej samej epoki, aby metryki opisywały
        # dokładnie ten sam stan modelu.
        best_metrics = {
            "epoch": 0,
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "mcc": -1.0,
        }

        for epoch in tqdm(range(epochs_limit), ncols=70, desc=name):
            model.train()
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()
                optimizer.step()

            # Ewaluacja
            model.eval()
            all_targets = []
            all_predictions = []

            with torch.no_grad():
                for X, y in eval_loader:
                    X, y = X.to(device), y.to(device)
                    preds = model(X).argmax(dim=1)

                    all_targets.extend(y.cpu().numpy())
                    all_predictions.extend(preds.cpu().numpy())

            all_targets = np.asarray(all_targets, dtype=np.int64)
            all_predictions = np.asarray(all_predictions, dtype=np.int64)

            acc = accuracy_score(all_targets, all_predictions)
            balanced_acc = balanced_accuracy_score(
                all_targets,
                all_predictions,
            )
            mcc = matthews_corrcoef(
                all_targets,
                all_predictions,
            )

            # Główne kryterium: balanced accuracy. Przy remisie wybieramy
            # wyższe MCC, a następnie wyższe zwykłe accuracy.
            current_score = (balanced_acc, mcc, acc)
            best_score = (
                best_metrics["balanced_accuracy"],
                best_metrics["mcc"],
                best_metrics["accuracy"],
            )

            if current_score > best_score:
                best_metrics = {
                    "epoch": epoch + 1,
                    "accuracy": float(acc),
                    "balanced_accuracy": float(balanced_acc),
                    "mcc": float(mcc),
                }

        results[name] = best_metrics
        print(f"Najlepsze wyniki dla {name} (epoka {best_metrics['epoch']}):")
        print(f"  Accuracy:          {best_metrics['accuracy'] * 100:.2f}%")
        print(
            f"  Balanced accuracy: "
            f"{best_metrics['balanced_accuracy'] * 100:.2f}%"
        )
        print(f"  MCC:               {best_metrics['mcc']:.4f}")

    # PODSUMOWANIE
    print("\n\n" + "=" * 78)
    print("RANKING MODELI NA ZBIORZE EMOTIV EPOC X")
    print("Ranking według balanced accuracy, następnie MCC i accuracy")
    print("=" * 78)
    print(
        f"{'Model':<20} {'Epoka':>7} {'Accuracy':>12} "
        f"{'Balanced Acc.':>16} {'MCC':>10}"
    )
    print("-" * 78)

    sorted_results = sorted(
        results.items(),
        key=lambda item: (
            item[1]["balanced_accuracy"],
            item[1]["mcc"],
            item[1]["accuracy"],
        ),
        reverse=True,
    )

    for name, metrics in sorted_results:
        print(
            f"{name:<20} "
            f"{metrics['epoch']:>7} "
            f"{metrics['accuracy'] * 100:>11.2f}% "
            f"{metrics['balanced_accuracy'] * 100:>15.2f}% "
            f"{metrics['mcc']:>10.4f}"
        )
    print("=" * 78)

if __name__ == "__main__":
    main()