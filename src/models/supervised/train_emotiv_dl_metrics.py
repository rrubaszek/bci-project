"""
Ewaluacja wszystkich modeli Deep Learning na ciągłych sygnałach Emotiv
z dwoma wariantami walidacji krzyżowej.
Uruchomienie: python -m src.models.supervised.train_emotiv_dl_metrics
"""
import random
import warnings
import json
from pathlib import Path

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
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from src.paths import DEFAULT_OUT_DIR, EMOTIV_CLEANED

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

READING_SERIES = (
    ("lewy1", "lewy1_bciciv2b.edf"),
    ("prawy1", "prawy1_bciciv2b.edf"),
    ("lewy2", "lewy2_bciciv2b.edf"),
    ("prawy2", "prawy2_bciciv2b.edf"),
)

CROSS_VALIDATION_CONFIGS = (
    {
        "name": "series_overlap",
        "fold_strategy": "leave_one_series_out",
        "log_dir_name": "leave_one_series_out",
        "duration": 1.0,
        "overlap": 0.95,
        "base_seed": 1000,
        "description": "jedna seria odczytu jako fold, sliding window 1.0s z overlapem 0.95s",
    },
    {
        "name": "mixed_short_no_overlap",
        "fold_strategy": "mixed_stratified",
        "log_dir_name": "mixed_stratified_fold",
        "duration": 0.4,
        "overlap": 0.0,
        "n_splits": 4,
        "base_seed": 2000,
        "description": "klasyczny StratifiedKFold: każdy fold miesza próbki ze wszystkich serii, krótkie okno 0.4s bez overlapu",
    },
)

TRAINING_VARIANTS = (
    {
        "name": "augmented",
        "repeats": 5,
        "noise_std": 0.02,
        "amplitude_min": 0.9,
        "amplitude_max": 1.1,
    },
    {
        "name": "not_augmented",
        "repeats": 1,
        "noise_std": 0.0,
        "amplitude_min": 1.0,
        "amplitude_max": 1.0,
    },
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
    otrzymują niewielkie skalowanie amplitudy oraz szum.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        repeats: int = 5,
        noise_std: float = 0.02,
        amplitude_min: float = 0.9,
        amplitude_max: float = 1.1,
    ):
        if repeats < 1:
            raise ValueError("repeats musi być większe lub równe 1.")

        self.X = torch.from_numpy(X[:, None, :, :]).float()
        self.y = torch.from_numpy(y).long()
        self.repeats = repeats
        self.noise_std = noise_std
        self.amplitude_min = amplitude_min
        self.amplitude_max = amplitude_max

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

        return x, y


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def class_distribution(y: np.ndarray) -> dict[int, int]:
    classes, counts = np.unique(y, return_counts=True)
    return {int(cls): int(count) for cls, count in zip(classes, counts)}


def source_distribution(source_names: np.ndarray) -> dict[str, int]:
    names, counts = np.unique(source_names, return_counts=True)
    return {str(name): int(count) for name, count in zip(names, counts)}


def standardize_fold(
    X_train: np.ndarray,
    X_eval: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(len(X_train), -1)).reshape(X_train.shape).astype(np.float32)
    X_eval = scaler.transform(X_eval.reshape(len(X_eval), -1)).reshape(X_eval.shape).astype(np.float32)
    return X_train, X_eval


def build_model_factories(n_ch: int, n_times: int):
    return {
        "EEGNet": lambda: EEGNet(n_channels=n_ch, n_times=n_times, dropout=0.6),
        "ShallowConvNet": lambda: ShallowConvNet(n_channels=n_ch, n_times=n_times, dropout=0.6),
        "DeepConvNet": lambda: DeepConvNet(n_channels=n_ch, n_times=n_times, dropout=0.6),
        "EEG-Conformer": lambda: EEGConformer(n_channels=n_ch, n_times=n_times, dropout=0.6),
    }


def compute_class_weights(y_train: np.ndarray, device: torch.device) -> torch.Tensor:
    class_counts = np.bincount(y_train, minlength=2)
    if np.any(class_counts == 0):
        raise ValueError(f"Fold treningowy nie zawiera obu klas: {class_counts.tolist()}")
    class_weights = len(y_train) / (2.0 * class_counts)
    return torch.tensor(class_weights, dtype=torch.float32, device=device)


def load_series_for_config(duration: float, overlap: float):
    series_data = []
    for series_name, filename in READING_SERIES:
        X, y = load_segmented_edf(filename, duration, overlap)
        if X is None or y is None:
            raise FileNotFoundError(
                f"Nie udało się przygotować serii {series_name}. "
                "Uruchom najpierw skrypt konwertera."
            )
        series_data.append({
            "name": series_name,
            "filename": filename,
            "X": X,
            "y": y,
            "source": np.full(len(y), series_name),
        })
    return series_data


def concatenate_series_data(series_data):
    return (
        np.concatenate([series["X"] for series in series_data], axis=0),
        np.concatenate([series["y"] for series in series_data], axis=0),
        np.concatenate([series["source"] for series in series_data], axis=0),
    )


def evaluate_model(model, eval_loader, device: torch.device):
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

    return {
        "accuracy": float(accuracy_score(all_targets, all_predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(all_targets, all_predictions)),
        "mcc": float(matthews_corrcoef(all_targets, all_predictions)),
        "y_true": all_targets.tolist(),
        "y_pred": all_predictions.tolist(),
    }


def train_one_model(
    model_name: str,
    model_factory,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    class_weights: torch.Tensor,
    device: torch.device,
    seed: int,
    epochs_limit: int = 100,
):
    set_seed(seed)
    model = model_factory().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)

    # Reset po inicjalizacji modelu sprawia, że kolejność batchy i losowa
    # augmentacja treningowa są identyczne dla modeli w tym samym foldzie.
    set_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        generator=generator,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=32,
        shuffle=False,
    )

    best_metrics = {
        "epoch": 0,
        "accuracy": 0.0,
        "balanced_accuracy": 0.0,
        "mcc": -1.0,
    }

    for epoch in tqdm(range(epochs_limit), ncols=70, desc=model_name):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

        metrics = evaluate_model(model, eval_loader, device)
        current_score = (
            metrics["balanced_accuracy"],
            metrics["mcc"],
            metrics["accuracy"],
        )
        best_score = (
            best_metrics["balanced_accuracy"],
            best_metrics["mcc"],
            best_metrics["accuracy"],
        )


        if current_score > best_score:
            best_metrics = {
                "epoch": epoch + 1,
                **metrics,
            }

    return best_metrics


def build_training_dataset(
    X_train: np.ndarray,
    y_train: np.ndarray,
    training_variant: dict,
) -> Dataset:
    return AugmentedEEGDataset(
        X_train,
        y_train,
        repeats=training_variant["repeats"],
        noise_std=training_variant["noise_std"],
        amplitude_min=training_variant["amplitude_min"],
        amplitude_max=training_variant["amplitude_max"],
    )


def run_prepared_fold(
    config_name: str,
    fold_index: int,
    train_description: str,
    eval_description: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    device: torch.device,
    fold_seed: int,
):
    print("\n" + "-" * 78)
    print(f"Fold {fold_index}: {config_name}")
    print(f"Trening: {train_description}")
    print(f"Ewaluacja: {eval_description}")
    print(f"Rozkład klas treningowych: {class_distribution(y_train)}")
    print(f"Rozkład klas ewaluacyjnych: {class_distribution(y_eval)}")

    X_train, X_eval = standardize_fold(X_train, X_eval)

    eval_dataset = TensorDataset(
        torch.from_numpy(X_eval[:, None, :, :]).float(),
        torch.from_numpy(y_eval).long(),
    )

    print(f"Bazowa pula treningowa: {len(X_train)} okien")
    print(f"Pula ewaluacyjna: {len(X_eval)} okien")

    n_ch, n_times = X_train.shape[1], X_train.shape[2]
    model_factories = build_model_factories(n_ch, n_times)
    class_weights = compute_class_weights(y_train, device)

    fold_results = {}
    for model_name, model_factory in model_factories.items():
        for training_variant in TRAINING_VARIANTS:
            train_dataset = build_training_dataset(X_train, y_train, training_variant)
            result_key = f"{model_name}__{training_variant['name']}"
            print(
                f"\nTrenowanie: {model_name} | "
                f"wariant={training_variant['name']} | "
                f"fold={fold_index} | seed={fold_seed}"
            )
            print(
                f"Pula treningowa dla wariantu {training_variant['name']}: "
                f"{len(train_dataset)} przykładów na epokę"
            )
            metrics = train_one_model(
                model_name=f"{model_name}/{training_variant['name']}",
                model_factory=model_factory,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                class_weights=class_weights,
                device=device,
                seed=fold_seed,
            )
            metrics["fold_index"] = fold_index
            metrics["model"] = model_name
            metrics["training_variant"] = training_variant["name"]
            metrics["augmentation_repeats"] = training_variant["repeats"]
            metrics["noise_std"] = training_variant["noise_std"]
            metrics["amplitude_min"] = training_variant["amplitude_min"]
            metrics["amplitude_max"] = training_variant["amplitude_max"]
            metrics["seed"] = fold_seed
            metrics["train_size"] = len(X_train)
            metrics["train_dataset_size"] = len(train_dataset)
            metrics["eval_size"] = len(X_eval)
            fold_results[result_key] = metrics
            print(
                f"Najlepszy checkpoint: epoka {metrics['epoch']} | "
                f"Acc={metrics['accuracy'] * 100:.2f}% | "
                f"Balanced={metrics['balanced_accuracy'] * 100:.2f}% | "
                f"MCC={metrics['mcc']:.4f}"
            )

    return fold_results


def run_leave_one_series_out_folds(config: dict, series_data, device: torch.device):
    results_by_model = {}

    for fold_index, eval_series in enumerate(series_data, start=1):
        fold_seed = config["base_seed"] + fold_index
        train_series = [
            series
            for series in series_data
            if series["name"] != eval_series["name"]
        ]
        X_train, y_train, _ = concatenate_series_data(train_series)
        X_eval = eval_series["X"]
        y_eval = eval_series["y"]

        fold_results = run_prepared_fold(
            config_name=config["name"],
            fold_index=fold_index,
            train_description=", ".join(series["name"] for series in train_series),
            eval_description=f"seria {eval_series['name']}",
            X_train=X_train,
            y_train=y_train,
            X_eval=X_eval,
            y_eval=y_eval,
            device=device,
            fold_seed=fold_seed,
        )
        for result_key, metrics in fold_results.items():
            metrics["fold"] = eval_series["name"]
            results_by_model.setdefault(result_key, []).append(metrics)

    return results_by_model


def run_mixed_stratified_folds(config: dict, series_data, device: torch.device):
    X_all, y_all, sources_all = concatenate_series_data(series_data)
    splitter = StratifiedKFold(
        n_splits=config["n_splits"],
        shuffle=True,
        random_state=42,
    )
    results_by_model = {}

    for fold_index, (train_idx, eval_idx) in enumerate(splitter.split(X_all, y_all), start=1):
        fold_seed = config["base_seed"] + fold_index
        train_sources = source_distribution(sources_all[train_idx])
        eval_sources = source_distribution(sources_all[eval_idx])
        fold_results = run_prepared_fold(
            config_name=config["name"],
            fold_index=fold_index,
            train_description=f"mieszane próbki ze wszystkich serii {train_sources}",
            eval_description=f"mieszane próbki ze wszystkich serii {eval_sources}",
            X_train=X_all[train_idx],
            y_train=y_all[train_idx],
            X_eval=X_all[eval_idx],
            y_eval=y_all[eval_idx],
            device=device,
            fold_seed=fold_seed,
        )
        for result_key, metrics in fold_results.items():
            metrics["fold"] = f"mixed_{fold_index}"
            results_by_model.setdefault(result_key, []).append(metrics)

    return results_by_model


def summarize_cross_validation_results(results_by_model):
    summary = {}
    for result_key, fold_metrics in results_by_model.items():
        first = fold_metrics[0]
        summary[result_key] = {
            "model": first["model"],
            "training_variant": first["training_variant"],
            "folds": len(fold_metrics),
            "epoch_mean": float(np.mean([m["epoch"] for m in fold_metrics])),
            "accuracy_mean": float(np.mean([m["accuracy"] for m in fold_metrics])),
            "accuracy_std": float(np.std([m["accuracy"] for m in fold_metrics], ddof=0)),
            "balanced_accuracy_mean": float(np.mean([m["balanced_accuracy"] for m in fold_metrics])),
            "balanced_accuracy_std": float(np.std([m["balanced_accuracy"] for m in fold_metrics], ddof=0)),
            "mcc_mean": float(np.mean([m["mcc"] for m in fold_metrics])),
            "mcc_std": float(np.std([m["mcc"] for m in fold_metrics], ddof=0)),
        }
    return summary


def write_cross_validation_results(
    config: dict,
    results_by_model: dict,
    summary: dict,
) -> Path:
    log_dir = DEFAULT_OUT_DIR / "logs" / "supervised" / config["log_dir_name"]
    log_dir.mkdir(parents=True, exist_ok=True)

    fold_results_path = log_dir / "fold_results.tsv"
    with fold_results_path.open("w", encoding="utf-8", newline="") as f:
        f.write(
            "config\tfold_strategy\tfold\tfold_index\tmodel\ttraining_variant\tepoch\t"
            "accuracy\tbalanced_accuracy\tmcc\tseed\ttrain_size\teval_size\t"
            "train_dataset_size\taugmentation_repeats\tnoise_std\tamplitude_min\t"
            "amplitude_max\tduration\toverlap\ty_true\ty_pred\n"
        )
        for result_key in sorted(results_by_model):
            for metrics in sorted(
                results_by_model[result_key],
                key=lambda item: item["fold_index"],
            ):
                f.write(
                    f"{config['name']}\t"
                    f"{config['fold_strategy']}\t"
                    f"{metrics['fold']}\t"
                    f"{metrics['fold_index']}\t"
                    f"{metrics['model']}\t"
                    f"{metrics['training_variant']}\t"
                    f"{metrics['epoch']}\t"
                    f"{metrics['accuracy']:.10f}\t"
                    f"{metrics['balanced_accuracy']:.10f}\t"
                    f"{metrics['mcc']:.10f}\t"
                    f"{metrics['seed']}\t"
                    f"{metrics['train_size']}\t"
                    f"{metrics['eval_size']}\t"
                    f"{metrics['train_dataset_size']}\t"
                    f"{metrics['augmentation_repeats']}\t"
                    f"{metrics['noise_std']}\t"
                    f"{metrics['amplitude_min']}\t"
                    f"{metrics['amplitude_max']}\t"
                    f"{config['duration']}\t"
                    f"{config['overlap']}\t"
                    f"{json.dumps(metrics['y_true'], separators=(',', ':'))}\t"
                    f"{json.dumps(metrics['y_pred'], separators=(',', ':'))}\n"
                )

    summary_path = log_dir / "summary.tsv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        f.write(
            "config\tfold_strategy\tmodel\ttraining_variant\tfolds\tepoch_mean\t"
            "accuracy_mean\taccuracy_std\tbalanced_accuracy_mean\t"
            "balanced_accuracy_std\tmcc_mean\tmcc_std\n"
        )
        for result_key in sorted(summary):
            metrics = summary[result_key]
            f.write(
                f"{config['name']}\t"
                f"{config['fold_strategy']}\t"
                f"{metrics['model']}\t"
                f"{metrics['training_variant']}\t"
                f"{metrics['folds']}\t"
                f"{metrics['epoch_mean']:.10f}\t"
                f"{metrics['accuracy_mean']:.10f}\t"
                f"{metrics['accuracy_std']:.10f}\t"
                f"{metrics['balanced_accuracy_mean']:.10f}\t"
                f"{metrics['balanced_accuracy_std']:.10f}\t"
                f"{metrics['mcc_mean']:.10f}\t"
                f"{metrics['mcc_std']:.10f}\n"
            )

    run_info_path = log_dir / "run_info.txt"
    with run_info_path.open("w", encoding="utf-8") as f:
        f.write(f"config: {config['name']}\n")
        f.write(f"fold_strategy: {config['fold_strategy']}\n")
        f.write(f"description: {config['description']}\n")
        f.write(f"duration: {config['duration']}\n")
        f.write(f"overlap: {config['overlap']}\n")
        f.write(f"base_seed: {config['base_seed']}\n")
        if "n_splits" in config:
            f.write(f"n_splits: {config['n_splits']}\n")
        f.write(f"fold_results: {fold_results_path.name}\n")
        f.write(f"summary: {summary_path.name}\n")

    print(f"\nZapisano wyniki foldów: {fold_results_path}")
    print(f"Zapisano podsumowanie: {summary_path}")
    return fold_results_path


def print_cross_validation_ranking(config_name: str, summary: dict):
    print("\n\n" + "=" * 104)
    print(f"RANKING MODELI NA ZBIORZE EMOTIV EPOC X - WALIDACJA KRZYŻOWA: {config_name}")
    print("Ranking według średniej balanced accuracy, następnie MCC i accuracy")
    print("=" * 104)
    print(
        f"{'Model':<20} {'Wariant':<15} {'Foldy':>5} {'Epoka śr.':>10} "
        f"{'Accuracy mean±std':>22} {'Balanced mean±std':>24} {'MCC mean±std':>18}"
    )
    print("-" * 104)

    sorted_results = sorted(
        summary.items(),
        key=lambda item: (
            item[1]["balanced_accuracy_mean"],
            item[1]["mcc_mean"],
            item[1]["accuracy_mean"],
        ),
        reverse=True,
    )

    for _result_key, metrics in sorted_results:
        print(
            f"{metrics['model']:<20} "
            f"{metrics['training_variant']:<15} "
            f"{metrics['folds']:>5} "
            f"{metrics['epoch_mean']:>10.1f} "
            f"{metrics['accuracy_mean'] * 100:>8.2f}%±{metrics['accuracy_std'] * 100:<6.2f} "
            f"{metrics['balanced_accuracy_mean'] * 100:>10.2f}%±{metrics['balanced_accuracy_std'] * 100:<6.2f} "
            f"{metrics['mcc_mean']:>8.4f}±{metrics['mcc_std']:<6.4f}"
        )
    print("=" * 104)


def run_cross_validation_config(config: dict, device: torch.device):
    print("\n" + "#" * 104)
    print(f"SCHEMAT WALIDACJI KRZYŻOWEJ: {config['name']}")
    print(config["description"])
    print("#" * 104)

    series_data = load_series_for_config(
        duration=config["duration"],
        overlap=config["overlap"],
    )

    if config["fold_strategy"] == "leave_one_series_out":
        results_by_model = run_leave_one_series_out_folds(config, series_data, device)
    elif config["fold_strategy"] == "mixed_stratified":
        results_by_model = run_mixed_stratified_folds(config, series_data, device)
    else:
        raise ValueError(f"Nieznana strategia foldów: {config['fold_strategy']}")

    summary = summarize_cross_validation_results(results_by_model)
    print_cross_validation_ranking(config["name"], summary)
    write_cross_validation_results(config, results_by_model, summary)
    return summary


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Urządzenie Akcelerujące: {device}\n")
    print("=" * 78)
    print("ROZPOCZĘCIE WALIDACJI KRZYŻOWEJ MODELI DEEP LEARNING")
    print("=" * 78)

    all_summaries = {}
    for config in CROSS_VALIDATION_CONFIGS:
        all_summaries[config["name"]] = run_cross_validation_config(config, device)

    return all_summaries

if __name__ == "__main__":
    main()
