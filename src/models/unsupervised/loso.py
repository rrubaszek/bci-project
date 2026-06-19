from sklearn.preprocessing import StandardScaler
import copy
import numpy as np
import mne
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
)
import pandas as pd

import matplotlib.pyplot as plt
from src.paths import EMOTIV_CLEANED


def get_ground_truth_labels(
    raw: mne.io.Raw, window_duration: float, window_overlap: float
) -> np.ndarray:
    total_duration = raw.times[-1]

    step_sec = window_duration - window_overlap

    labels = []
    current_time = 0.0

    while current_time + window_duration <= total_duration:
        window_center = current_time + (window_duration / 2.0)

        if 2.5 <= window_center <= 5.0:
            labels.append(1)
        else:
            labels.append(0)

        current_time += step_sec

    return np.array(labels)


def run_loso_evaluation(models: dict, args) -> dict:
    cleaned_files = list(EMOTIV_CLEANED.glob("*.fif"))

    session_raws = {
        f.stem: mne.io.read_raw_fif(f, preload=True, verbose=False)
        for f in cleaned_files
    }
    sessions = list(session_raws.keys())

    # Lista, do której będziemy zbierać pojedyncze wiersze do pliku CSV
    detailed_rows = []

    # Słownik na średnie wyniki (żeby zachować stary print na końcu, jeśli chcesz)
    final_results = {}

    for model_name, pipeline in models.items():
        print("\n==================================================")
        print(f" Uruchamiam LOSO CV dla modelu: {model_name}")
        print("==================================================")

        if model_name == "GMM":
            current_duration = args.gmm_window_duration
            current_overlap = args.gmm_window_overlap
        elif model_name == "HMM":
            current_duration = args.hmm_window_duration
            current_overlap = args.hmm_window_overlap
        else:
            current_duration = 1.0
            current_overlap = 0.5

        loso_accuracy = []
        loso_balanced_acc = []
        loso_f1 = []
        loso_mcc = []

        for test_session in sessions:
            # --- START ZMIANY: Standaryzacja per-plik ---

            # 1. Pobieramy i skalujemy niezależnie pliki treningowe
            train_raws_scaled = []
            for s in sessions:
                if s != test_session:
                    raw_copy = session_raws[s].copy()
                    data = raw_copy.get_data()  # Kształt: (n_channels, n_times)

                    # Scaler potrzebuje kształtu (n_samples, n_features), stąd transpozycja .T
                    scaler = StandardScaler()
                    scaled_data_T = scaler.fit_transform(data.T)
                    scaled_data = scaled_data_T.T  # Powrót do (n_channels, n_times)

                    # Tworzymy nowy obiekt RawArray z zachowaniem oryginalnego info
                    raw_scaled = mne.io.RawArray(
                        scaled_data, raw_copy.info, verbose=False
                    )
                    train_raws_scaled.append(raw_scaled)

            # 2. Pobieramy i skalujemy niezależnie plik testowy
            test_raw_copy = session_raws[test_session].copy()
            test_data = test_raw_copy.get_data()

            test_scaler = StandardScaler()
            test_scaled_data_T = test_scaler.fit_transform(test_data.T)
            test_scaled_data = test_scaled_data_T.T

            test_raw = mne.io.RawArray(
                test_scaled_data, test_raw_copy.info, verbose=False
            )

            # 3. Łączymy już niezależnie przeskalowane pliki treningowe
            combined_train_raw = mne.concatenate_raws(train_raws_scaled, verbose=False)

            # --- KONIEC ZMIANY ---

            fold_pipeline = copy.deepcopy(pipeline)
            fold_pipeline.fit(combined_train_raw)

            # Predykcja na zbiorze treningowym do mapowania klastrów
            train_preds = fold_pipeline.predict(combined_train_raw).ravel()
            train_trues = get_ground_truth_labels(
                combined_train_raw, current_duration, current_overlap
            ).ravel()

            min_len_tr = min(len(train_preds), len(train_trues))
            train_preds = train_preds[:min_len_tr]
            train_trues = train_trues[:min_len_tr]

            unique_states = np.unique(train_preds)

            # Dynamiczne mapowanie klastrów
            state_ratios = {}
            for state in unique_states:
                ratio = np.mean(train_trues[train_preds == state] == 1)
                state_ratios[state] = ratio

            sorted_states = sorted(
                state_ratios.keys(), key=lambda x: state_ratios[x], reverse=True
            )

            label_mapping = {}
            if len(sorted_states) > 0:
                label_mapping[sorted_states[0]] = 1
                for state in sorted_states[1:]:
                    label_mapping[state] = 0

            # Predykcja testowa
            test_preds_raw = fold_pipeline.predict(test_raw).ravel()
            y_true = get_ground_truth_labels(
                test_raw, current_duration, current_overlap
            ).ravel()

            min_len_te = min(len(test_preds_raw), len(y_true))
            test_preds_raw = test_preds_raw[:min_len_te]
            y_true = y_true[:min_len_te]

            y_pred = np.array([label_mapping.get(state, 0) for state in test_preds_raw])

            plot_session_timeline(
                raw_session=test_raw,
                pipeline=fold_pipeline,
                duration=current_duration,
                overlap=current_overlap,
                channel_name="FC5",
                session_name=test_session,
            )

            # Obliczanie metryk
            acc = accuracy_score(y_true, y_pred)
            b_acc = balanced_accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_true, y_pred)

            loso_accuracy.append(acc)
            loso_balanced_acc.append(b_acc)
            loso_f1.append(f1)
            loso_mcc.append(mcc)

            # DODANIE WIERSZA DO SZCZEGÓŁOWYCH WYNIKÓW
            detailed_rows.append(
                {
                    "test_file": test_session,
                    "Model": model_name,
                    "acc": round(acc, 4),
                    "b_acc": round(b_acc, 4),
                    "f1": round(f1, 4),
                    "mcc": round(mcc, 4),
                }
            )

            print(
                f"Sesja testowa: {test_session} -> Balanced Acc: {b_acc:.4f} | F1: {f1:.4f} | MCC: {mcc:.4f}"
            )

        final_results[model_name] = {
            "acc_mean": np.mean(loso_accuracy),
            "b_acc_mean": np.mean(loso_balanced_acc),
            "f1_mean": np.mean(loso_f1),
            "mcc_mean": np.mean(loso_mcc),
        }

        print(f"\n Wynik końcowy {model_name}:")
        print(f" Mean Balanced Acc: {final_results[model_name]['b_acc_mean']:.4f}")
        print(f" Mean F1-Score:     {final_results[model_name]['f1_mean']:.4f}")
        print(f" Mean MCC:          {final_results[model_name]['mcc_mean']:.4f}")

    df_results = pd.DataFrame(detailed_rows)
    csv_path = "loso_cv_detailed_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n✅ Zapisano szczegółowe wyniki do pliku: {csv_path}")

    return final_results


def plot_session_timeline(
    raw_session, pipeline, duration, overlap, channel_name="C3", session_name="Session"
):
    """
    Rysuje sygnał EEG oraz porównanie Ground Truth vs Predykcje Modelu w czasie.
    """
    # 1. Pobieranie danych sygnału dla wybranego kanału
    if channel_name in raw_session.ch_names:
        ch_idx = raw_session.ch_names.index(channel_name)
    else:
        ch_idx = 0
        channel_name = raw_session.ch_names[0]

    data, times = raw_session[ch_idx, :]
    data = data.ravel() * 1e6  # Konwersja do mikrowoltów (uV) dla lepszej czytelności

    # 2. Generowanie predykcji i etykiet prawdziwych
    preds_raw = pipeline.predict(raw_session).ravel()
    y_true = get_ground_truth_labels(raw_session, duration, overlap).ravel()

    min_len = min(len(preds_raw), len(y_true))
    preds_raw = preds_raw[:min_len]
    y_true = y_true[:min_len]

    # [TUTAJ MAPOWANIE] - Wykorzystujemy logikę współczynnika ruchu do dopasowania klas
    unique_states = np.unique(preds_raw)
    state_ratios = {
        state: np.mean(y_true[preds_raw == state] == 1) for state in unique_states
    }
    sorted_states = sorted(
        state_ratios.keys(), key=lambda x: state_ratios[x], reverse=True
    )

    label_mapping = {sorted_states[0]: 1} if sorted_states else {}
    for state in sorted_states[1:]:
        label_mapping[state] = 0

    y_pred = np.array([label_mapping.get(s, 0) for s in preds_raw])

    # 3. Przeliczenie indeksów okien na sekundy osi czasu
    # Każde okno reprezentuje środek lub początek swojego czasu trwania
    step = duration - overlap
    window_times = np.arange(min_len) * step + (duration / 2.0)

    # 4. Tworzenie wykresu
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    # Panel 1: Sygnał EEG
    ax1.plot(
        times, data, color="#1f77b4", alpha=0.7, label=f"Sygnał EEG ({channel_name})"
    )
    ax1.set_title(
        f"Analiza osi czasu dla sesji: {session_name}", fontsize=14, fontweight="bold"
    )
    ax1.set_ylabel("Amplituda [µV]")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    # Panel 2: Stan Zaadnotowany vs Przewidziany
    # Używamy where='mid' dla ładnego wyrównania okien czasowych
    ax2.step(
        window_times,
        y_true,
        where="mid",
        label="Stan Zaadnotowany (Ground Truth)",
        color="#2ca02c",
        linewidth=2.5,
    )
    ax2.step(
        window_times,
        y_pred,
        where="mid",
        label="Predykcja Modelu",
        color="#d62728",
        linestyle="--",
        linewidth=2,
    )

    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["0 (Relaks)", "1 (Ruch)"])
    ax2.set_xlabel("Czas [sekundy]")
    ax2.set_ylabel("Stan")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    plt.tight_layout()

    # Zapis do pliku
    filename = f"timeline_{session_name}_{pipeline.__class__.__name__}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"📊 Wykres osi czasu zapisany jako: {filename}")
