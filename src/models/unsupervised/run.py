"""Entry point for running preprocessing and starting unsupervised model pipelines."""

from curses import raw

import mne
import numpy as np
import argparse

from src.preprocessing.unsupervised.extract.factory import ExtractorFactory, ExtractorType
from src.preprocessing.unsupervised.load import load_raw_data
from src.preprocessing.unsupervised.filter import preprocess_eeg
from src.models.unsupervised.factory import ModelFactory, ModelTypes
from src.models.unsupervised.visualize import create_all_visualizations
from src.paths import DEFAULT_OUT_DIR


def main(args):
    print("Ładowanie danych EEG...")
    raw_data: list[mne.io.Raw] = load_raw_data()
    
    if not raw_data:
        print("Nie wczytano danych. Sprawdź ścieżkę do plików.")
        return
    
    
    print("\nPrzetwarzanie wstępne danych...")
    preprocessed_data: list[mne.io.Raw] = []
    for i, r in enumerate(raw_data):
        processed = preprocess_eeg(r)  # Filtracja i resampling danych EEG
        if processed is not None:
            preprocessed_data.append(processed)
    
    if not preprocessed_data:
        print("Nie udało się przetworzyć żadnych plików.")
        return
    
    print(f"Przetworzono {len(preprocessed_data)} plików")

    print("\nInicjalizacja modeli...")
    factory = ModelFactory()
    
    psd_extractor = ExtractorFactory.create_extractor(
        ExtractorType.PSD, 
        window_duration=args.window_duration,
        window_overlap=args.window_overlap
    )
    
    ica_extractor = ExtractorFactory.create_extractor(
        ExtractorType.ICA,
        n_components=7,
        window_duration=args.window_duration,
        window_overlap=args.window_overlap
    )
    
    models = {
        "GMM": factory.create_pipeline(
            model_type=ModelTypes.GMM,
            extractor=ica_extractor,
            pca_components=args.gmm_pca,
            n_clusters=args.gmm_clusters
        ),
        "HMM": factory.create_pipeline(
            model_type=ModelTypes.HMM,
            extractor=psd_extractor,
            pca_components=args.hmm_pca,
            n_states=args.hmm_states
        ),
    }
    print(f"Inicjalizowano {len(models)} modele")
    
    print("\nTrenowanie modeli...")
    results = {model_name: {} for model_name in models.keys()}
    
    for model_name, pipeline in models.items():
        print(f"\n  Trenowanie {model_name}...")
        pipeline.fit(preprocessed_data)
        predictions = pipeline.predict(preprocessed_data)
        results[model_name] = predictions
        
        unique_predictions = len(set(predictions))
        print(f"    Trenowanie zakończone")
        print(f"    Liczba predykcji: {len(predictions)}")
        print(f"    Unikalne wartości: {unique_predictions}")
        print(f"    Rozkład: {dict(zip(*np.unique(predictions, return_counts=True)))}")
    
    for model_name, preds in results.items():
        unique, counts = np.unique(preds, return_counts=True)
        print(f"\n{model_name}:")
        for u, c in zip(unique, counts):
            print(f"  Klaster/Stan {int(u)}: {c} próbek ({c/len(preds)*100:.1f}%)")
    
    # Visualizations
    if args.visualize:
        
        try:
            # Prepare features for visualization
            all_features_psd = []
            all_features_ica = []

            for r in raw_data:
                all_features_psd.append(psd_extractor.extract(r))
                all_features_ica.append(ica_extractor.extract(r))

            features_psd = np.concatenate(all_features_psd, axis=0)
            features_ica = np.concatenate(all_features_ica, axis=0)

            create_all_visualizations(
                predictions=results,
                raw_data_list=preprocessed_data,
                features=features_psd,
                output_dir=DEFAULT_OUT_DIR / "figures"
            )
        except Exception as e:
            print(f"\nBłąd podczas tworzenia wizualizacji: {e}")


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Unsupervised EEG analysis with GMM and HMM models"
    )
    
    # Data parameters
    parser.add_argument("--window-duration", type=float, default=0.1,
                       help="Window duration for feature extraction (seconds)")
    parser.add_argument("--window-overlap", type=float, default=0.05,
                       help="Window overlap for feature extraction (seconds)")
    
    # GMM parameters
    parser.add_argument("--gmm-clusters", type=int, default=2,
                       help="Number of Gaussian clusters for GMM")
    parser.add_argument("--gmm-pca", type=int, default=3,
                       help="Number of PCA components for GMM")
    
    # HMM parameters
    parser.add_argument("--hmm-states", type=int, default=2,
                       help="Number of hidden states for HMM")
    parser.add_argument("--hmm-pca", type=int, default=1,
                       help="Number of PCA components for HMM")
    
    # Visualization
    parser.add_argument("--visualize", action="store_true", default=False,
                       help="Create visualizations of predictions")
    parser.add_argument("--no-visualize", action="store_false", dest="visualize",
                       help="Skip visualization")
    
    return parser.parse_args()


def cli_entry_point():
    args = parse_args()
    main(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
