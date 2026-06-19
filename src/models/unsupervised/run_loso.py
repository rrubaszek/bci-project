from email.policy import default
from src.models.unsupervised.loso import run_loso_evaluation
from src.models.unsupervised.factory import ModelTypes, ModelFactory
from src.preprocessing.unsupervised.extract.factory import (
    ExtractorFactory,
    ExtractorType,
)
import argparse
import pandas as pd


def parse_arguments():
    """Przykładowy parser - dostosuj do tego, co już masz w projekcie."""
    parser = argparse.ArgumentParser(
        description="Uruchomienie ewaluacji LOSO CV dla BCI"
    )
    parser.add_argument(
        "--gmm_window_duration",
        type=float,
        default=1.25,
        help="Długość okna w sekundach",
    )
    parser.add_argument(
        "--gmm_window_overlap",
        type=float,
        default=0,
        help="Nakładanie się okien w sekundach",
    )

    parser.add_argument(
        "--hmm_window_duration",
        type=float,
        default=1,
        help="Długość okna w sekundach",
    )
    parser.add_argument(
        "--hmm_window_overlap",
        type=float,
        default=0.5,
        help="Nakładanie się okien w sekundach",
    )
    parser.add_argument(
        "--gmm_pca_components",
        type=float,
        default=3,
        help="Procent komponentów PCA",
    )
    parser.add_argument(
        "--hmm_pca_components",
        type=float,
        default=0.95,
        help="Procent komponentów PCA",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    print("Inicjalizacja środowiska z argumentami:", args)

    # 2. Tworzenie ekstraktorów (Twój kod)
    psd_extractor = ExtractorFactory.create_extractor(
        ExtractorType.PSD,
        window_duration=args.hmm_window_duration,
        window_overlap=args.hmm_window_overlap,
    )

    ica_extractor = ExtractorFactory.create_extractor(
        ExtractorType.ICA,
        n_components=10,
        window_duration=args.gmm_window_duration,
        window_overlap=args.gmm_window_overlap,
    )

    # 3. Tworzenie modeli/pipeline'ów (Twój kod)
    models = {
        "GMM": ModelFactory.create_pipeline(
            model_type=ModelTypes.GMM,
            extractor=ica_extractor,
            n_clusters=2,
            pca_components=args.gmm_pca_components,
        ),
        "HMM": ModelFactory.create_pipeline(
            model_type=ModelTypes.HMM,
            extractor=psd_extractor,
            n_states=2,
            pca_components=args.hmm_pca_components,
        ),
    }

    results = run_loso_evaluation(models, args)
    df_results = pd.DataFrame(results).T

    df_results.index.name = "Model"

    df_results = df_results.round(4)

    output_csv_path = "loso_cv_results.csv"

    df_results.to_csv(output_csv_path)
    print(f"\n✅ Zapisano wyniki do pliku: {output_csv_path}")


if __name__ == "__main__":
    main()
