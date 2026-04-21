import mne
import numpy as np
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

def main():

    # === Load data ===
    file_path = "data/B0101T.gdf"
    raw = mne.io.read_raw_gdf(file_path, preload=True)

    # Keep EEG only
    raw.pick_types(eeg=True, eog=False)

    # Bandpass filter
    raw.filter(8., 30., fir_design='firwin')

    events, event_id = mne.events_from_annotations(raw)

    event_dict = {
        'left': event_id['769'],
        'right': event_id['770']
    }

    # === Epoching ===
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_dict,
        tmin=0.5,   # avoid visual response
        tmax=4.0,
        baseline=None,
        preload=True
    )

    # === Remove rejected trials (1023) ===
    # Find indices of rejected events
    reject_events = events[events[:, 2] == 1023]

    if len(reject_events) > 0:
        reject_times = reject_events[:, 0] / raw.info['sfreq']
        bad_trials = []

        for i, epoch in enumerate(epochs):
            start = epochs.events[i, 0] / raw.info['sfreq']
            end = start + (epochs.tmax - epochs.tmin)

            # check overlap
            for rt in reject_times:
                if start <= rt <= end:
                    bad_trials.append(i)
                    break

        epochs.drop(bad_trials)

    print("Remaining trials:", len(epochs))

    # === Prepare data ===
    X = epochs.get_data()
    y = epochs.events[:, -1]

    # Convert labels: 769->0, 770->1
    y = np.where(y == event_id['769'], 0, 1)

    print("Shape:", X.shape)

    # === CSP + LDA ===
    csp = CSP(n_components=4, log=True)
    lda = LinearDiscriminantAnalysis()

    clf = Pipeline([
        ('CSP', csp),
        ('LDA', lda)
    ])

    # === Evaluate ===
    scores = cross_val_score(clf, X, y, cv=5)

    print("Accuracy:", np.mean(scores))


if __name__ == "__main__":
    main()
