import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

import plotly.graph_objects as go
import plotly.express as px


# -----------------------------
# MAIN ENTRY
# -----------------------------
def create_all_visualizations(
    predictions: dict,
    raw_data_list,
    features,
    output_dir: Path
):
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names = list(predictions.keys())

    # -------------------------
    # 1. STATE TIMELINE (MATPLOTLIB)
    # -------------------------
    # for model_name, preds in predictions.items():
    #     _plot_state_timeline_matplotlib(preds, model_name, output_dir)
    #     _plot_state_timeline_plotly(preds, model_name, output_dir)

    # -------------------------
    # 2. MODEL COMPARISON TIMELINE
    # -------------------------
    if len(model_names) >= 2:
        _plot_model_comparison_matplotlib(predictions, output_dir)
        _plot_model_comparison_plotly(predictions, output_dir)

    # -------------------------
    # 3. PCA EMBEDDING (IF AVAILABLE)
    # -------------------------
    try:
        _plot_pca_matplotlib(features, predictions, output_dir)
        _plot_pca_plotly(features, predictions, output_dir)
    except Exception as e:
        print(f"PCA visualization skipped: {e}")

    # # -------------------------
    # # 4. PSD PER STATE (IF FEATURES STRUCTURED)
    # # -------------------------
    # try:
    #     _plot_psd_states_matplotlib(features, predictions, output_dir)
    #     _plot_psd_states_plotly(features, predictions, output_dir)
    # except Exception as e:
    #     print(f"PSD visualization skipped: {e}")

    # -------------------------
    # 5. HMM TRANSITIONS
    # -------------------------
    if "HMM" in predictions:
        _plot_transition_matrix(predictions["HMM"], output_dir)

    # -------------------------
    # 6. DWELL TIME
    # -------------------------
    for model_name, preds in predictions.items():
        _plot_dwell_time(preds, model_name, output_dir)


# =========================================================
# 1. STATE TIMELINE
# =========================================================
def _plot_state_timeline_matplotlib(preds, model_name, out):
    plt.figure(figsize=(12, 3))
    plt.plot(preds, linewidth=0.8)
    plt.title(f"{model_name} - State timeline (Matplotlib)")
    plt.xlabel("Time window")
    plt.ylabel("State")
    plt.yticks(sorted(set(preds)))
    plt.tight_layout()
    plt.savefig(out / f"{model_name}_timeline.png")
    plt.close()


def _plot_state_timeline_plotly(preds, model_name, out):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=preds, mode="lines", name="state"))
    fig.update_layout(
        title=f"{model_name} - State timeline (Plotly)",
        xaxis_title="Time window",
        yaxis_title="State"
    )
    fig.write_html(out / f"{model_name}_timeline.html")


# =========================================================
# 2. MODEL COMPARISON
# =========================================================
def _plot_model_comparison_matplotlib(predictions, out):
    plt.figure(figsize=(12, 4))

    for name, preds in predictions.items():
        plt.plot(preds, label=name, alpha=0.7)

    plt.title("Model comparison - state sequences")
    plt.xlabel("Time")
    plt.ylabel("State")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "model_comparison.png")
    plt.close()


def _plot_model_comparison_plotly(predictions, out):
    fig = go.Figure()

    for name, preds in predictions.items():
        fig.add_trace(go.Scatter(y=preds, mode="lines", name=name))

    fig.update_layout(title="Model comparison")
    fig.write_html(out / "model_comparison.html")


# =========================================================
# 3. PCA VISUALIZATION
# =========================================================
def _plot_pca_matplotlib(features, predictions, out):
    if features is None:
        return

    X = np.concatenate(features, axis=0)

    plt.figure(figsize=(6, 5))

    for model_name, preds in predictions.items():
        plt.scatter(X[:, 0], X[:, 1], c=preds, cmap="viridis", s=5, alpha=0.5)
        plt.title(f"PCA space - {model_name}")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        plt.savefig(out / f"{model_name}_pca.png")
        plt.clf()


def _plot_pca_plotly(features, predictions, out):
    if features is None:
        return

    X = np.concatenate(features, axis=0)

    for model_name, preds in predictions.items():
        fig = px.scatter(
            x=X[:, 0],
            y=X[:, 1],
            color=preds,
            title=f"PCA embedding - {model_name}"
        )
        fig.write_html(out / f"{model_name}_pca.html")


# =========================================================
# 4. PSD PER STATE
# =========================================================
def _plot_psd_states_matplotlib(features, predictions, out):
    # simplified: assumes PSD features already flattened
    X = np.concatenate(features, axis=0)

    for model_name, preds in predictions.items():
        plt.figure(figsize=(8, 4))

        for state in np.unique(preds):
            mean_spec = X[preds == state].mean(axis=0)
            plt.plot(mean_spec, label=f"State {state}")

        plt.title(f"PSD per state - {model_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / f"{model_name}_psd_states.png")
        plt.close()


def _plot_psd_states_plotly(features, predictions, out):
    X = np.concatenate(features, axis=0)

    for model_name, preds in predictions.items():
        fig = go.Figure()

        for state in np.unique(preds):
            mean_spec = X[preds == state].mean(axis=0)
            fig.add_trace(go.Scatter(y=mean_spec, name=f"State {state}"))

        fig.update_layout(title=f"PSD per state - {model_name}")
        fig.write_html(out / f"{model_name}_psd_states.html")


# =========================================================
# 5. TRANSITION MATRIX (HMM)
# =========================================================
def _plot_transition_matrix(preds, out):
    states = np.unique(preds)
    idx = {s: i for i, s in enumerate(states)}

    mat = np.zeros((len(states), len(states)))

    for i in range(len(preds) - 1):
        mat[idx[preds[i]], idx[preds[i+1]]] += 1

    mat = mat / (mat.sum(axis=1, keepdims=True) + 1e-8)

    plt.figure(figsize=(5, 4))
    plt.imshow(mat, cmap="Blues")
    plt.title("HMM transition matrix")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out / "hmm_transition_matrix.png")
    plt.close()


# =========================================================
# 6. DWELL TIME
# =========================================================
def _plot_dwell_time(preds, model_name, out):
    durations = []
    current = preds[0]
    length = 1

    for p in preds[1:]:
        if p == current:
            length += 1
        else:
            durations.append(length)
            current = p
            length = 1

    durations.append(length)

    plt.figure(figsize=(6, 4))
    plt.hist(durations, bins=20)
    plt.title(f"{model_name} - dwell time")
    plt.xlabel("Duration")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out / f"{model_name}_dwell_time.png")
    plt.close()