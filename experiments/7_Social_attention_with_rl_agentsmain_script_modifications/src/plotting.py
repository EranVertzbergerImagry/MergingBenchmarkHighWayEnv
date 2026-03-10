"""Training plot generation from episodes.csv."""
import os
import csv

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def generate_training_plot(episodes_csv, plot_path, description="", window=50):
    """Read episodes.csv and generate an interactive Plotly HTML chart."""
    if not HAS_PLOTLY:
        print("  (plotly not installed — skipping training plot)")
        return

    episodes, rewards, lengths = [], [], []
    try:
        with open(episodes_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                episodes.append(int(row["episode"]))
                rewards.append(float(row["reward"]))
                lengths.append(int(row["length"]))
    except (OSError, KeyError):
        print("  (could not read episodes.csv — skipping training plot)")
        return

    if len(episodes) < 2:
        return

    episodes = np.array(episodes)
    rewards = np.array(rewards)
    lengths = np.array(lengths)

    def rolling_mean(data, w):
        if len(data) < w:
            return np.arange(1, len(data) + 1), data
        cs = np.cumsum(data)
        cs = np.insert(cs, 0, 0.0)
        means = (cs[w:] - cs[:-w]) / w
        x = np.arange(w, len(data) + 1)
        return x, means

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=False,
        subplot_titles=("Episode Reward", "Episode Length"),
        vertical_spacing=0.12,
    )

    # --- Episode Reward ---
    fig.add_trace(go.Scatter(
        x=episodes, y=rewards, mode="lines",
        name="Reward", opacity=0.3, line=dict(color="royalblue"),
        hovertemplate="Ep %{x}<br>Reward: %{y:.2f}<extra></extra>",
    ), row=1, col=1)
    rx, rm = rolling_mean(rewards, window)
    fig.add_trace(go.Scatter(
        x=rx, y=rm, mode="lines",
        name=f"Reward (mean {window})", line=dict(color="royalblue", width=2),
        hovertemplate="Ep %{x}<br>Mean: %{y:.2f}<extra></extra>",
    ), row=1, col=1)

    # --- Episode Length ---
    fig.add_trace(go.Scatter(
        x=episodes, y=lengths, mode="lines",
        name="Length", opacity=0.3, line=dict(color="darkorange"),
        hovertemplate="Ep %{x}<br>Length: %{y}<extra></extra>",
    ), row=2, col=1)
    rx, rm = rolling_mean(lengths, window)
    fig.add_trace(go.Scatter(
        x=rx, y=rm, mode="lines",
        name=f"Length (mean {window})", line=dict(color="darkorange", width=2),
        hovertemplate="Ep %{x}<br>Mean: %{y:.1f}<extra></extra>",
    ), row=2, col=1)

    fig.update_xaxes(title_text="Episode", row=1, col=1)
    fig.update_xaxes(title_text="Episode", row=2, col=1)
    fig.update_yaxes(title_text="Reward", row=1, col=1)
    fig.update_yaxes(title_text="Length", row=2, col=1)

    fig.update_layout(
        title=f"rl-agents EgoAttention DQN — {description}",
        height=700, width=1000,
        hovermode="x unified",
    )

    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    with open(plot_path, "w") as f:
        f.write(fig.to_html())
    print(f"  Training plot saved to {plot_path}")
