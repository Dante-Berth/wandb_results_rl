import os
import re
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.style.use("tableau-colorblind10")


def parse_folder_name(folder_name):
    """
    Returns (seed:int, obs_type:str) or (None, None) if parsing fails
    """
    pattern = r"async_sac_tip_(\d+)_([a-zA-Z0-9_]+)_\d+"
    match = re.match(pattern, folder_name)
    if not match:
        return None, None
    seed = int(match.group(1))
    obs_type = match.group(2)
    return seed, obs_type


csv_root = "./wandb_csv_exports"


def load_all_runs(csv_root):
    """
    Returns:
        dict[obs_type] -> list of DataFrames (one per seed)
    """
    groups = defaultdict(list)

    for folder in os.listdir(csv_root):
        folder_path = os.path.join(csv_root, folder)
        if not os.path.isdir(folder_path):
            continue

        seed, obs_type = parse_folder_name(folder)
        if seed is None:
            continue

        csv_path = os.path.join(folder_path, "history.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        df = df[["step", "return"]].copy()
        df["seed"] = seed

        groups[obs_type].append(df)

    return groups


groups = load_all_runs(csv_root)


def prepare_group_data_rolling_then_mean(all_data, window=100):
    """
    all_data: list of DataFrames with columns [step, return, seed]
    """
    if not all_data:
        return None

    df = pd.concat(all_data, ignore_index=True)

    # Pivot: step × seed
    reshaped = df.pivot(index="step", columns="seed", values="return")
    reshaped = reshaped.sort_index()
    reshaped = reshaped.interpolate(method="linear", axis=0)

    # Rolling mean per seed
    smoothed = reshaped.copy()
    for seed in smoothed.columns:
        smoothed[seed] = (
            smoothed[seed].rolling(window=window, center=True, min_periods=1).mean()
        )

    # Aggregate across seeds
    out = pd.DataFrame(index=smoothed.index)
    out["mean"] = smoothed.mean(axis=1)
    out["min"] = smoothed.min(axis=1)
    out["max"] = smoothed.max(axis=1)
    out["std"] = smoothed.std(axis=1)

    return out


aggregated = {}

for obs_type, dfs in groups.items():
    aggregated[obs_type] = prepare_group_data_rolling_then_mean(dfs, window=100)


fig, (ax_main, ax_zoom) = plt.subplots(
    ncols=2,
    figsize=(14, 6),
    gridspec_kw={"width_ratios": [3, 1]},  # 75% / 25%
)

# ---- main plot (full range) ----
for obs_type, df in aggregated.items():
    if df is None:
        continue

    ax_main.plot(df.index, df["mean"], label=obs_type)
    ax_main.fill_between(
        df.index,
        df["mean"] - df["std"],
        df["mean"] + df["std"],
        alpha=0.2,
    )

ax_main.set_xlabel("Step")
ax_main.set_ylabel("Return")
ax_main.set_title("Performance for different state spaces")
ax_main.legend()
ax_main.grid(True)

# ---- zoom plot (restricted range) ----
for obs_type, df in aggregated.items():
    if df is None:
        continue

    ax_zoom.plot(df.index, df["mean"])
    ax_zoom.fill_between(
        df.index,
        df["mean"] - df["std"],
        df["mean"] + df["std"],
        alpha=0.2,
    )

# Zoom limits
ax_zoom.set_xlim(1.75e5, 2e5)
ax_zoom.set_ylim(160, 185)

ax_zoom.set_title("Zoom")
ax_zoom.grid(True)

for ax in [ax_main, ax_zoom]:
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    # Optional: Ensure the offset text (the 'x10^5') is visible and formatted nicely
    ax.xaxis.get_offset_text().set_fontsize(10)

fig.savefig("performance.pdf", format="pdf", bbox_inches="tight")

plt.tight_layout()
plt.close()
