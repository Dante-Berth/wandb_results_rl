import wandb
import pandas as pd
import numpy as np
import plotly.graph_objects as go

api = wandb.Api()

# Replace with your project and entity
wandb_project_name = "SAC_IMAGE_SIMPLE_PHYSIGYM"
wandb_entity = "corporate-manu-sureli"

# Fetch all runs
runs = api.runs(f"{wandb_entity}/{wandb_project_name}")

key = "charts/episodic_return"
all_data = {}

for run in runs:
    seed = run.config.get("seed", None)  # Get the seed if available
    print(f"Processing seed: {seed}")

    # Fetch full history of episodic returns
    history = run.history(keys=[key, "_step"])

    if history is not None and not history.empty:
        history = history[history["_step"] <= int(6e5) + 5000]
        history["seed"] = seed  # Tag with seed
        all_data[f"seed_{seed}"] = history

# Combine all runs into a single DataFrame
if all_data:
    df = pd.concat(all_data, ignore_index=False)
    reshaped_df = df.pivot(
        index="_step", columns="seed", values="charts/episodic_return"
    )

    reshaped_df.columns = [f"seed_{col}" for col in reshaped_df.columns]
    reshaped_df = reshaped_df.interpolate(method="linear", axis=0)

    # Compute the mean, min, and max of the first 4 columns (seeds)
    reshaped_df["mean"] = reshaped_df[["seed_1", "seed_2", "seed_3", "seed_4"]].mean(
        axis=1
    )
    reshaped_df["min"] = reshaped_df[["seed_1", "seed_2", "seed_3", "seed_4"]].min(
        axis=1
    )
    reshaped_df["max"] = reshaped_df[["seed_1", "seed_2", "seed_3", "seed_4"]].max(
        axis=1
    )

    fig = go.Figure()

    # Add min-max shaded region
    fig.add_trace(
        go.Scatter(
            x=reshaped_df.index,
            y=reshaped_df["max"],
            fill=None,
            line=dict(color="rgba(0,0,255,0)"),  # Transparent line
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=reshaped_df.index,
            y=reshaped_df["min"],
            fill="tonexty",  # Fill area between min and max
            fillcolor="rgba(0,0,255,0.2)",  # Light blue shading
            line=dict(color="rgba(0,0,255,0)"),  # Transparent line
            showlegend=False,
        )
    )

    # Add mean curve (darker line)
    fig.add_trace(
        go.Scatter(
            x=reshaped_df.index,
            y=reshaped_df["mean"],
            line=dict(color="blue", width=2),  # Darker blue line
            name="Mean Episodic Return",
        )
    )

    # Layout settings with custom x-axis ticks and labels
    fig.update_layout(
        xaxis_title="Million Steps",
        yaxis_title="Average Return",
        xaxis=dict(
            tickvals=[
                i * int(1e5) for i in range(7)
            ],  # Values from 0.0M to 10M (adjust if needed)
            ticktext=[f"{i / 10}" for i in range(7)],  # Labels from 0.0M to 1.0M
            tickfont=dict(size=18),  # Increase tick label font size
            title_font=dict(size=18),  # Increase axis title font size
        ),
        yaxis=dict(
            tickfont=dict(size=18),  # Increase tick label font size
            title_font=dict(size=18),  # Increase axis title font size
        ),
        
        template="plotly_white",
    )

    fig.show()

else:
    print("No data found for the given key.")
