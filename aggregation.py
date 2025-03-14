import wandb
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Initialize the Weights & Biases API
api = wandb.Api()

# Replace with your specific project and entity details
wandb_project_name = "SAC_IMAGE_SIMPLE_PHYSIGYM"
wandb_entity = "corporate-manu-sureli"

# Fetch all runs from the specified project
runs = api.runs(f"{wandb_entity}/{wandb_project_name}")

# Key corresponding to episodic return in W&B logs
key = "charts/episodic_return"
all_data = {}

# Iterate through each run to collect relevant data
for run in runs:
    seed = run.config.get("seed", None)  # Extract seed value if available
    print(f"Processing seed: {seed}")

    # Retrieve the full training history for the episodic return metric
    history = run.history(keys=[key, "_step"])

    # Filter and process data if available
    if history is not None and not history.empty:
        history = history[history["_step"] <= int(6e5) + 5000]  # Limit data to 600k steps + buffer
        history["seed"] = seed  # Tag each entry with its seed value
        all_data[f"seed_{seed}"] = history

# Combine all individual run data into a single DataFrame
if all_data:
    df = pd.concat(all_data, ignore_index=False)
    
    # Reshape data for easier visualization
    reshaped_df = df.pivot(index="_step", columns="seed", values="charts/episodic_return")
    reshaped_df.columns = [f"seed_{col}" for col in reshaped_df.columns]
    reshaped_df = reshaped_df.interpolate(method="linear", axis=0)  # Fill missing values

    # Compute statistical measures (mean, min, max) across first 4 seeds
    reshaped_df["mean"] = reshaped_df[["seed_1", "seed_2", "seed_3", "seed_4"]].mean(axis=1)
    reshaped_df["min"] = reshaped_df[["seed_1", "seed_2", "seed_3", "seed_4"]].min(axis=1)
    reshaped_df["max"] = reshaped_df[["seed_1", "seed_2", "seed_3", "seed_4"]].max(axis=1)

    # Create a Plotly figure
    fig = go.Figure()

    # Add shaded region for min-max range
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
            fill="tonexty",  # Fill between min and max
            fillcolor="rgba(0,0,255,0.2)",  # Light blue shading
            line=dict(color="rgba(0,0,255,0)"),  # Transparent line
            showlegend=False,
        )
    )

    # Add mean episodic return curve
    fig.add_trace(
        go.Scatter(
            x=reshaped_df.index,
            y=reshaped_df["mean"],
            line=dict(color="blue", width=2),  # Darker blue line
            name="Mean Episodic Return",
        )
    )

    # Customize layout settings
    fig.update_layout(
        xaxis_title="Million Steps",
        yaxis_title="Average Return",
        xaxis=dict(
            tickvals=[i * int(1e5) for i in range(7)],  # Tick positions (0M to 600k steps)
            ticktext=[f"{i / 10}" for i in range(7)],  # Labels in millions
            tickfont=dict(size=18),
            title_font=dict(size=18),
        ),
        yaxis=dict(
            tickfont=dict(size=18),
            title_font=dict(size=18),
        ),
        template="plotly_white",
    )

    # Display the figure
    fig.show()

else:
    print("No data found for the given key.")