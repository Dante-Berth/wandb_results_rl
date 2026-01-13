import re
import numpy as np
import os
import pandas as pd

OUTPUT_ROOT = "./merged_by_env"
ROOT = "./wandb_csv_exports"


def extract_mean_drugs(value):
    """
    Handles:
    - float
    - string like "{'drug_1': array([0.0595561], dtype=float32)}"
    """
    if isinstance(value, (float, int)):
        return float(value)

    if isinstance(value, str):
        # extract first float inside brackets
        match = re.search(r"\[\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\]", value)
        if match:
            return float(match.group(1))

    return np.nan


def parse_folder_name(folder_name):
    pattern = r"async_sac_tip_(\d+)_([a-zA-Z0-9_]+)_\d+"
    match = re.match(pattern, folder_name)
    if not match:
        return None, None
    return int(match.group(1)), match.group(2)


def process_episode(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    if df.empty or "step" not in df.columns:
        return None

    # --- fix mean_drugs ---
    if "mean_drugs" in df.columns:
        df["mean_drugs"] = df["mean_drugs"].apply(extract_mean_drugs)
    elif "drug_1" in df.columns:
        df["mean_drugs"] = df["drug_1"].apply(extract_mean_drugs)
        df.drop(labels="drug_1", axis=False)
        print("drop")
    else:
        df["mean_drugs"] = np.nan
    if "cumulative_drugs" not in df.columns:
        # --- add metadata ---
        df["cumulative_drugs"] = df["mean_drugs"].cumsum()
        df.to_csv("data.csv", index=False)
        print("save")
    return None


tasks = []

for run_folder in os.listdir(ROOT):
    run_path = os.path.join(ROOT, run_folder)
    if not os.path.isdir(run_path):
        continue

    seed, state_space = parse_folder_name(run_folder)
    if seed is None:
        continue

    for env_name in os.listdir(run_path):
        if not env_name.startswith("env"):
            continue

        env_id = int(env_name.replace("env", ""))
        env_path = os.path.join(run_path, env_name)

        for episode_name in os.listdir(env_path):
            if not episode_name.startswith("episode"):
                continue

            episode_id = int(episode_name.replace("episode", ""))
            csv_path = os.path.join(env_path, episode_name, "data.csv")

            if not os.path.exists(csv_path):
                continue
            process_episode(csv_path)
            tasks.append((seed, state_space, env_id, episode_id, csv_path))
exit()
dictionnary = {}
for task in tasks:
    if f"{seed}_{env_id}_{episode_id}" not in dictionnary.keys():
        dictionnary[f"{seed}_{env_id}_{episode_id}"] = [csv_path]
    else:
        dictionnary[f"{seed}_{env_id}_{episode_id}"].append(csv_path)
METRICS = [
    "reward",
    "mean_drugs",
    "r_cancer_cells",
    "number_tumor",
    "number_cell_1",
    "number_cell_2",
]
for key in dictionnary.keys():
    list_df = []
    for csv_path in dictionnary[key]:
        list_df.append(pd.read_csv(csv_path))
