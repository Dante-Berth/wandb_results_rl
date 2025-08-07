import wandb
import pandas as pd
import os
from tqdm import tqdm


def download_wandb_history(
    entity: str,
    project: str,
    run_ids: list,
    output_dir: str = "wandb_csv_exports",
    name_discounted_episodic_return: str = "charts/discounted_cumulative_return",
    name_episodic_length: str = "charts/episodic_length",
):
    """
    Downloads selected run histories from Weights & Biases using scan_history(),
    saves relevant metrics to CSV files in the output directory.

    Parameters:
    - entity (str): WandB entity/team name.
    - project (str): WandB project name.
    - run_ids (list): List of WandB run IDs to download.
    - output_dir (str): Folder to save the resulting CSV files.
    """
    # Authenticate
    wandb.login()

    # Create output folder if not exists
    os.makedirs(output_dir, exist_ok=True)

    for run_id in run_ids:
        print(f"\nProcessing run: {run_id}")
        try:
            run = wandb.Api().run(f"{entity}/{project}/{run_id}")
            history_iter = run.scan_history()

            records = []
            for row in tqdm(history_iter, desc=f"Downloading {run_id}", unit="step"):
                record = {
                    "_step": row.get("_step"),
                    "discounted_cumulative_return": row.get(
                        name_discounted_episodic_return
                    ),
                    "episodic_length": row.get(name_episodic_length),
                }
                records.append(record)

            df = pd.DataFrame(records)
            df = df.dropna(
                subset=["discounted_cumulative_return", "episodic_length"]
            ).copy()
            df = df.sort_values("_step")
            df["steps"] = df["episodic_length"].cumsum()

            out_path = os.path.join(output_dir, f"{run_id}.csv")
            df.to_csv(out_path, index=False)
            print(f"✅ Saved: {out_path}")

        except Exception as e:
            print(f"❌ Error with run {run_id}: {e}")


if __name__ == "__main__":
    download_wandb_history(
        entity="corporate-manu-sureli",
        project="SAC_IMAGE_TIB",
        run_ids=[
            "fgt1rt8r",
            "trwfkvmy",
            "33f01rmn",
            "lhwk49b4",
            "phn6kqx3",
            "sjm05wn4",
            "26r063cr",
        ],
        output_dir="wandb_csv_exports",
    )
