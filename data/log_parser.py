import os

import pandas as pd


def gather_runs(csv_file_name="grid_stats", run_type="grid"):
    df = pd.read_csv(f"csv/{csv_file_name}.csv")
    for file_name in sorted(os.listdir(f"out/{run_type}")):
        if (not file_name.endswith(".out")) or file_name[:-3] + "csv" in os.listdir(
            "csv"
        ):
            continue
        print(file_name)
        with open(f"out/{run_type}/{file_name}") as f:
            lines = f.readlines()

            hyperparameters = [
                line.split(" ")[1:]
                for line in lines
                if line.startswith("hyperparameters")
            ]
            mean_scores = [
                float(line.split(" ")[1])
                for line in lines
                if line.startswith("mean_score")
            ]

        dropout = round(float(hyperparameters[0][0]), 16)
        l2 = round(float(hyperparameters[0][1]), 16)
        epsilon = round(float(hyperparameters[0][2]), 16)

        print(l2, dropout, epsilon)

        if (
            len(
                df.loc[
                    (df["l2"] == l2)
                    & (df["dropout"] == dropout)
                    & (df["epsilon"] == epsilon)
                ]
            )
            == 0
        ):
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "type": run_type,
                            "dropout": dropout,
                            "l2": l2,
                            "epsilon": epsilon,
                            "test_score": [mean_scores],
                        }
                    ),
                ]
            )
    df.to_csv(f"csv/{csv_file_name}.csv", index=False)
    df.to_csv(f"../coinrun/{csv_file_name}.csv", index=False)


gather_runs()
