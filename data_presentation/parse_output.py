import os

import pandas as pd

df = pd.DataFrame(
    columns=[
        # "timestep",
        "eplenmean",
        "eprewmean",
        "fps",
        "total_timesteps",
        "policy_loss",
        "value_loss",
        "policy_entropy",
        "approxkl",
        "clipfrac",
        "l2_loss",
        "test_scores",
    ]
)
for file_name in os.listdir("out"):
    print(file_name[:-3] + "csv")
    if (not file_name.endswith(".out")) or file_name[:-3] + "csv" in os.listdir("csv"):
        continue
    print(file_name)
    with open("out/" + file_name) as f:
        lines = f.readlines()

        # timestep = [line.split(" ")[1] for line in lines if line.startswith("timestep")]
        eplenmean = [line.split(" ")[1] for line in lines if line.startswith("eplenmean")]
        eprew = [line.split(" ")[1] for line in lines if line.startswith("eprew")]
        fps = [line.split(" ")[1] for line in lines if line.startswith("fps")]
        total_timesteps = [
            line.split(" ")[1] for line in lines if line.startswith("total_timesteps")
        ]
        policy_loss = [line.split(" ")[1] for line in lines if line.startswith("policy_loss")]
        value_loss = [line.split(" ")[1] for line in lines if line.startswith("value_loss")]
        policy_entropy = [line.split(" ")[1] for line in lines if line.startswith("policy_entropy")]
        approxkl = [line.split(" ")[1] for line in lines if line.startswith("approxkl")]
        clipfrac = [line.split(" ")[1] for line in lines if line.startswith("clipfrac")]
        l2_loss = [line.split(" ")[1] for line in lines if line.startswith("l2_loss")]
        train_scores = [
            sum(eval(line)) / max(len(eval(line)), 1) for line in lines if line.startswith("[")
        ]
        test_score = [line.split(" ")[1] for line in lines if line.startswith("mean_score")]

    total_timesteps = [int(t) for t in total_timesteps]
    rows = []
    multiplier = 0
    interval = 5e5

    for i in range(len(total_timesteps)):
        if (i > 0 and total_timesteps[i] < total_timesteps[i - 1]):
            print(total_timesteps[i], total_timesteps[i - 1])
            multiplier += 1
        rows.append(
            {
                # "timestep": timestep[i],
                "eplenmean": eplenmean[i],
                "eprew": eprew[i],
                "fps": fps[i],
                "total_timesteps": total_timesteps[i] + multiplier * interval,
                "policy_loss": policy_loss[i],
                "value_loss": value_loss[i],
                "policy_entropy": policy_entropy[i],
                "approxkl": approxkl[i],
                "clipfrac": clipfrac[i],
                "l2_loss": l2_loss[i],
                "train_score": train_scores[i],
            }
        )

    df = pd.DataFrame(rows)
    df = df.apply(pd.to_numeric, errors="coerce")
    print(df)
    df.to_csv("csv/" + file_name.split(".")[0] + ".csv")

    rows_test = []
    interval = 5e5
    for i in range(len(test_score)):
        rows_test.append({"total_timesteps": i * interval, "test_score": test_score[i]})
    df = pd.DataFrame(rows_test)
    df = df.apply(pd.to_numeric, errors="coerce")
    print(df)
    df.to_csv("csv/" + file_name.split(".")[0] + "_test.csv")