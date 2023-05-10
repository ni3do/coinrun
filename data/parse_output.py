import os

import pandas as pd

df = pd.DataFrame(
    columns=[
        "timestep",
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
for file_name in os.listdir("data/out"):
    if (not file_name.endswith(".out")) or file_name in os.listdir("data/csv"):
        continue
    print(file_name)
    with open("log/" + file_name) as f:
        lines = f.readlines()

        timestep = [line.split(" ")[1] for line in lines if line.startswith("timestep")]
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
        test_scores = [
            sum(eval(line)) / max(len(eval(line)), 1) for line in lines if line.startswith("[")
        ]

    rows = []
    for i in range(len(timestep)):
        rows.append(
            {
                "timestep": timestep[i],
                "eplenmean": eplenmean[i],
                "eprew": eprew[i],
                "fps": fps[i],
                "total_timesteps": total_timesteps[i],
                "policy_loss": policy_loss[i],
                "value_loss": value_loss[i],
                "policy_entropy": policy_entropy[i],
                "approxkl": approxkl[i],
                "clipfrac": clipfrac[i],
                "l2_loss": l2_loss[i],
                "test_score": test_scores[i],
            }
        )

    df = pd.DataFrame(rows)
    df = df.apply(pd.to_numeric, errors="coerce")
    print(df)
    df.to_csv("data/csv/" + file_name.split(".")[0] + ".csv")
