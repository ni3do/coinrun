import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

for file_name in sorted(os.listdir("data/csv")):
    if not os.path.exists(f"data/plots/{file_name.split('.')[0]}"):
        os.makedirs("data/plots/" + file_name.split(".")[0])

    df = pd.read_csv("data/csv/" + file_name, sep=",")
    print(file_name)

    agg_interval = 50
    bins = len(df) // agg_interval
    # agg_df = df.groupby(np.arange(len(df)//2)).mean()
    # print(agg_df)
    for key in df.keys():
        # timestep,eplenmean,eprew,fps,total_timesteps,policy_loss,value_loss,policy_entropy,approxkl,clipfrac,l2_loss,test_score
        if key in ["Unnamed: 0", "timestep", "eplenmean", "eprw", "fps", "total_timesteps"]:
            continue

        plt.figure(figsize=(10, 6))
        ax = sns.regplot(x="timestep", y=key, data=df, marker="o", x_bins=bins)
        plt.savefig(
            f"data/plots/{file_name.split('.')[0]}/" + file_name.split(".")[0] + "-" + key + ".png",
            dpi=600,
        )
