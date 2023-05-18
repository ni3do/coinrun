import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

SCATTER = False
ORDER = 3

for file_name in sorted(os.listdir("csv")):
    if not os.path.exists(f"plots/{file_name.split('.')[0]}"):
        os.makedirs("plots/" + file_name.split(".")[0])

    df = pd.read_csv("csv/" + file_name, sep=",")
    print(file_name)

    if file_name.endswith("_test.csv"):
        # agg_interval = 1
        bins = len(df)
    else:
        # agg_interval = 50
        bins = 50
        
    # bins = len(df) // agg_interval
    # agg_df = df.groupby(np.arange(len(df)//2)).mean()
    # print(agg_df)
    for key in df.keys():
        if os.path.exists(
            f"plots/{file_name.split('.')[0]}/" + file_name.split(".")[0] + "-" + key + ".png"
        ):
            continue
        # timestep,eplenmean,eprew,fps,total_timesteps,policy_loss,value_loss,policy_entropy,approxkl,clipfrac,l2_loss,test_score
        if key in ["Unnamed: 0", "timestep", "eplenmean", "eprw", "fps", "total_timesteps"]:
            continue

        plt.figure(figsize=(10, 6))
        # ax = sns.regplot(x="total_timesteps", y=key, data=df, x_bins=bins, marker="o", order=3)
        ax = sns.regplot(x="total_timesteps", y=key, data=df, x_bins=bins, scatter=SCATTER, order=ORDER, marker="o")
        # ax = sns.lineplot(x="total_timesteps", y=key, data=df, marker="o")
        plt.savefig(
            f"plots/{file_name.split('.')[0]}/" + file_name.split(".")[0] + "-" + key + ".png",
            dpi=600,
        )
