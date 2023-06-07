import os

import optuna
import pandas as pd
from optuna.samplers import RandomSampler


def create_bash_script(l2, dropout, epsilon):
    dp_line = f"dp={dropout}\n"
    l_two_line = f"l_two={l2}\n"
    epsilon_line = f"epsilon={epsilon}\n"
    model_name_line = f'model_name="reg-$dp-$l_two-$epsilon"\n'

    with open("scripts/base_0.sh", "r") as f:
        base_0 = f.read()
    with open("scripts/base_1.sh", "r") as f:
        base_1 = f.read()

    with open(
        f"scripts/auto/{str(dropout).split('.')[1]}_{str(l2).split('.')[1]}_{str(epsilon).split('.')[1]}.sh",
        "w",
    ) as f:
        f.write(base_0)
        f.write("\n")
        f.write(dp_line)
        f.write(l_two_line)
        f.write(epsilon_line)
        f.write(model_name_line)
        print("\n")
        f.write(base_1)


# def get_trials():
#     trials = []

#     df = pd.read_csv("model_stats.csv")
#     for index, row in df.iterrows():
#         trials.append(
#             optuna.trial.create_trial(
#                 params={
#                     "dropout": row["dropout"],
#                     "l2": row["l2"],
#                     "epsilon": row["epsilon"],
#                 },
#                 distributions={
#                     "dropout": optuna.distributions.FloatDistribution(0.0001, 0.1),
#                     "l2": optuna.distributions.FloatDistribution(0.00001, 0.001),
#                     "epsilon": optuna.distributions.FloatDistribution(0.0001, 0.1),
#                 },
#                 value=row["test_score"],
#             )
#         )
#     print(f"Loaded {len(trials)} trials")
#     return trials


def objective(trial):
    dropout = round(trial.suggest_float("dropout", 0.0001, 0.1), 16)
    l2 = round(trial.suggest_float("l2", 0.00001, 0.001), 16)
    epsilon = round(trial.suggest_float("epsilon", 0.0001, 0.1), 16)

    df = pd.read_csv("model_stats.csv")

    rows = df.loc[
        (df["l2"] == l2) & (df["dropout"] == dropout) & (df["epsilon"] == epsilon)
    ]
    if len(rows) == 0:
        print(f"New model needed")
        print(f"dropout: {dropout}, l2: {l2}, epsilon: {epsilon}")
        create_bash_script(l2, dropout, epsilon)
        print(f"Running script")

        raise Exception(
            f"New model needed with: dropout: {dropout}, l2: {l2}, epsilon: {epsilon}"
        )

    else:
        print(f"Model already exists")

        test_scores = eval(
            df.loc[
                (df["l2"] == l2)
                & (df["dropout"] == dropout)
                & (df["epsilon"] == epsilon)
            ]["test_score"].loc[0]
        )
        return test_scores[-1]


study = optuna.create_study(sampler=RandomSampler(1337), direction="maximize")
study.optimize(objective, n_trials=10)
