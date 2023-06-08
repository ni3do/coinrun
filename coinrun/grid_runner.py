import pandas as pd


def create_bash_script(l2, dropout, epsilon):
    dp_line = f"dp={dropout}\n"
    l_two_line = f"l_two={l2}\n"
    epsilon_line = f"epsilon={epsilon}\n"
    model_name_line = f'model_name="reg-$dp-$l_two-$epsilon"\n'

    with open("scripts/base_0.sh", "r") as f:
        base_0 = f.read()
    with open("scripts/base_1.sh", "r") as f:
        base_1 = f.read()

    try:
        dropout_stripped = str(dropout).split(".")[1]
    except IndexError:
        dropout_stripped = str(dropout)

    try:
        l2_stripped = str(l2).split(".")[1]
    except IndexError:
        l2_stripped = str(l2)
    try:
        epsilon_stripped = str(epsilon).split(".")[1]
    except IndexError:
        epsilon_stripped = str(epsilon)

    with open(
        f"scripts/auto/{l2_stripped}_{dropout_stripped}_{epsilon_stripped}.sh",
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


l2_arr = [0.00002, 0.00005, 0.0001]
dropout_arr = [0.01, 0.05, 0.1]
epsilon_arr = [0.001, 0.005, 0.01]

for l2 in l2_arr:
    for dropout in dropout_arr:
        for epsilon in epsilon_arr:
            create_bash_script(l2, dropout, epsilon)
