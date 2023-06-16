import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('csv/grid_stats.csv')

fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("3D Plot of Grid Search Results")
ax.set_xlabel("Dropout rate")
ax.set_ylabel("L2 regularization")
ax.set_zlabel("Epsilon greediness")

rows = []
minima = 10
maxima = 0
best_run = None
# type,dropout,l2,epsilon,test_score
for idx, row in df.iterrows():
    test_scores = eval(row["test_score"])
    last_score = test_scores[len(test_scores)-1]
    
    if last_score < minima:
        minima = last_score
    if last_score > maxima:
        maxima = last_score
        best_run = row
    # print(row["dropout"], row["l2"], row["epsilon"])
    # print(test_scores)
    rows.append([row["dropout"], row["l2"], row["epsilon"], last_score])

norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
mapper = cm.ScalarMappable(norm=norm)

for row in rows:
    ax.scatter(row[0], row[1], row[2], color=mapper.to_rgba(row[3]), s=200)

plt.colorbar(mappable=mapper)

fig.legend()
fig.savefig('3d_plot.png')
print(best_run)

for l2 in [0.00002, 0.00005, 0.0001]:
    plt.clf()
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.set_title(f"Plot of Grid Search Results for L2={l2}")
    ax.set_xlabel("Dropout rate")
    ax.set_ylabel("Epsilon greediness")
    for row in rows:
        if row[1] == l2:
            ax.scatter(row[0], row[2], color=mapper.to_rgba(row[3]), s=1000)
    
    plt.colorbar(mappable=mapper)
    fig.legend()
    fig.savefig(f'3d_plot_l2_{l2}.png')

plt.clf()
for l2 in [0.00002, 0.00005, 0.0001]:
    X = [0.1, 0.05, 0.01]
    Y = [0.01, 0.005, 0.001]
    Z = np.ndarray((3,3))
    x_ctr = 0
    print(Z)
    for dropout in [0.1, 0.05, 0.01]:
        y_ctr = 0
        for epsilon in [0.01, 0.005, 0.001]:
            print(df.loc[
                (df["l2"] == l2)
                & (df["dropout"] == dropout)
                & (df["epsilon"] == epsilon)
            ]["test_score"])
            Z[x_ctr][y_ctr] = eval(
            df.loc[
                (df["l2"] == l2)
                & (df["dropout"] == dropout)
                & (df["epsilon"] == epsilon)
            ]["test_score"].iloc[0]
            )[-1]
            y_ctr += 1
        x_ctr += 1
            
    print(Z)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=True)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.savefig('3d_plot_surface.png')