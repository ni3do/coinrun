import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg


def fit_3d(data, order=2, steps=100, x_name="dropout", y_name="epsilon"):
    # print(f"Fitting {x_name} and {y_name}")
    # print(f"data x_name:\n", data[x_name])
    # print(f"data y_name:\n", data[y_name])
    X, Y = np.meshgrid(
        np.arange(
            data[x_name].min(),
            data[x_name].max(),
            (data[x_name].max() - data[x_name].min()) / steps,
        ),
        np.arange(
            data[y_name].min(),
            data[y_name].max(),
            (data[y_name].max() - data[y_name].min()) / steps,
        ),
    )
    XX = X.flatten()
    YY = Y.flatten()

    np_data = data.to_numpy()
    # 1: linear, 2: quadratic, 3: cubic
    if order == 1:
        # best-fit linear plane
        A = np.c_[np_data[:, 0], np_data[:, 1], np.ones(np_data.shape[0])]
        C, _, _, _ = scipy.linalg.lstsq(A, np_data[:, 2])  # coefficients

        # evaluate it on grid
        Z = C[0] * X + C[1] * Y + C[2]

        # or expressed using matrix/vector product
        # Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)

    elif order == 2:
        # best-fit quadratic curve
        A = np.c_[
            np.ones(np_data.shape[0]),
            np_data[:, :2],
            np.prod(np_data[:, :2], axis=1),
            np_data[:, :2] ** 2,
        ]
        C, _, _, _ = scipy.linalg.lstsq(A, np_data[:, 2])

        # evaluate it on a grid
        Z = np.dot(
            np.c_[np.ones(XX.shape), XX, YY, XX * YY, XX**2, YY**2], C
        ).reshape(X.shape)

    elif order == 3:
        # best-fit cubic curve
        A = np.c_[
            np.ones(np_data.shape[0]),
            np_data[:, :2],
            np.prod(np_data[:, :2], axis=1),
            np_data[:, :2] ** 2,
            np_data[:, :2] ** 3,
        ]
        C, _, _, _ = scipy.linalg.lstsq(A, np_data[:, 2])
        # evaluate it on a grid
        Z = np.dot(
            np.c_[
                np.ones(XX.shape), XX, YY, XX * YY, XX**2, YY**2, XX**3, YY**3
            ],
            C,
        ).reshape(X.shape)

    return X, Y, Z, C


def fit_4d(data, order=2, steps=100):
    np_data = data.to_numpy()
    X, Y, T = np.meshgrid(
        np.arange(
            data["dropout"].min(),
            data["dropout"].max(),
            (data["dropout"].max() - data["dropout"].min()) / steps,
        ),
        np.arange(
            data["epsilon"].min(),
            data["epsilon"].max(),
            (data["epsilon"].max() - data["epsilon"].min()) / steps,
        ),
        np.arange(
            data["l2"].min(),
            data["l2"].max(),
            (data["l2"].max() - data["l2"].min()) / steps,
        ),
    )
    XX = X.flatten()
    YY = Y.flatten()
    TT = T.flatten()

    print("np_data[:,3]:\n", np_data[:, 3])
    # 1: linear, 2: quadratic, 3: cubic
    if order == 1:
        # best-fit linear plane
        A = np.c_[
            np_data[:, 0], np_data[:, 1], np_data[:, 2], np.ones(np_data.shape[0])
        ]
        C, _, _, _ = scipy.linalg.lstsq(A, np_data[:, 3])  # coefficients
        # evaluate it on grid
        Z = C[0] * X + C[1] * Y + C[2] * T + C[3]
        # or expressed using matrix/vector product
        # Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)

    elif order == 2:
        # best-fit quadratic curve
        A = np.c_[
            np.ones(np_data.shape[0]),
            np_data[:, :2],
            np.prod(np_data[:, :2], axis=1),
            np_data[:, :2] ** 2,
        ]
        C, _, _, _ = scipy.linalg.lstsq(A, np_data[:, 2])
        # evaluate it on a grid
        Z = np.dot(
            np.c_[np.ones(XX.shape), XX, YY, XX * YY, XX**2, YY**2], C
        ).reshape(X.shape)

    elif order == 3:
        # best-fit cubic curve
        A = np.c_[
            np.ones(np_data.shape[0]),
            np_data[:, :2],
            np.prod(np_data[:, :2], axis=1),
            np_data[:, :2] ** 2,
            np_data[:, :2] ** 3,
        ]
        C, _, _, _ = scipy.linalg.lstsq(A, np_data[:, 2])
        # evaluate it on a grid
        Z = np.dot(
            np.c_[
                np.ones(XX.shape), XX, YY, XX * YY, XX**2, YY**2, XX**3, YY**3
            ],
            C,
        ).reshape(X.shape)

    return X, Y, Z, T, C


if __name__ == "__main__":
    value_dict = {
        "dropout": [0.1, 0.05, 0.01],
        "epsilon": [0.01, 0.005, 0.001],
        "l2": [0.0001, 0.00005, 0.00002],
    }

    remaining_axis = {
        "dropout": ["l2", "epsilon"],
        "epsilon": ["dropout", "l2"],
        "l2": ["dropout", "epsilon"],
    }

    data = pd.read_csv("csv/grid_stats2.csv")

    for order in [1, 2, 3]:
        for fixed_axis in value_dict.keys():
            for fixed_val in value_dict[fixed_axis]:
                # norm = matplotlib.colors.Normalize(vmin=0, vmax=10, clip=True)
                # mapper = cm.ScalarMappable(norm=norm)
                print(f"Iteration: {order}, {fixed_axis}, {fixed_val}")
                trunc_data = data[data[fixed_axis] == fixed_val].drop(
                    columns=[fixed_axis], inplace=False
                )
                # print(trunc_data)
                X, Y, Z, C = fit_3d(
                    trunc_data,
                    order=order,
                    steps=100,
                    x_name=remaining_axis[fixed_axis][0],
                    y_name=remaining_axis[fixed_axis][1],
                )
                # print("C:\n", C)
                # print("Z:\n", Z)
                # plot points and fitted surface
                # fig = plt.figure(figsize=(10, 10))
                fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
                surf = ax.plot_surface(
                    X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True
                )
                # ax.scatter(data[:,0], data[:,1], data[:,2], c='r', s=50)
                ax.set_title(
                    f"Fitted plane (order {order}) for fixed {fixed_axis}={fixed_val}"
                )
                plt.xlabel(remaining_axis[fixed_axis][0])
                plt.ylabel(remaining_axis[fixed_axis][1])
                ax.set_xlim(
                    value_dict[remaining_axis[fixed_axis][0]][0],
                    value_dict[remaining_axis[fixed_axis][0]][-1],
                )
                ax.set_xticklabels([f"{x:.0e}" for x in ax.get_xticks()])
                ax.set_ylim(
                    value_dict[remaining_axis[fixed_axis][1]][0],
                    value_dict[remaining_axis[fixed_axis][1]][-1],
                )
                ax.set_yticklabels([f"{x:.0e}" for x in ax.get_yticks()])
                ax.set_zlim(5, 10)
                ax.set_zlabel("test_score")
                ax.axis("equal")
                ax.axis("tight")
                fig.colorbar(surf, shrink=0.5, aspect=5, label="test_score")
                # plt.show()
                fig.savefig(
                    f"plot/surface-fits/{fixed_axis}_{fixed_val}_o{order}.png", dpi=100
                )
                fig.clf()

    # X, Y, Z, T, C = fit_4d(data, order=1, steps=2)
    # print(C)
    # print(X[:,:,0])
    # print(T)

    # X = X[:,:,0]
    # Y = Y[:,:,0]
    # T = T[:,:,0]
    # print("4d X:\n", X)
    # print("4d Y:\n", Y)
    # print("4d T:\n", T)
    # print("4d Z:\n", Z)
