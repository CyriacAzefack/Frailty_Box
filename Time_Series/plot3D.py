# Facebook Prophet example
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import PolyCollection

sns.set_style('darkgrid')


def main():
    activity = 'sleeping2'
    raw_dataset = pd.read_csv(f'./data/{activity}_dataset.csv', sep=';').drop(['tw_id'], axis=1)

    raw_dataset.insert(0, "T_init", [0] * len(raw_dataset))
    raw_dataset["T_last"] = [0] * len(raw_dataset)

    nb_feat = len(raw_dataset.columns)
    nb_tstep = len(raw_dataset)
    # t_step = period / nb_feat

    z = np.arange(len(raw_dataset))
    T = np.arange(nb_feat)

    sx = T.size
    sy = z.size

    T = np.tile(T, (sy, 1))

    z = np.tile(z, (sx, 1)).T

    U = raw_dataset.values

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    verts = []
    for i in range(T.shape[0]):
        verts.append(list(zip(T[i, :], U[i, :])))

    poly = PolyCollection(verts, facecolors=(1, 1, 1, 1), edgecolors=(0, 0, 1, 1))
    poly.set_alpha(0.3)
    ax.add_collection3d(poly, zs=z[:, 0], zdir='y')
    ax.set_xlim3d(np.min(T), np.max(T))
    ax.set_ylim3d(np.min(z), np.max(z))
    ax.set_zlim3d(np.min(U), np.max(U))
    ax.set_xlabel("Heure de la journée")
    ax.set_zlabel("Nombre d'occurrences")
    ax.set_ylabel('ID de la fenêtre temporelle')

    # surf = ax.plot_wireframe(T, z, U, cstride=1000)
    plt.show()


if __name__ == '__main__':
    main()
