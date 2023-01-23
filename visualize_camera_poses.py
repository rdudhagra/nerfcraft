import json
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d import Axes3D
import os

# Variables
DATA_DIR = "./torch-ngp/data/hamerschlag"

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


# Read the data
with open(os.path.join(DATA_DIR, "transforms.json"), "r") as f:
    data = json.load(f)

    for frame in data["frames"]:
        transform_matrix = frame["transform_matrix"]
        start_point = np.array([0, 0, 0, 1])
        start_point = np.dot(transform_matrix, start_point)

        for color, end_point in {
            "r": np.array([1, 0, 0, 1]),
            "g": np.array([0, 1, 0, 1]),
            "b": np.array([0, 0, 1, 1]),
        }.items():
            end_point = np.dot(transform_matrix, end_point)

            a = Arrow3D(
                [start_point[0], end_point[0]],
                [start_point[1], end_point[1]],
                [start_point[2], end_point[2]],
                mutation_scale=5,
                lw=2,
                arrowstyle="-|>",
                color=color,
            )
            ax.add_artist(a)

            # Plot line
            ax.plot(
                [start_point[0], start_point[0]],
                [start_point[1], start_point[1]],
                [start_point[2], start_point[2]],
                color=color,
            )

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set axes equal
limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
x, y, z = np.mean(limits, axis=1)
radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
ax.set_xlim3d([x - radius, x + radius])
ax.set_ylim3d([y - radius, y + radius])
ax.set_zlim3d([z - radius, z + radius])

plt.show()
