import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

cmap = {
    "X": "red",
    "Y": "green",
    "Z": "blue",
    "H": "yellow",
    "0": "gray",
}

H_PIPE_COLORS = ["yellow"] * 6

def set_axes_equal(ax):

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]

    x_mid = sum(x_limits) / 2
    y_mid = sum(y_limits) / 2
    z_mid = sum(z_limits) / 2

    plot_radius = max([x_range, y_range, z_range]) / 2

    ax.set_xlim(
        x_mid - plot_radius,
        x_mid + plot_radius,
    )

    ax.set_ylim(
        y_mid - plot_radius,
        y_mid + plot_radius,
    )

    ax.set_zlim(
        z_mid - plot_radius,
        z_mid + plot_radius,
    )


class Cube:

    def __init__(
        self,
        pos,
        orientation,
        dir=None,
        type=None,
    ):

        self.pos = pos
        self.orientation = orientation
        self.dir = dir
        self.type = type

    def get_vertices(self):

        r = 0.2

        x, y, z = self.pos

        return [
            [x - r, y - r, z - r],
            [x + r, y - r, z - r],
            [x + r, y + r, z - r],
            [x - r, y + r, z - r],
            [x - r, y - r, z + r],
            [x + r, y - r, z + r],
            [x + r, y + r, z + r],
            [x - r, y + r, z + r],
        ]

    def get_faces(self):

        v = self.get_vertices()

        return [
            [v[1], v[2], v[6], v[5]],
            [v[0], v[3], v[7], v[4]],
            [v[2], v[6], v[7], v[3]],
            [v[1], v[5], v[4], v[0]],
            [v[4], v[5], v[6], v[7]],
            [v[0], v[1], v[2], v[3]],
        ]

    def get_colors(self):

        return [cmap[ch] for ch in self.orientation for _ in range(2)]

    def draw(self, ax):

        faces = self.get_faces()

        colors = self.get_colors()

        ax.add_collection3d(
            Poly3DCollection(
                faces,
                facecolors=colors,
                edgecolors="k",
            )
        )


class Pipe:

    def __init__(self, cube1, cube2):

        self.cube1 = cube1
        self.cube2 = cube2

    def get_faces(self):

        v1 = self.cube1.get_vertices()
        v2 = self.cube2.get_vertices()

        d = (
            self.cube2.pos[0] - self.cube1.pos[0],
            self.cube2.pos[1] - self.cube1.pos[1],
            self.cube2.pos[2] - self.cube1.pos[2],
        )

        if d == (1, 0, 0):

            v = [
                v2[0],
                v1[1],
                v1[2],
                v2[3],
                v2[4],
                v1[5],
                v1[6],
                v2[7],
            ]

        elif d == (-1, 0, 0):

            v = [
                v1[0],
                v2[1],
                v2[2],
                v1[3],
                v1[4],
                v2[5],
                v2[6],
                v1[7],
            ]

        elif d == (0, 1, 0):

            v = [
                v2[0],
                v2[1],
                v1[2],
                v1[3],
                v2[4],
                v2[5],
                v1[6],
                v1[7],
            ]

        elif d == (0, -1, 0):

            v = [
                v1[0],
                v1[1],
                v2[2],
                v2[3],
                v1[4],
                v1[5],
                v2[6],
                v2[7],
            ]

        elif d == (0, 0, 1):

            v = [
                v2[0],
                v2[1],
                v2[2],
                v2[3],
                v1[4],
                v1[5],
                v1[6],
                v1[7],
            ]

        elif d == (0, 0, -1):

            v = [
                v1[0],
                v1[1],
                v1[2],
                v1[3],
                v2[4],
                v2[5],
                v2[6],
                v2[7],
            ]

        return [
            [v[1], v[2], v[6], v[5]],
            [v[0], v[3], v[7], v[4]],
            [v[2], v[6], v[7], v[3]],
            [v[1], v[5], v[4], v[0]],
            [v[4], v[5], v[6], v[7]],
            [v[0], v[1], v[2], v[3]],
        ]

    def draw(self, ax, colors = None):

        faces = self.get_faces()
        if colors is None:
            colors = self.cube1.get_colors()
        ax.add_collection3d(
            Poly3DCollection(
                faces,
                facecolors=colors,
                edgecolors="k",
            )
        )
