import heapq
import itertools

import matplotlib.pyplot as plt

from visualize import Cube, Pipe, set_axes_equal

# ---------------------------------------
# State indices
# ---------------------------------------

POS = 0
ORIENT = 1
DIR = 2
TYPE = 3


# ---------------------------------------
# Face categories
# ---------------------------------------

X = "X"
Z = "Z"
O = "0"

OPPOSITE = {
    X: Z,
    Z: X,
}


# ---------------------------------------
# Directions
# ---------------------------------------

directions = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
]


# ---------------------------------------
# Helpers
# ---------------------------------------


def add(a, b):

    return (
        a[0] + b[0],
        a[1] + b[1],
        a[2] + b[2],
    )


def negate(v):

    return (
        -v[0],
        -v[1],
        -v[2],
    )


def abs_tuple(v):

    return (
        abs(v[0]),
        abs(v[1]),
        abs(v[2]),
    )


def heuristic(a, b):

    pa = a[POS]
    pb = b[POS]

    return abs(pa[0] - pb[0]) + abs(pa[1] - pb[1]) + abs(pa[2] - pb[2])


# ---------------------------------------
# All possible orientations
# ---------------------------------------

ORIENTATIONS = [
    "XXX",
    "XXZ",
    "XZX",
    "XZZ",
    "ZXX",
    "ZXZ",
    "ZZX",
    "ZZZ",
]


# ---------------------------------------
# SEARCH TRANSITIONS
# ---------------------------------------

SEARCH_TRANSITIONS = {}


def search_rotate_orientation(
    orientation,
    old_dir,
    new_dir,
):

    old_axis = abs_tuple(old_dir).index(1)

    new_axis = abs_tuple(new_dir).index(1)

    o = list(orientation)

    o[old_axis], o[new_axis] = (
        o[new_axis],
        o[old_axis],
    )

    return "".join(o)


for orientation in ORIENTATIONS:

    for old_dir in directions:

        for new_dir in directions:

            if new_dir == negate(old_dir):
                continue

            SEARCH_TRANSITIONS[
                (
                    orientation,
                    old_dir,
                    new_dir,
                )
            ] = search_rotate_orientation(
                orientation,
                old_dir,
                new_dir,
            )


# ---------------------------------------
# DRAW TRANSITIONS
# ---------------------------------------

DRAW_TRANSITIONS = {}


def draw_rotate_orientation(
    orientation,
    in_dir,
    out_dir,
):
    in_axis = abs_tuple(in_dir).index(1)
    out_axis = abs_tuple(out_dir).index(1)

    o = list(orientation)

    o[in_axis] = orientation[out_axis]

    return "".join(o)


for orientation in ORIENTATIONS:

    for in_dir in directions:

        for out_dir in directions:

            if out_dir == negate(in_dir):
                continue

            DRAW_TRANSITIONS[
                (
                    orientation,
                    in_dir,
                    out_dir,
                )
            ] = draw_rotate_orientation(
                orientation,
                in_dir,
                out_dir,
            )


# ---------------------------------------
# Start neighbors
# ---------------------------------------


def get_start_neighbors(start):

    neighbours = []

    usable_face = OPPOSITE[start[TYPE]]

    for i in range(3):

        if start[ORIENT][i] == usable_face:

            c1 = (
                add(
                    start[POS],
                    directions[2 * i],
                ),
                start[ORIENT],
                directions[2 * i],
                None,
            )

            c2 = (
                add(
                    start[POS],
                    directions[2 * i + 1],
                ),
                start[ORIENT],
                directions[2 * i + 1],
                None,
            )

            neighbours.append(c1)
            neighbours.append(c2)

    return neighbours


# ---------------------------------------
# Neighbour generation
# ---------------------------------------


def get_neighbours(state):

    neighbours = []

    current_dir = state[DIR]

    current_orientation = state[ORIENT]

    for dir in directions:

        if dir == negate(current_dir):
            continue

        pos = add(
            state[POS],
            dir,
        )

        new_orientation = SEARCH_TRANSITIONS[
            (
                current_orientation,
                current_dir,
                dir,
            )
        ]

        neighbour = (
            pos,
            new_orientation,
            dir,
            None,
        )

        neighbours.append(neighbour)

    return neighbours


# ---------------------------------------
# Goal check
# ---------------------------------------


def check_goal(current, goal):

    if goal[TYPE] == O:
        return current[POS] == goal[POS]

    if current[POS] != goal[POS]:
        return False

    current_abs_dir = abs_tuple(current[DIR])

    entry_axis = current_abs_dir.index(1)

    entry_face = goal[ORIENT][entry_axis]

    if entry_face != OPPOSITE[goal[TYPE]]:
        return False

    for i in range(3):

        if i == entry_axis:
            continue

        if current[ORIENT][i] != goal[ORIENT][i]:
            return False

    return True


# ---------------------------------------
# A*
# ---------------------------------------


def a_star(start, goal, obstacles):
    counter = itertools.count()
    open_set = []
    came_from = {}
    g_score = {start: 0}

    path_positions = {start: {start[POS]}}

    for neighbour in get_start_neighbors(start):
        if neighbour[POS] in obstacles:
            continue

        g = 1
        g_score[neighbour] = g
        came_from[neighbour] = start
        path_positions[neighbour] = path_positions[start] | {neighbour[POS]}

        f = g + heuristic(neighbour, goal)
        heapq.heappush(open_set, (f, g, next(counter), neighbour))

    expanded = 0
    MAX_EXPANDED = 50000

    while open_set:
        expanded += 1
        if expanded > MAX_EXPANDED:
            print("Exceeded max expanded nodes")
            return None
        
        _, current_g, _, current = heapq.heappop(open_set)

        if current_g != g_score[current]:
            continue

        if check_goal(current, goal):
            path = []
            while current in came_from:
                path.append(current[POS])
                current = came_from[current]
            path.append(start[POS])
            return path[::-1]

        for neighbour in get_neighbours(current):
            if neighbour[POS] in obstacles:
                continue
            if neighbour[POS] in path_positions[current]:  # O(1), no chain walk
                continue

            tentative_g = current_g + 1
            if tentative_g >= g_score.get(neighbour, float("inf")):
                continue

            g_score[neighbour] = tentative_g
            came_from[neighbour] = current

            path_positions[neighbour] = path_positions[current] | {neighbour[POS]}

            f_score = tentative_g + heuristic(neighbour, goal)
            heapq.heappush(open_set, (f_score, tentative_g, next(counter), neighbour))

    return None


# ---------------------------------------
# Visualization
# ---------------------------------------


def draw_path(
    path,
    start,
    goal,
    ax,
    hadamard=False
):

    cubes = []

    cubes.append(
        Cube(
            start[POS],
            start[ORIENT],
            start[DIR],
            start[TYPE],
        )
    )

    orientation = start[ORIENT]

    for i in range(
        1,
        len(path) - 1,
    ):

        in_dir = (
            path[i][0] - path[i - 1][0],
            path[i][1] - path[i - 1][1],
            path[i][2] - path[i - 1][2],
        )

        out_dir = (
            path[i + 1][0] - path[i][0],
            path[i + 1][1] - path[i][1],
            path[i + 1][2] - path[i][2],
        )

        orientation = DRAW_TRANSITIONS[
            (
                orientation,
                in_dir,
                out_dir,
            )
        ]

        cubes.append(
            Cube(
                path[i],
                orientation,
            )
        )

    cubes.append(
        Cube(
            goal[POS],
            goal[ORIENT],
            goal[DIR],
            goal[TYPE],
        )
    )

    for cube in cubes[1:-1]:
        cube.draw(ax)

    for i in range(len(cubes) - 1):

        pipe = Pipe(cubes[i], cubes[i + 1])

        if hadamard and i == len(cubes) - 2:
            pipe.draw(ax, colors=["yellow"] * 6)
        else:
            pipe.draw(ax)

# ---------------------------------------
# Main
# ---------------------------------------

if __name__ == "__main__":

    fig = plt.figure()

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    start = (
        (0, 0, 0),
        "XZZ",
        None,
        X,
    )

    goal = (
        (0, 5, 1),
        "XZX",
        None,
        Z,
    )

    obstacles = {(0,5,0)}

    path = a_star(
        start,
        goal,
        obstacles,
    )

    print(start, goal)

    if path is not None:

        draw_path(
            path,
            start,
            goal,
            ax,
        )
        c1 = Cube(*start)
        c1.draw(ax)
        c2 = Cube(*goal)
        c2.draw(ax)

    else:

        print("No path found")

    set_axes_equal(ax)

    ax.set_box_aspect([1, 1, 1])

    plt.show()
