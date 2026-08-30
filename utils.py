from visualize import Cube, Pipe

# ==============================================================================
# 1. STATE TUPLE INDEX CONSTANTS
# ==============================================================================
# States are represented as tuples: (pos, orientation, dir, type)
POS = 0        # 3D coordinate tuple: (x, y, z)
ORIENT = 1     # 3-character string: e.g., "XZZ"
DIR = 2        # Direction vector tuple: e.g., (1, 0, 0)
TYPE = 3       # Face type label: X, Z, or 0 ("0")


OPPOSITE = {
    "X": "Z",
    "Z": "X",
}


# ==============================================================================
# 3. 3D DIRECTION VECTORS
# ==============================================================================
# Order: [+X, -X, +Y, -Y, +Z, -Z]
directions = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
]

directions_2d = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
]



# ==============================================================================
# 4. VECTOR HELPER FUNCTIONS
# ==============================================================================
def add(a, b):
    """Component-wise addition of two 3D vector tuples."""
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def negate(v):
    """Inverts the direction of a 3D vector tuple."""
    return (-v[0], -v[1], -v[2])


def abs_tuple(v):
    """Returns absolute values of a 3D vector tuple to identify its axis."""
    return (abs(v[0]), abs(v[1]), abs(v[2]))


# ==============================================================================
# 5. ORIENTATION DOMAIN & TRANSITION LOOKUP TABLES
# ==============================================================================
ORIENTATIONS = [
    "XXX", "XXZ", "XZX", "XZZ",
    "ZXX", "ZXZ", "ZZX", "ZZZ",
]

SEARCH_TRANSITIONS = {}
DRAW_TRANSITIONS = {}


def search_rotate_orientation(orientation, old_dir, new_dir):
    """Swaps face codes between the old travel axis and the new travel axis."""
    old_axis = abs_tuple(old_dir).index(1)
    new_axis = abs_tuple(new_dir).index(1)

    o = list(orientation)
    o[old_axis], o[new_axis] = o[new_axis], o[old_axis]
    return "".join(o)


def draw_rotate_orientation(orientation, in_dir, out_dir):
    """Computes orientation string for visual rendering."""
    in_axis = abs_tuple(in_dir).index(1)
    out_axis = abs_tuple(out_dir).index(1)

    o = list(orientation)
    o[in_axis] = orientation[out_axis]
    return "".join(o)


for orientation in ORIENTATIONS:
    for old_dir in directions:
        for new_dir in directions:
            if new_dir == negate(old_dir):
                continue
            SEARCH_TRANSITIONS[(orientation, old_dir, new_dir)] = (
                search_rotate_orientation(orientation, old_dir, new_dir)
            )

for orientation in ORIENTATIONS:
    for in_dir in directions:
        for out_dir in directions:
            if out_dir == negate(in_dir):
                continue
            DRAW_TRANSITIONS[(orientation, in_dir, out_dir)] = draw_rotate_orientation(
                orientation, in_dir, out_dir
            )

def draw_path(path, start, goal, ax):
    """Renders a calculated A* path in Matplotlib 3D space."""
    cubes = []

    cubes.append(Cube(start[POS], start[ORIENT], start[DIR], start[TYPE]))

    orientation = start[ORIENT]

    for i in range(1, len(path) - 1):
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

        orientation = DRAW_TRANSITIONS[(orientation, in_dir, out_dir)]
        cubes.append(Cube(path[i], orientation))

    cubes.append(Cube(goal[POS], goal[ORIENT], goal[DIR], goal[TYPE]))

    for cube in cubes[1:-1]:
        cube.draw(ax)

    for i in range(len(cubes) - 1):
        c1 = cubes[i]
        c2 = cubes[i + 1]
        pipe = Pipe(pos1=c1.pos, pos2=c2.pos, orientation=c1.orientation)
        pipe.draw(ax)


def draw_straight_path(start_pos, goal_pos, orientation, ax):
    """Draws a straight line path of cubes and connecting pipes."""
    dx = goal_pos[0] - start_pos[0]
    dy = goal_pos[1] - start_pos[1]
    dz = goal_pos[2] - start_pos[2]

    step_x = 0 if dx == 0 else (1 if dx > 0 else -1)
    step_y = 0 if dy == 0 else (1 if dy > 0 else -1)
    step_z = 0 if dz == 0 else (1 if dz > 0 else -1)

    length = max(abs(dx), abs(dy), abs(dz))

    path = [
        (
            start_pos[0] + i * step_x,
            start_pos[1] + i * step_y,
            start_pos[2] + i * step_z,
        )
        for i in range(length + 1)
    ]

    for pos in path[1:-1]:
        cube = Cube(pos, orientation)
        cube.draw(ax)

    for i in range(len(path) - 1):
        pipe = Pipe(pos1=path[i], pos2=path[i + 1], orientation=orientation)
        pipe.draw(ax)

