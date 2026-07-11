import pyzx as zx
from three_deg_simp import simplify_graph
import math
from pyzx.utils import VertexType
import random
import matplotlib.pyplot as plt

from visualize import Cube, Pipe, set_axes_equal
from a_star import a_star, draw_path, X, Z

# X_ORIENT = ["XZZ", "ZXZ", "ZZX"]
X_ORIENT = ["XZZ"]


# Z_ORIENT = ["ZXX", "XZX", "XXZ"]
Z_ORIENT = ["XZX"]


TYPE = {
    VertexType.BOUNDARY: "0",
    VertexType.Z: Z,
    VertexType.X: X,
    VertexType.H_BOX: "H",
}


def assign_initial_layers(g):
    layer = {}
    qubits = sorted({g.qubit(v) for v in g.vertices()})

    for q in qubits:
        verts = [v for v in g.vertices() if g.qubit(v) == q]
        verts.sort(key=g.row)
        for i, v in enumerate(verts):
            layer[v] = i
    return layer


def next_layer_z(occupied):
    if not occupied:
        return 0
    max_z = max(p[2] for p in occupied)
    return max_z + 1


def assign_orientation(g):
    orientation = {}

    qubits = sorted({g.qubit(v) for v in g.vertices()})

    for q in qubits:

        # Vertices on this qubit ordered from input to output
        verts = [v for v in g.vertices() if g.qubit(v) == q]
        verts.sort(key=g.row)

        flipped = False

        for i, v in enumerate(verts):

            if g.type(v) == VertexType.BOUNDARY:
                orientation[v] = "000"

            elif g.type(v) == VertexType.X:
                orientation[v] = "ZXZ" if flipped else "XZZ"

            elif g.type(v) == VertexType.Z:
                orientation[v] = "ZXX" if flipped else "XZX"

            else:
                orientation[v] = "000"

            # Look at the edge to the next vertex on the same qubit
            if i + 1 < len(verts):

                nxt = verts[i + 1]

                e = g.edge(v, nxt)
                if e is not None and g.edge_type(e) == zx.EdgeType.HADAMARD:
                    flipped = not flipped

    return orientation


circ = zx.Circuit.from_qasm_file("benchmark/ghz_16.qasm")
# # circ = zx.Circuit.from_qasm_file("benchmark/bv_16.qasm")
# circ = zx.Circuit.from_qasm_file("random_bench/seed_2.qasm")

g = circ.to_graph()
g = simplify_graph(g)
layer = assign_initial_layers(g)
success = True
layers = {}

for v, l in layer.items():
    layers.setdefault(l, []).append(v)
for l in layers:
    layers[l].sort(key=g.qubit)

print(layers)
zx.draw(g, labels=True)

max_qubit = max(g.qubits().values()) + 1
max_layer = max(layers.keys()) + 1
STEP = 2
k = math.ceil(math.sqrt(max_qubit))

states = {}
cubes = {}
placed = {}
for node in g.vertices():
    placed[node] = False

adj = {}
wire_edges = []

for e in g.edges():
    s, t = g.edge_st(e)

    if g.qubit(s) == g.qubit(t):
        wire_edges.append(e)
        continue

    adj.setdefault(s, []).append((e, t))
    adj.setdefault(t, []).append((e, s))

orientations = assign_orientation(g)
occupied = set()
MAX_Z = 10 * len(layers)
qubit_obstacles = set()
for q in range(max_qubit):

    x = STEP * (q // k)
    y = STEP * (q % k)

    for z in range(MAX_Z + 1):
        qubit_obstacles.add((x, y, z))

routed_edges = set()
routed_paths = []
current_z = 0


for layer_num in sorted(layers.keys()):
    for node in layers[layer_num]:

        node_type = TYPE[g.type(node)]
        x = STEP * (g.qubit(node) // k)
        y = STEP * (g.qubit(node) % k)
        z = current_z
        pos = (x, y, z)

        direction = None
        orientation = orientations[node]
        states[node] = (
            pos,
            orientation,
            direction,
            node_type,
        )

        cubes[node] = Cube(
            pos,
            orientation,
            direction,
            node_type,
        )

        placed[node] = True

        occupied.add(pos)

    # ----------------------------
    # Route newly available edges
    # ----------------------------

    for node in layers[layer_num]:

        for e, nbr in adj.get(node, []):

            if e in routed_edges:
                continue

            if not placed[nbr]:
                continue

            s = node
            t = nbr

            # Ignore boundary direction
            if TYPE[g.type(s)] == "0":
                s, t = t, s

            start = states[s]
            goal = states[t]

            obstacles = occupied | qubit_obstacles
            obstacles.discard(start[0])
            obstacles.discard(goal[0])

            path = a_star(
                start,
                goal,
                obstacles,
            )

            if path is None:
                print(f"Failed : {s} -> {t}")
                success = False
                continue

            print(f"Routed : {s} -> {t}")

            routed_edges.add(e)
            is_hadamard = g.edge_type(e) == zx.EdgeType.HADAMARD
            routed_paths.append((path, start, goal, is_hadamard))

            # Reserve routed voxels
            for p in path[1:-1]:
                occupied.add(p)

    current_z = next_layer_z(occupied)

# ----------------------------
# Route wire edges (straight Z)
# ----------------------------

for e in wire_edges:

    s, t = g.edge_st(e)
    if TYPE[g.type(s)] == "0":
        s, t = t, s

    if not placed[s] or not placed[t]:
        continue

    start = states[s]
    goal = states[t]

    start_pos = start[0]
    goal_pos = goal[0]

    # Same qubit, so x and y are identical
    assert start_pos[0] == goal_pos[0]
    assert start_pos[1] == goal_pos[1]

    path = []

    z_start = start_pos[2]
    z_end = goal_pos[2]

    direction = 1 if z_end >= z_start else -1

    for z in range(z_start, z_end + direction, direction):
        path.append((start_pos[0], start_pos[1], z))

    is_hadamard = g.edge_type(e) == zx.EdgeType.HADAMARD
    routed_paths.append((path, start, goal, is_hadamard))

    # Reserve intermediate voxels
    for p in path[1:-1]:
        occupied.add(p)


# ----------------------------
# Final visualization
# ----------------------------

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Draw all cubes
for node in g.vertices():
    cubes[node].draw(ax)

# Draw all routed paths
for path, start, goal, hadamard in routed_paths:
    draw_path(path, start, goal, ax, hadamard=hadamard)

print("Success: ", success)

# ----------------------------
# Bounding box statistics
# ----------------------------

xs = [p[0] for p in occupied]
ys = [p[1] for p in occupied]
zs = [p[2] for p in occupied]

xmin, xmax = min(xs), max(xs)
ymin, ymax = min(ys), max(ys)
zmin, zmax = min(zs), max(zs)

# Dimensions in voxel coordinates (inclusive)
x_dim = xmax - xmin + 1
y_dim = ymax - ymin + 1
z_dim = zmax - zmin + 1

volume = x_dim * y_dim * z_dim

print("\nBounding box")
print(f"  X: [{xmin}, {xmax}]  size = {x_dim}")
print(f"  Y: [{ymin}, {ymax}]  size = {y_dim}")
print(f"  Z: [{zmin}, {zmax}]  size = {z_dim}")
print(f"Bounding box volume = {volume}")

set_axes_equal(ax)
ax.set_box_aspect([1, 1, 1])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Final Routed ZX Graph")

plt.show()
