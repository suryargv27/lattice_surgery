import math
import random
import numpy as np
from collections import defaultdict
from fractions import Fraction


def is_pi4_family_phase(phase):
    """True for pi/4 or 7pi/4 (i.e. denominator-4 phases, T/T-dagger family)."""
    if phase is None:
        return False
    frac = Fraction(phase) % 2
    return frac.denominator == 4


def build_qubit_interaction_stats(g):
    cnot_weight = defaultdict(int)
    t_count = defaultdict(int)

    for v in g.vertices():
        q = g.qubit(v)
        if q < 0:
            continue
        if is_pi4_family_phase(g.phase(v)):
            t_count[q] += 1

    for e in g.edges():
        u, v = g.edge_st(e)
        qu, qv = g.qubit(u), g.qubit(v)
        if qu < 0 or qv < 0 or qu == qv:
            continue
        a, b = min(qu, qv), max(qu, qv)
        cnot_weight[(a, b)] += 1

    return cnot_weight, t_count


def spectral_initial_layout(qubits, cnot_weight):
    n = len(qubits)
    idx = {q: i for i, q in enumerate(qubits)}

    W = np.zeros((n, n))
    for (a, b), w in cnot_weight.items():
        if a in idx and b in idx:
            i, j = idx[a], idx[b]
            W[i, j] += w
            W[j, i] += w

    deg = np.diag(W.sum(axis=1))
    L = deg - W

    if n <= 2:
        return {q: (float(i), 0.0) for i, q in enumerate(qubits)}

    eigvals, eigvecs = np.linalg.eigh(L)
    order = np.argsort(eigvals)
    v1 = eigvecs[:, order[1]]
    v2 = eigvecs[:, order[2]]

    def norm(v):
        span = v.max() - v.min()
        return (v - v.min()) / span if span > 1e-9 else np.zeros_like(v)

    v1n, v2n = norm(v1), norm(v2)
    return {q: (float(v1n[idx[q]]), float(v2n[idx[q]])) for q in qubits}


def generate_centered_grid_cells(n, step=2):
    """
    Generates at least `n` integer grid cells (in units of `step`), centered
    on (0, 0), ordered by increasing distance (ring/shell) from the center.
    No square/rectangle constraint -- just a centered lattice grown outward
    until it has enough cells, so "exterior" == "far from origin" in a
    genuinely radial sense.
    """
    cells = set()
    r = 0
    # grow a square ring-by-ring (in lattice-cell units, not grid units)
    # until we have enough points, then keep only the n closest to center
    while len(cells) < n:
        r += 1
        for x in range(-r, r + 1):
            for y in range(-r, r + 1):
                if max(abs(x), abs(y)) == r:  # ring shell only, avoid re-adding interior
                    cells.add((x, y))
        # also make sure (0,0) is included on first pass
        cells.add((0, 0))

    cells = list(cells)
    cells.sort(key=lambda c: math.hypot(c[0], c[1]))
    chosen = cells[:n]
    return [(x * step, y * step) for x, y in chosen]


def layout_score(positions, qubits, cnot_weight, t_count, center):
    cnot_cost = 0.0
    for (a, b), w in cnot_weight.items():
        pa, pb = positions[a], positions[b]
        d = math.hypot(pa[0] - pb[0], pa[1] - pb[1])
        cnot_cost += w * d

    t_cost = 0.0
    max_t = max(t_count.values()) if t_count else 0
    if max_t > 0:
        for q in qubits:
            pq = positions[q]
            d_center = math.hypot(pq[0] - center[0], pq[1] - center[1])
            t_weight = t_count.get(q, 0) / max_t
            t_cost += -t_weight * d_center

    return cnot_cost + t_cost


def simulated_annealing_layout(
    g,
    step=2,
    iterations=20000,
    start_temp=5.0,
    end_temp=0.01,
    seed=None,
):
    """
    Assigns each qubit line an (x, y) on a centered, step=2-spaced 2D grid
    (no square constraint). CNOT-heavy qubit pairs are pulled close together;
    qubits with many pi/4 / 7pi/4 nodes are pushed toward the grid exterior
    (far from the centroid of all placed qubits).

    Phase 1: spectral embedding from the CNOT interaction graph for a good
             continuous starting layout.
    Phase 2: simulated annealing (grid-cell swaps) refines against the
             combined CNOT-closeness / T-gate-exteriority objective.
    """
    rng = random.Random(seed)

    qubits = sorted({g.qubit(v) for v in g.vertices() if g.qubit(v) >= 0})
    n = len(qubits)
    if n == 0:
        return {}
    if n == 1:
        return {qubits[0]: (0, 0)}

    cnot_weight, t_count = build_qubit_interaction_stats(g)

    # ---- Phase 1: spectral initial layout ----
    spectral_pos = spectral_initial_layout(qubits, cnot_weight)

    # ---- Generate the centered candidate grid cells (no square limit) ----
    grid_cells = generate_centered_grid_cells(n, step=step)

    # Assign qubits to cells: central spectral points -> central cells
    spectral_center = (0.5, 0.5)
    ordering = sorted(
        qubits,
        key=lambda q: math.hypot(
            spectral_pos[q][0] - spectral_center[0],
            spectral_pos[q][1] - spectral_center[1],
        ),
    )
    remaining_cells = sorted(grid_cells, key=lambda c: math.hypot(c[0], c[1]))

    assignment = {}
    for q in ordering:
        assignment[q] = remaining_cells.pop(0)

    positions = dict(assignment)
    center = (0.0, 0.0)  # grid is already centered at origin

    # ---- Phase 2: simulated annealing (swap-based) ----
    current_score = layout_score(positions, qubits, cnot_weight, t_count, center)
    best_positions = dict(positions)
    best_score = current_score

    for it in range(iterations):
        temp = start_temp * ((end_temp / start_temp) ** (it / iterations))

        qa, qb = rng.sample(qubits, 2)
        positions[qa], positions[qb] = positions[qb], positions[qa]

        new_score = layout_score(positions, qubits, cnot_weight, t_count, center)
        delta = new_score - current_score

        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-9)):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_positions = dict(positions)
        else:
            positions[qa], positions[qb] = positions[qb], positions[qa]

    return best_positions

from collections import defaultdict, deque
from pyzx.utils import VertexType


from collections import deque
from fractions import Fraction
from pyzx.utils import VertexType


def compute_t_depth(g, t_predicate=None):
    """
    Builds the T-only dependency sub-DAG from a ZX graph and computes T-depth
    via longest-path layering, restricted to T-type nodes.

    An edge t_i -> t_j exists in the sub-DAG iff t_j depends on t_i, i.e.
    there is a path from t_i to t_j in the original graph that does not pass
    through any other T-node. Found via reverse BFS from each T-node,
    stopping (not expanding past) any other T-node encountered.

    Parameters
    ----------
    g : pyzx BaseGraph
        The ZX graph, with g.row(v) giving a valid topological ordering
        (e.g. already passed through assign_layers).
    t_predicate : callable(g, v) -> bool, optional
        Identifies T-type nodes. Defaults to Z-type vertices with phase a
        multiple of pi/4 but not pi/2 (true T / T-dagger nodes).

    Returns
    -------
    dict with keys:
      "t_count"        : int, total number of T-nodes.
      "t_depth"        : int, number of layers in the T-sub-DAG (0 if none).
      "max_factories_needed" : int, width of the widest T-depth layer.
      "t_depth_layers" : list of lists of vertex ids; t_depth_layers[i] is
                         the set of T-node ids assigned to T-depth layer i.
    """
    if t_predicate is None:
        def t_predicate(g, v):
            if g.type(v) != VertexType.Z:
                return False
            phase = g.phase(v)
            if phase is None:
                return False
            frac = Fraction(phase % 2).limit_denominator(64)
            return frac.denominator == 4

    t_nodes = [v for v in g.vertices() if t_predicate(g, v)]
    t_node_set = set(t_nodes)

    if not t_nodes:
        return {
            "t_count": 0,
            "t_depth": 0,
            "max_factories_needed": 0,
            "t_depth_layers": [],
        }

    topo_order = sorted(g.vertices(), key=lambda v: (g.row(v), g.qubit(v)))
    topo_rank = {v: i for i, v in enumerate(topo_order)}

    def predecessors_in_graph(v):
        return [u for u in g.neighbors(v) if topo_rank[u] < topo_rank[v]]

    t_dag_pred = {t: set() for t in t_nodes}
    t_dag_succ = {t: set() for t in t_nodes}

    for t in t_nodes:
        visited = set()
        frontier = deque(predecessors_in_graph(t))
        visited.update(frontier)

        while frontier:
            u = frontier.popleft()

            if u in t_node_set:
                t_dag_pred[t].add(u)
                t_dag_succ[u].add(t)
                continue

            for p in predecessors_in_graph(u):
                if p not in visited:
                    visited.add(p)
                    frontier.append(p)

    remaining_in_degree = {t: len(t_dag_pred[t]) for t in t_nodes}
    t_layer = {}

    queue = deque(t for t in t_nodes if remaining_in_degree[t] == 0)
    for t in queue:
        t_layer[t] = 0

    while queue:
        u = queue.popleft()
        u_layer = t_layer[u]

        for v in t_dag_succ[u]:
            t_layer[v] = max(t_layer.get(v, 0), u_layer + 1)
            remaining_in_degree[v] -= 1
            if remaining_in_degree[v] == 0:
                queue.append(v)

    t_depth = max(t_layer.values()) + 1 if t_layer else 0

    t_depth_layers = [[] for _ in range(t_depth)]
    for t, l in t_layer.items():
        t_depth_layers[l].append(t)

    for layer in t_depth_layers:
        layer.sort(key=lambda v: (g.qubit(v), g.row(v)))

    max_factories_needed = max((len(layer) for layer in t_depth_layers), default=0)

    return {
        "t_count": len(t_nodes),
        "t_depth": t_depth,
        "max_factories_needed": max_factories_needed,
        "t_depth_layers": t_depth_layers,
    }