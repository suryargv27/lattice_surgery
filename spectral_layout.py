import numpy as np
import matplotlib.pyplot as plt

from collections import deque
from scipy.optimize import linear_sum_assignment

try:
    from ortools.sat.python import cp_model
except ImportError as exc:
    raise ImportError(
        "OR-Tools is required. Install it with:\n"
        "    pip install ortools"
    ) from exc


# ============================================================
# CONSTANTS
# ============================================================

MOORE_OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

CARDINAL_OFFSETS = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
]


# ============================================================
# BASIC GRID HELPERS
# ============================================================

def moore_neighbors(point):
    """Return the eight Moore neighbors of a grid point."""
    x, y = point

    return {
        (x + dx, y + dy)
        for dx, dy in MOORE_OFFSETS
    }


def cardinal_neighbors(point):
    """Return the four 4-connected neighbors of a grid point."""
    x, y = point

    return {
        (x + dx, y + dy)
        for dx, dy in CARDINAL_OFFSETS
    }


def points_to_array(points):
    """Convert a collection of points into an Nx2 NumPy array."""
    if not points:
        return np.empty((0, 2), dtype=float)

    return np.asarray(sorted(points), dtype=float)


# ============================================================
# 1. VALID DATA CANDIDATES
# ============================================================

def valid_data_candidates(S):
    """
    Partition S into four parity classes.

    A point is a valid data candidate when all eight of its Moore
    neighbors are also contained in S.

    Return the largest valid parity class.
    """
    S = set(S)

    parity_sets = {
        (0, 0): [],
        (0, 1): [],
        (1, 0): [],
        (1, 1): [],
    }

    for x, y in S:
        parity_sets[(x % 2, y % 2)].append((x, y))

    valid_sets = {}

    for parity, points in parity_sets.items():

        valid_sets[parity] = {
            point
            for point in points
            if all(
                neighbor in S
                for neighbor in moore_neighbors(point)
            )
        }

    return max(valid_sets.values(), key=len)


# ============================================================
# 2. CENTROID AND CENTRAL CANDIDATE
# ============================================================

def candidate_centroid(C):
    """Return the centroid of candidate set C."""
    C_array = np.asarray(list(C), dtype=float)

    if len(C_array) == 0:
        raise ValueError("Candidate set C is empty.")

    return C_array.mean(axis=0)


def nearest_candidate_to_centroid(C, centroid):
    """Return the candidate in C closest to centroid."""
    cx, cy = centroid

    return min(
        C,
        key=lambda point: (
            (point[0] - cx) ** 2
            + (point[1] - cy) ** 2
        ),
    )


# ============================================================
# 3. CANDIDATE GRAPH
# ============================================================

def candidate_neighbors(point, C):
    """
    Return candidate neighbors at distance 2 in the four
    cardinal directions.
    """
    x, y = point

    possible_neighbors = [
        (x + 2, y),
        (x - 2, y),
        (x, y + 2),
        (x, y - 2),
    ]

    return [
        neighbor
        for neighbor in possible_neighbors
        if neighbor in C
    ]


# ============================================================
# 4. COMPACT BFS SET D
# ============================================================

def compact_bfs_set(C, start, n):
    """
    Run BFS on the candidate graph and return the first n points.

    IMPORTANT:
    The returned LIST preserves a deterministic ordering. This ordering
    is later used directly by the Hungarian algorithm and MUST NOT be
    replaced by a set ordering.
    """
    if start not in C:
        raise ValueError(
            "Start point is not in the candidate set."
        )

    visited = {start}
    ordered_D = [start]
    queue = deque([start])

    while queue and len(ordered_D) < n:

        point = queue.popleft()

        for neighbor in sorted(candidate_neighbors(point, C)):

            if neighbor not in visited:

                visited.add(neighbor)
                ordered_D.append(neighbor)
                queue.append(neighbor)

                if len(ordered_D) == n:
                    break

    if len(ordered_D) < n:
        raise ValueError(
            f"Only {len(ordered_D)} candidate points are reachable, "
            f"but n={n}."
        )

    return ordered_D


# ============================================================
# 5. SPECTRAL EMBEDDING
# ============================================================

def spectral_embedding(W, dim=2):
    """
    Compute a dim-dimensional spectral embedding using the first
    nontrivial eigenvectors of the graph Laplacian.
    """
    W = np.asarray(W, dtype=float)

    if W.ndim != 2:
        raise ValueError("W must be a 2D matrix.")

    if W.shape[0] != W.shape[1]:
        raise ValueError("W must be square.")

    n = W.shape[0]

    if n <= dim:
        raise ValueError(
            f"Need n > {dim} for a {dim}D embedding."
        )

    # Symmetrize.
    W = 0.5 * (W + W.T)

    # Remove self-interactions.
    W = W.copy()
    np.fill_diagonal(W, 0.0)

    degree = W.sum(axis=1)
    laplacian = np.diag(degree) - W

    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)

    # Skip the trivial constant eigenvector.
    Z = eigenvectors[:, 1:dim + 1]

    return Z, eigenvalues


# ============================================================
# 6. CENTER AND SCALE
# ============================================================

def center_and_scale(X):
    """
    Center X at the origin and scale it to unit RMS radius.
    """
    X = np.asarray(X, dtype=float)

    center = X.mean(axis=0)
    X_centered = X - center

    scale = np.sqrt(
        np.mean(
            np.sum(X_centered ** 2, axis=1)
        )
    )

    if scale < 1e-12:
        scale = 1.0

    X_normalized = X_centered / scale

    return X_normalized, center, scale


# ============================================================
# 7. PRINCIPAL AXES
# ============================================================

def principal_axes(X):
    """Return PCA axes of X."""
    _, _, Vt = np.linalg.svd(
        X,
        full_matrices=False,
    )

    return Vt.T


# ============================================================
# 8. ALIGN SPECTRAL COORDINATES
# ============================================================

def align_spectral_to_physical(Z, D_coords):
    """
    Align spectral coordinates with physical D coordinates.

    Several rotations/reflections are generated. Hungarian matching is
    performed for each orientation and the minimum-cost orientation is
    selected.
    """
    Z_norm, _, _ = center_and_scale(Z)
    D_norm, _, _ = center_and_scale(D_coords)

    Rz = principal_axes(Z_norm)
    Rd = principal_axes(D_norm)

    Z_base = Z_norm @ Rz @ Rd.T

    transforms = {
        "identity":
            np.array([[1, 0], [0, 1]], dtype=float),

        "flip_x":
            np.array([[-1, 0], [0, 1]], dtype=float),

        "flip_y":
            np.array([[1, 0], [0, -1]], dtype=float),

        "flip_xy":
            np.array([[-1, 0], [0, -1]], dtype=float),

        "swap":
            np.array([[0, 1], [1, 0]], dtype=float),

        "swap_flip_x":
            np.array([[0, -1], [1, 0]], dtype=float),

        "swap_flip_y":
            np.array([[0, 1], [-1, 0]], dtype=float),

        "swap_flip_xy":
            np.array([[0, -1], [-1, 0]], dtype=float),
    }

    alignment_candidates = []

    for name, transform in transforms.items():

        Z_aligned = Z_base @ transform

        alignment_candidates.append(
            (name, Z_aligned)
        )

    return alignment_candidates, D_norm


# ============================================================
# 9. HUNGARIAN MATCHING
# ============================================================

def hungarian_match(Z_aligned, D_normalized):
    """
    Compute the minimum-cost one-to-one matching.

    Cost[i, k] = ||Z_aligned[i] - D_normalized[k]||^2

    assignment[i] = k means:

        spectral/logical point i
            ->
        physical D point D_ordered[k]

    Therefore D_normalized MUST preserve the exact D_ordered ordering.
    """
    differences = (
        Z_aligned[:, np.newaxis, :]
        - D_normalized[np.newaxis, :, :]
    )

    cost_matrix = np.sum(
        differences ** 2,
        axis=2,
    )

    rows, cols = linear_sum_assignment(cost_matrix)

    assignment = np.empty(
        len(rows),
        dtype=int,
    )

    assignment[rows] = cols

    total_cost = cost_matrix[rows, cols].sum()

    return assignment, cost_matrix, total_cost


# ============================================================
# 10. SPLIT ORIGINAL LATTICE
# ============================================================

def build_lattice_sets(original_S, D):
    """
    Split original lattice into four disjoint sets:

        D
        bus
        ancilla
        S

    Steps:
        1. Find Moore-neighbor shell around D.
        2. Points in this shell that have a Moore neighbor outside
           shell ∪ D become bus.
        3. Remaining shell points become ancilla.
        4. Everything else becomes S.
    """
    original_S = set(original_S)
    D = set(D)

    if not D.issubset(original_S):
        raise ValueError(
            "All D points must belong to original_S."
        )

    temp_set = set()

    for point in D:

        for neighbor in moore_neighbors(point):

            if (
                neighbor in original_S
                and neighbor not in D
            ):
                temp_set.add(neighbor)

    union_set = temp_set | D

    bus = set()
    ancilla = set()

    for point in temp_set:

        has_neighbor_outside_union = any(
            neighbor not in union_set
            for neighbor in moore_neighbors(point)
        )

        if has_neighbor_outside_union:
            bus.add(point)
        else:
            ancilla.add(point)

    S = (
        original_S
        - D
        - bus
        - ancilla
    )

    return {
        "S": S,
        "D": D,
        "bus": bus,
        "ancilla": ancilla,
        "temp_set": temp_set,
        "union_set": union_set,
    }


# ============================================================
# 11. COMPLETE DATA PLACEMENT
# ============================================================

def spectral_data_placement(S, n, t, W):
    """
    Complete pipeline:

        original lattice
            ->
        valid candidates
            ->
        compact ordered D
            ->
        spectral embedding
            ->
        alignment
            ->
        Hungarian matching
            ->
        split into S, ancilla, bus, D

    IMPORTANT:
    D_ordered is preserved separately because its order is the exact
    order used by the Hungarian cost matrix and assignment.
    """
    original_S = set(S)

    t = np.asarray(t)
    W = np.asarray(W, dtype=float)

    if len(t) != n:
        raise ValueError("t must have length n.")

    if W.shape != (n, n):
        raise ValueError(
            f"W must have shape ({n}, {n})."
        )

    # --------------------------------------------------------
    # Step 1: valid candidate set.
    # --------------------------------------------------------

    C = valid_data_candidates(original_S)

    if len(C) < n:
        raise ValueError(
            f"Only {len(C)} valid candidates are available, "
            f"but n={n}."
        )

    # --------------------------------------------------------
    # Step 2: centroid and central candidate.
    # --------------------------------------------------------

    centroid = candidate_centroid(C)

    c0 = nearest_candidate_to_centroid(
        C,
        centroid,
    )

    # --------------------------------------------------------
    # Step 3: compact D.
    #
    # D_ordered is a LIST.
    # Its ordering is critical for Hungarian matching.
    # --------------------------------------------------------

    D_ordered = compact_bfs_set(
        C=C,
        start=c0,
        n=n,
    )

    D_coords = np.asarray(
        D_ordered,
        dtype=float,
    )

    # --------------------------------------------------------
    # Step 4: spectral embedding.
    # --------------------------------------------------------

    Z, eigenvalues = spectral_embedding(
        W,
        dim=2,
    )

    # --------------------------------------------------------
    # Step 5: generate physical/spectral alignments.
    # --------------------------------------------------------

    alignment_candidates, D_normalized = (
        align_spectral_to_physical(
            Z,
            D_coords,
        )
    )

    # --------------------------------------------------------
    # Step 6: Hungarian matching for every orientation.
    # --------------------------------------------------------

    best = None

    for orientation, Z_aligned in alignment_candidates:

        assignment, cost_matrix, matching_cost = (
            hungarian_match(
                Z_aligned,
                D_normalized,
            )
        )

        if (
            best is None
            or matching_cost < best["matching_cost"]
        ):
            best = {
                "orientation": orientation,
                "Z_aligned": Z_aligned,
                "assignment": assignment,
                "cost_matrix": cost_matrix,
                "matching_cost": matching_cost,
            }

    assignment = best["assignment"]

    # --------------------------------------------------------
    # Step 7: correct logical -> physical mapping.
    #
    # CRITICAL:
    #
    # assignment[i] indexes D_ordered, NOT a sorted D set.
    # --------------------------------------------------------

    data_positions = {
        i: D_ordered[assignment[i]]
        for i in range(n)
    }

    # --------------------------------------------------------
    # Step 8: split lattice.
    # --------------------------------------------------------

    lattice_sets = build_lattice_sets(
        original_S,
        D_ordered,
    )

    return {

        # Candidate information.
        "C": C,
        "centroid": centroid,
        "c0": c0,

        # Final lattice sets.
        "S": lattice_sets["S"],
        "D": lattice_sets["D"],
        "bus": lattice_sets["bus"],
        "ancilla": lattice_sets["ancilla"],

        # IMPORTANT:
        # Exact order used by Hungarian matching.
        "D_ordered": D_ordered,

        # Intermediate sets.
        "temp_set": lattice_sets["temp_set"],
        "union_set": lattice_sets["union_set"],

        # Spectral / Hungarian information.
        "spectral_coords": Z,
        "aligned_spectral_coords": best["Z_aligned"],
        "D_normalized": D_normalized,
        "assignment": assignment,
        "cost_matrix": best["cost_matrix"],
        "data_positions": data_positions,
        "matching_cost": best["matching_cost"],
        "orientation": best["orientation"],
        "eigenvalues": eigenvalues,
    }


# ============================================================
# 12. FACTORY GEOMETRY
# ============================================================

def factory_cells(center):
    """
    Return all nine cells of the 3x3 factory centered at center.
    """
    x, y = center

    return {
        (x + dx, y + dy)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
    }


def factory_outputs(center):
    """
    Return the eight outer cells of a 3x3 factory.
    """
    return (
        factory_cells(center)
        - {center}
    )


def valid_factory_candidates(S):
    """
    Find all valid 3x3 factories completely contained in S.
    """
    S = set(S)

    candidates = []

    for center in sorted(S):

        cells = factory_cells(center)

        if cells.issubset(S):

            candidates.append({
                "center": center,
                "cells": cells,
                "outputs": factory_outputs(center),
            })

    return candidates


# ============================================================
# 13. FACTORY REACHABILITY
# ============================================================

def reachable_bus_for_factory(
    factory,
    selected_factories,
    S,
    bus,
):
    """
    Determine whether one output of this factory can reach a bus point
    through 4-connectivity.

    Routing rules:
        - factory cells are occupied;
        - factory centers cannot be used as routers;
        - a factory's own 8 outputs are valid signal sources;
        - empty S cells are routers;
        - bus cells are absorbing endpoints;
        - other factories block routing through all their cells.
    """
    S = set(S)
    bus = set(bus)

    if not bus:
        return False, None

    # All selected factory cells are occupied.
    occupied = set()

    for selected_factory in selected_factories:
        occupied.update(selected_factory["cells"])

    # This factory's outputs are legal starting points.
    current_outputs = set(factory["outputs"])

    # Empty S routers.
    empty_S = S - occupied

    # Traversable graph:
    #
    # empty S cells
    # + this factory's output cells as sources
    # + bus cells as sinks
    traversable = (
        empty_S
        | current_outputs
        | bus
    )

    # The center must never route.
    traversable.discard(factory["center"])

    # Multi-source BFS from all outputs.
    queue = deque()
    visited = set()
    parent = {}

    for output in current_outputs:

        if output in traversable:

            queue.append(output)
            visited.add(output)
            parent[output] = None

    while queue:

        point = queue.popleft()

        # Bus absorbs the flow.
        if point in bus:

            path = []

            current = point

            while current is not None:

                path.append(current)
                current = parent[current]

            path.reverse()

            return True, path

        for neighbor in cardinal_neighbors(point):

            if (
                neighbor in traversable
                and neighbor not in visited
            ):
                visited.add(neighbor)
                parent[neighbor] = point
                queue.append(neighbor)

    return False, None


def validate_factory_placement(
    selected_factories,
    S,
    bus,
):
    """
    Every selected factory must have at least one output-to-bus route.
    """
    routes = {}

    for factory in selected_factories:

        reachable, path = reachable_bus_for_factory(
            factory=factory,
            selected_factories=selected_factories,
            S=S,
            bus=bus,
        )

        if not reachable:
            return False, {}

        routes[factory["center"]] = path

    return True, routes


# ============================================================
# 14. MAXIMUM FACTORY PLACEMENT
# ============================================================

def solve_max_factories(
    S,
    bus,
    max_iterations=10000,
):
    """
    Maximize the number of valid factories.

    CP-SAT handles:
        1. one Boolean variable per candidate factory;
        2. no overlapping 3x3 factories;
        3. maximize number of selected factories.

    After each solution:
        - validate output-to-bus reachability;
        - if invalid, add a no-good cut;
        - solve again.

    The first reachability-valid solution returned at the current
    optimum is the maximum valid placement under these constraints.
    """
    S = set(S)
    bus = set(bus)

    candidates = valid_factory_candidates(S)

    if not candidates:

        return {
            "factories": [],
            "routes": {},
            "num_factories": 0,
            "candidate_count": 0,
            "iterations": 0,
        }

    model = cp_model.CpModel()

    x = [
        model.NewBoolVar(f"factory_{i}")
        for i in range(len(candidates))
    ]

    # --------------------------------------------------------
    # Non-overlap constraints.
    #
    # For every grid cell:
    #
    # sum(factory_i containing this cell) <= 1
    # --------------------------------------------------------

    cell_to_candidates = {}

    for i, candidate in enumerate(candidates):

        for cell in candidate["cells"]:

            cell_to_candidates.setdefault(
                cell,
                [],
            ).append(i)

    for indices in cell_to_candidates.values():

        if len(indices) > 1:

            model.Add(
                sum(x[i] for i in indices) <= 1
            )

    # --------------------------------------------------------
    # Objective.
    # --------------------------------------------------------

    model.Maximize(
        sum(x)
    )

    solver = cp_model.CpSolver()

    solver.parameters.num_search_workers = 8

    # --------------------------------------------------------
    # Solve -> validate reachability -> add cuts if necessary.
    # --------------------------------------------------------

    for iteration in range(1, max_iterations + 1):

        status = solver.Solve(model)

        if status not in (
            cp_model.OPTIMAL,
            cp_model.FEASIBLE,
        ):

            return {
                "factories": [],
                "routes": {},
                "num_factories": 0,
                "candidate_count": len(candidates),
                "iterations": iteration,
            }

        selected_indices = [
            i
            for i in range(len(candidates))
            if solver.Value(x[i]) == 1
        ]

        selected_factories = [
            candidates[i]
            for i in selected_indices
        ]

        valid, routes = validate_factory_placement(
            selected_factories=selected_factories,
            S=S,
            bus=bus,
        )

        # Valid maximum solution.
        if valid:

            return {
                "factories": selected_factories,
                "routes": routes,
                "num_factories": len(selected_factories),
                "candidate_count": len(candidates),
                "iterations": iteration,
            }

        # ----------------------------------------------------
        # No-good cut.
        #
        # The exact currently selected invalid combination cannot
        # occur again.
        #
        # At least one currently selected factory must be removed.
        # ----------------------------------------------------

        model.Add(
            sum(x[i] for i in selected_indices)
            <= len(selected_indices) - 1
        )

    raise RuntimeError(
        "Maximum factory solver iterations exceeded."
    )


# ============================================================
# 15. PLOT 1: CORRECTED HUNGARIAN MATCHING
# ============================================================

def plot_hungarian_matching(result):
    """
    Plot the ACTUAL Hungarian mapping.

    IMPORTANT:
    D_ordered is used directly.

    This guarantees:

        assignment[i] = k

    is plotted as:

        spectral point i
            ->
        D_ordered[k]

    No sorting of D is performed here.
    """
    Z = result["aligned_spectral_coords"]

    # CRITICAL FIX:
    # Use exactly the same ordering as the Hungarian cost matrix.
    D_coords = np.asarray(
        result["D_ordered"],
        dtype=float,
    )

    D_normalized = result["D_normalized"]

    assignment = result["assignment"]
    n = len(assignment)

    fig, ax = plt.subplots(
        figsize=(9, 9)
    )

    # --------------------------------------------------------
    # Draw actual matching edges.
    # --------------------------------------------------------

    for i in range(n):

        k = assignment[i]

        ax.plot(
            [Z[i, 0], D_normalized[k, 0]],
            [Z[i, 1], D_normalized[k, 1]],
            color="gray",
            linewidth=1.0,
            alpha=0.7,
            zorder=1,
        )

    # --------------------------------------------------------
    # Spectral points.
    # --------------------------------------------------------

    ax.scatter(
        Z[:, 0],
        Z[:, 1],
        s=100,
        color="red",
        label="Spectral points",
        zorder=2,
    )

    # --------------------------------------------------------
    # D points.
    #
    # D_normalized[k] corresponds exactly to D_ordered[k].
    # --------------------------------------------------------

    ax.scatter(
        D_normalized[:, 0],
        D_normalized[:, 1],
        s=130,
        color="yellow",
        edgecolors="black",
        linewidths=0.8,
        label="D points",
        zorder=3,
    )

    # --------------------------------------------------------
    # Label each physical D point with its logical index.
    # --------------------------------------------------------

    for i in range(n):

        k = assignment[i]

        x, y = D_normalized[k]

        ax.text(
            x,
            y,
            str(i),
            color="black",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            zorder=4,
        )

    ax.set_title(
        "Hungarian Matching"
    )

    ax.set_xlabel("Coordinate 1")
    ax.set_ylabel("Coordinate 2")

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.grid(False)
    ax.legend()

    plt.tight_layout()
    plt.show()


# ============================================================
# 16. PLOT 2: S, ANCILLA, BUS, D
# ============================================================

def plot_lattice_sets(result):
    """
    Plot the final lattice partition.
    """
    S_coords = points_to_array(
        result["S"]
    )

    ancilla_coords = points_to_array(
        result["ancilla"]
    )

    bus_coords = points_to_array(
        result["bus"]
    )

    D_coords = points_to_array(
        result["D"]
    )

    fig, ax = plt.subplots(
        figsize=(10, 10)
    )

    # S
    if len(S_coords):

        ax.scatter(
            S_coords[:, 0],
            S_coords[:, 1],
            s=25,
            color="orange",
            label="S",
            zorder=1,
        )

    # Ancilla
    if len(ancilla_coords):

        ax.scatter(
            ancilla_coords[:, 0],
            ancilla_coords[:, 1],
            s=70,
            color="darkorange",
            edgecolors="black",
            linewidths=0.4,
            label="ancilla",
            zorder=2,
        )

    # Bus
    if len(bus_coords):

        ax.scatter(
            bus_coords[:, 0],
            bus_coords[:, 1],
            s=80,
            color="red",
            edgecolors="black",
            linewidths=0.5,
            label="bus",
            zorder=3,
        )

    # D
    if len(D_coords):

        ax.scatter(
            D_coords[:, 0],
            D_coords[:, 1],
            s=150,
            color="yellow",
            edgecolors="black",
            linewidths=0.7,
            label="D",
            zorder=4,
        )

    ax.set_title(
        "Lattice Partition: S, Ancilla, Bus, D"
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.grid(False)
    ax.legend()

    plt.tight_layout()
    plt.show()


# ============================================================
# 17. PLOT 3: FACTORIES
# ============================================================

def plot_factories(
    result,
    factory_result,
    show_routes=True,
):
    """
    Plot the lattice and selected factories.

    Colors:
        S               -> orange
        ancilla         -> dark orange
        bus             -> red
        D               -> yellow
        factory block   -> transparent magenta
        factory border  -> magenta
        factory outputs -> magenta outline
        factory center  -> solid magenta
        route           -> green
    """
    S = result["S"]
    ancilla = result["ancilla"]
    bus = result["bus"]
    D = result["D"]

    factories = factory_result["factories"]
    routes = factory_result["routes"]

    S_coords = points_to_array(S)
    ancilla_coords = points_to_array(ancilla)
    bus_coords = points_to_array(bus)
    D_coords = points_to_array(D)

    fig, ax = plt.subplots(
        figsize=(12, 11)
    )

    # --------------------------------------------------------
    # Background lattice.
    # --------------------------------------------------------

    if len(S_coords):

        ax.scatter(
            S_coords[:, 0],
            S_coords[:, 1],
            s=20,
            color="orange",
            alpha=0.65,
            label="S",
            zorder=1,
        )

    if len(ancilla_coords):

        ax.scatter(
            ancilla_coords[:, 0],
            ancilla_coords[:, 1],
            s=65,
            color="darkorange",
            edgecolors="black",
            linewidths=0.4,
            label="ancilla",
            zorder=2,
        )

    if len(bus_coords):

        ax.scatter(
            bus_coords[:, 0],
            bus_coords[:, 1],
            s=80,
            color="red",
            edgecolors="black",
            linewidths=0.5,
            label="bus",
            zorder=3,
        )

    if len(D_coords):

        ax.scatter(
            D_coords[:, 0],
            D_coords[:, 1],
            s=140,
            color="yellow",
            edgecolors="black",
            linewidths=0.7,
            label="D",
            zorder=4,
        )

    # --------------------------------------------------------
    # Draw witness routes.
    # --------------------------------------------------------

    if show_routes:

        first_route = True

        for center, path in routes.items():

            if len(path) < 2:
                continue

            path_coords = np.asarray(
                path,
                dtype=float,
            )

            ax.plot(
                path_coords[:, 0],
                path_coords[:, 1],
                color="green",
                linewidth=2.0,
                alpha=0.8,
                label=(
                    "Factory route"
                    if first_route
                    else None
                ),
                zorder=5,
            )

            first_route = False

    # --------------------------------------------------------
    # Draw factories.
    # --------------------------------------------------------

    first_factory = True

    for factory in factories:

        center = factory["center"]
        outputs = factory["outputs"]

        cx, cy = center

        # ----------------------------------------------------
        # Full 3x3 magenta square.
        #
        # Points are centered at integer coordinates.
        #
        # The occupied cells extend from:
        #
        #   cx - 1 to cx + 1
        #   cy - 1 to cy + 1
        #
        # Therefore the visual outer boundary is:
        #
        #   cx - 1.5 to cx + 1.5
        #   cy - 1.5 to cy + 1.5
        # ----------------------------------------------------

        block = plt.Rectangle(
            (cx - 1.5, cy - 1.5),
            width=3.0,
            height=3.0,
            facecolor="magenta",
            edgecolor="magenta",
            linewidth=2.5,
            alpha=0.18,
            label=(
                "Factory 3x3 block"
                if first_factory
                else None
            ),
            zorder=6,
        )

        ax.add_patch(block)

        # Factory outputs.
        output_coords = points_to_array(
            outputs
        )

        if len(output_coords):

            ax.scatter(
                output_coords[:, 0],
                output_coords[:, 1],
                s=110,
                facecolors="none",
                edgecolors="magenta",
                linewidths=2.0,
                label=(
                    "Factory outputs"
                    if first_factory
                    else None
                ),
                zorder=7,
            )

        # Factory center.
        ax.scatter(
            [cx],
            [cy],
            s=200,
            marker="s",
            color="magenta",
            edgecolors="black",
            linewidths=0.9,
            label=(
                "Factory center"
                if first_factory
                else None
            ),
            zorder=8,
        )

        first_factory = False

    # --------------------------------------------------------
    # Final formatting.
    # --------------------------------------------------------

    ax.set_title(
        f"Maximum Valid Factory Placement "
        f"({len(factories)} factories)"
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.grid(False)

    # --------------------------------------------------------
    # Put legend outside the plot.
    # --------------------------------------------------------

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )

    # Reserve space on the right for the legend.
    plt.tight_layout(
        rect=[0, 0, 0.80, 1]
    )

    plt.show()


# ============================================================
# 18. EXAMPLE
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Original lattice.
    # --------------------------------------------------------

    original_S = {
        (x, y)
        for x in range(15)
        for y in range(15)
    }

    # --------------------------------------------------------
    # Number of data qubits.
    # --------------------------------------------------------

    n = 12

    # --------------------------------------------------------
    # Demand vector.
    #
    # Retained for compatibility with the placement function.
    # --------------------------------------------------------

    t = np.array([
        3, 8, 2, 5, 10, 4,
        6, 1, 7, 3, 9, 2,
    ])

    # --------------------------------------------------------
    # Example interaction matrix.
    # --------------------------------------------------------

    rng = np.random.default_rng(42)

    W = rng.integers(
        low=0,
        high=10,
        size=(n, n),
    ).astype(float)

    W = np.triu(W, 1)
    W = W + W.T

    np.fill_diagonal(W, 0.0)

    # ========================================================
    # STAGE 1:
    # DATA PLACEMENT
    # ========================================================

    result = spectral_data_placement(
        S=original_S,
        n=n,
        t=t,
        W=W,
    )

    print("=" * 70)
    print("DATA PLACEMENT")
    print("=" * 70)

    print()
    print(
        "Best orientation:",
        result["orientation"],
    )

    print(
        "Hungarian matching cost:",
        result["matching_cost"],
    )

    print()
    print("Ordered D used by Hungarian:")
    print(result["D_ordered"])

    print()
    print("Correct logical -> physical mapping:")

    for logical_index in range(n):

        physical_index = result["assignment"][
            logical_index
        ]

        physical_point = result["D_ordered"][
            physical_index
        ]

        print(
            f"Logical {logical_index}"
            f" -> D_ordered[{physical_index}]"
            f" = {physical_point}"
        )

    print()
    print("Set sizes:")

    print(
        "S      :",
        len(result["S"]),
    )

    print(
        "ancilla:",
        len(result["ancilla"]),
    )

    print(
        "bus    :",
        len(result["bus"]),
    )

    print(
        "D      :",
        len(result["D"]),
    )

    print()
    print(
        "Partition check:",
        len(result["S"])
        + len(result["ancilla"])
        + len(result["bus"])
        + len(result["D"]),
        "==",
        len(original_S),
    )

    # ========================================================
    # PLOT 1:
    # CORRECT HUNGARIAN MATCHING
    # ========================================================

    plot_hungarian_matching(result)

    # ========================================================
    # PLOT 2:
    # S, ANCILLA, BUS, D
    # ========================================================

    plot_lattice_sets(result)

    # ========================================================
    # STAGE 2:
    # FACTORY OPTIMIZATION
    # ========================================================

    factory_result = solve_max_factories(
        S=result["S"],
        bus=result["bus"],
    )

    print()
    print("=" * 70)
    print("FACTORY PLACEMENT")
    print("=" * 70)

    print()
    print(
        "Number of valid 3x3 candidates:",
        factory_result["candidate_count"],
    )

    print(
        "Maximum valid factories:",
        factory_result["num_factories"],
    )

    print(
        "Solver / validation iterations:",
        factory_result["iterations"],
    )

    print()
    print("Factory centers:")

    for i, factory in enumerate(
        factory_result["factories"]
    ):

        center = factory["center"]

        print(
            f"Factory {i}: {center}"
        )

        route = factory_result["routes"].get(
            center
        )

        if route is not None:

            print(
                "  Witness route:",
                route,
            )

    # ========================================================
    # PLOT 3:
    # FINAL FACTORY PLACEMENT
    # ========================================================

    plot_factories(
        result=result,
        factory_result=factory_result,
        show_routes=True,
    )