import time
import numpy as np
from collections import defaultdict
import pyzx as zx
from pyzx.utils import VertexType

from simplify import create_hboxes, simplify_green_spiders, assign_layers
from pft import track_pauli_frames, is_pi_4_phase
from layout import compute_t_depth, build_qubit_interaction_stats
from h_routing import assign_node_orientations
from tracker import QubitTracker
from distillation import DistillationFactoryRegistry
from spectral_layout import spectral_data_placement, solve_max_factories
from layer_processing import (
    process_boundary_nodes,
    process_h_nodes,
    process_s_nodes,
    process_t_nodes_distillation,
    process_cnot_nodes,
)
from reporting import print_volume_report, visualize_circuit


def generate_test_S():
    """
    Test-only square grid generator for spectral_data_placement's S input.
    Sized off n so there's comfortably more than n valid data candidates.
    """
    S = {
        (x, y)
        for x in range(10)
        for y in range(10)
    }

    return S


def build_t_demand_vector(qubits, t_count):
    """Per-qubit T-count vector, aligned to the same qubits ordering used for W."""
    return np.array([t_count.get(q, 0) for q in qubits])


def build_cnot_weight_matrix(qubits, cnot_weight):
    """Dense n x n CNOT interaction matrix, row/col i <-> qubits[i]."""
    n = len(qubits)
    idx = {q: i for i, q in enumerate(qubits)}
    W = np.zeros((n, n))
    for (a, b), w in cnot_weight.items():
        if a in idx and b in idx:
            i, j = idx[a], idx[b]
            W[i, j] += w
            W[j, i] += w
    return W


def main(visualize=True, S=None):
    start_time = time.perf_counter()

    g = zx.generate.cliffordT(6, 50, p_hsh=0)
    g = create_hboxes(g)
    g = simplify_green_spiders(g)
    zx.draw(g)

    final_pauli_records, pending_flushes = track_pauli_frames(g)
    layer_map = assign_layers(g)

    t_stats = compute_t_depth(g)
    print(f"T-count: {t_stats['t_count']}")
    print(f"T-depth: {t_stats['t_depth']}")
    print(f"Max factories needed: {t_stats['max_factories_needed']}")
    for i, layer in enumerate(t_stats["t_depth_layers"]):
        print(f"  Layer {i}: {layer}")

    # --- extract n, t, W for the spectral layout pipeline ---
    qubits = sorted({g.qubit(v) for v in g.vertices() if g.qubit(v) >= 0})
    n = len(qubits)

    cnot_weight, t_count = build_qubit_interaction_stats(g)
    W = build_cnot_weight_matrix(qubits, cnot_weight)
    t_demand = build_t_demand_vector(qubits, t_count)

    if S is None:
        S = generate_test_S()

    # --- spectral data placement (replaces simulated_annealing_layout) ---
    layout_result = spectral_data_placement(S=S, n=n, t=t_demand, W=W)

    qubit_xy = {
        qubits[i]: (int(pos[0]), int(pos[1]))
        for i, pos in layout_result["data_positions"].items()
    }

    regions = {
        "bus": layout_result["bus"],
        "routing": layout_result["ancilla"],
        "data": layout_result["D"],
    }

    # --- distillation factory placement (replaces cultivation regions) ---
    factory_result = solve_max_factories(
        S=layout_result["S"], bus=layout_result["bus"]
    )

    if factory_result["num_factories"] == 0:
        raise RuntimeError(
            "No distillation factories could be placed for this S/bus -- "
            "increase grid size or padding in generate_test_S()."
        )

    print(
        f"Placed {factory_result['num_factories']} distillation factories "
        f"out of {factory_result['candidate_count']} candidates"
    )

    factory_cells = {
        cell for f in factory_result["factories"] for cell in f["cells"]
    }

    orientations = assign_node_orientations(g)
    zx.draw(g)

    layers = defaultdict(list)
    for v, l_idx in layer_map.items():
        layers[l_idx].append(v)

    tracker = QubitTracker(qubit_xy, extra_columns=factory_cells)
    factory_registry = DistillationFactoryRegistry(
        factory_result["factories"], start_z=0
    )

    cube_objects = []
    t_positions = set()

    for l_idx in sorted(layers.keys()):
        layer_nodes = layers[l_idx]
        print(f"Processing Layer : {l_idx}")

        process_boundary_nodes(g, layer_nodes, qubit_xy, tracker, cube_objects)
        process_h_nodes(g, layer_nodes, qubit_xy, tracker, cube_objects)
        process_s_nodes(g, layer_nodes, qubit_xy, tracker, orientations, cube_objects)
        process_t_nodes_distillation(
            g,
            layer_nodes,
            qubit_xy,
            tracker,
            orientations,
            regions,
            factory_registry,
            cube_objects,
            t_positions,
        )
        process_cnot_nodes(
            g, layer_nodes, qubit_xy, tracker, orientations, cube_objects
        )

    total_runtime = time.perf_counter() - start_time
    print_volume_report(tracker, t_positions, total_runtime)

    final_z = max((pos[2] for pos in tracker.placed_obstacles), default=0)

    if visualize:
        visualize_circuit(
            cube_objects, factory_registry=factory_registry, final_z=final_z
        )


if __name__ == "__main__":
    main(visualize=True)