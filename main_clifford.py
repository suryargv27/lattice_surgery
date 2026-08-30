import time
import numpy as np
from collections import defaultdict
import pyzx as zx

from simplify import create_hboxes, simplify_green_spiders, assign_layers
from pft import track_pauli_frames
from layout import build_qubit_interaction_stats
from h_routing import assign_node_orientations
from tracker import QubitTracker
from spectral_layout import spectral_data_placement
from layer_processing import (
    process_boundary_nodes,
    process_h_nodes,
    process_s_nodes,
    process_cnot_nodes,
)
from reporting import print_volume_report, visualize_circuit
from column_fill import fill_qubit_columns

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


def build_cnot_weight_matrix(qubits, cnot_weight):
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

    g = zx.generate.CNOT_HAD_PHASE_circuit(3, 30,seed=42).to_graph()  # p_t=0 -> Clifford-only
    g = create_hboxes(g)
    g = simplify_green_spiders(g)
    zx.draw(g)

    track_pauli_frames(g)
    layer_map = assign_layers(g)

    qubits = sorted({g.qubit(v) for v in g.vertices() if g.qubit(v) >= 0})
    n = len(qubits)

    cnot_weight, _ = build_qubit_interaction_stats(g)
    W = build_cnot_weight_matrix(qubits, cnot_weight)
    t_demand = np.zeros(n)  # no T-gates; placement ignores t anyway

    if S is None:
        S = generate_test_S()

    layout_result = spectral_data_placement(S=S, n=n, t=t_demand, W=W)

    qubit_xy = {
        qubits[i]: (int(pos[0]), int(pos[1]))
        for i, pos in layout_result["data_positions"].items()
    }

    orientations = assign_node_orientations(g)
    zx.draw(g)

    layers = defaultdict(list)
    for v, l_idx in layer_map.items():
        layers[l_idx].append(v)

    tracker = QubitTracker(qubit_xy)

    cube_objects = []
    t_positions = set()  # stays empty; kept only for print_volume_report's shape

    for l_idx in sorted(layers.keys()):
        layer_nodes = layers[l_idx]
        print(f"Processing Layer : {l_idx}")

        process_boundary_nodes(g, layer_nodes, qubit_xy, tracker, cube_objects)
        process_h_nodes(g, layer_nodes, qubit_xy, tracker, cube_objects)
        process_s_nodes(g, layer_nodes, qubit_xy, tracker, orientations, cube_objects)
        process_cnot_nodes(g, layer_nodes, qubit_xy, tracker, orientations, cube_objects)

    cube_objects.extend(fill_qubit_columns(qubit_xy, tracker, cube_objects))
    total_runtime = time.perf_counter() - start_time
    print_volume_report(tracker, t_positions, total_runtime)

    final_z = max((pos[2] for pos in tracker.placed_obstacles), default=0)

    if visualize:
        visualize_circuit(cube_objects, final_z=final_z)


if __name__ == "__main__":
    main(visualize=True)