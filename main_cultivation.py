import time
from collections import defaultdict
import pyzx as zx
from pyzx.utils import VertexType

from simplify import create_hboxes, simplify_green_spiders, assign_layers
from pft import track_pauli_frames
from layout import simulated_annealing_layout, compute_t_depth
from h_routing import assign_node_orientations
from create_rings import compute_layout_regions, plot_layout_regions
from tracker import QubitTracker
from cultivation import CultivationRegistry
from layer_processing import (
    process_boundary_nodes,
    process_h_nodes,
    process_s_nodes,
    process_t_nodes,
    process_cnot_nodes,
)
from reporting import print_volume_report, visualize_circuit


def main(visualize=True):
    start_time = time.perf_counter()

    g = zx.generate.cliffordT(6, 30, p_hsh=0)
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

    qubit_xy = simulated_annealing_layout(g, step=2, iterations=20000, seed=42)
    regions = compute_layout_regions(qubit_xy)
    plot_layout_regions(regions)

    orientations = assign_node_orientations(g)
    zx.draw(g)

    layers = defaultdict(list)
    for v, l_idx in layer_map.items():
        layers[l_idx].append(v)

    tracker = QubitTracker(qubit_xy, cultivation_sites=regions["cultivation"])
    cultivation_registry = CultivationRegistry(
        regions["cultivation"], code_distance=17, lam=0.00227, start_z=0
    )
    cube_objects = []
    t_positions = set()

    for l_idx in sorted(layers.keys()):
        layer_nodes = layers[l_idx]
        print(f"Processing Layer : {l_idx}")

        process_boundary_nodes(g, layer_nodes, qubit_xy, tracker, cube_objects)
        process_h_nodes(g, layer_nodes, qubit_xy, tracker, cube_objects)
        process_s_nodes(g, layer_nodes, qubit_xy, tracker, orientations, cube_objects)
        process_t_nodes(
            g,
            layer_nodes,
            qubit_xy,
            tracker,
            orientations,
            regions,
            cultivation_registry,
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
            cube_objects, cultivation_registry=cultivation_registry, final_z=final_z
        )


if __name__ == "__main__":
    main(visualize=True)
