import matplotlib.pyplot as plt
from visualize import set_axes_equal
from cnot_routing import draw_path
from geometry import compute_bounding_volume
from cultivation import build_all_cultivation_columns
from distillation import build_all_factory_columns


def print_volume_report(tracker, t_positions, total_runtime):
    non_t_positions = tracker.placed_obstacles - t_positions
    volume_stats = compute_bounding_volume(non_t_positions)

    print("\n" + "=" * 60)
    print(f"Total runtime: {total_runtime:.3f} seconds")
    if volume_stats:
        print(
            f"Bounding volume (excluding T nodes): "
            f"{volume_stats['dx']} x {volume_stats['dy']} x {volume_stats['dz']} "
            f"= {volume_stats['volume']} unit cells"
        )
        print(f"  Bounding box min: {volume_stats['min']}")
        print(f"  Bounding box max: {volume_stats['max']}")
        print(f"  Actual placed cell count (non-T): {volume_stats['cell_count']}")
    else:
        print("No non-T geometry was placed.")
    print("=" * 60 + "\n")

    return volume_stats


def visualize_circuit(cube_objects, cultivation_registry=None, factory_registry=None, final_z=0):
    if cultivation_registry is not None and factory_registry is not None:
        raise ValueError(
            "visualize_circuit received both cultivation_registry and "
            "factory_registry -- pass exactly one, matching whichever "
            "mechanic produced cube_objects."
        )

    if cultivation_registry is not None:
        cube_objects = cube_objects + build_all_cultivation_columns(
            cultivation_registry, final_z
        )
    elif factory_registry is not None:
        cube_objects = cube_objects + build_all_factory_columns(
            factory_registry, final_z
        )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for item in cube_objects:
        if isinstance(item, tuple) and item[0] == "path":
            _, path, chosen_start, chosen_goal = item
            draw_path(path, chosen_start, chosen_goal, ax)
        else:
            item.draw(ax)

    set_axes_equal(ax)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Full 3D Layered Quantum Circuit Routing")
    plt.show()