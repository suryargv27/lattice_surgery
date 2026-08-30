from pyzx.utils import VertexType
from visualize import Cube
from h_routing import h_routing
from s_routing import s_routing
from t_routing import route_bus_to_ready_site, route_bus_to_ready_factory, a_star_2d
from cnot_routing import route_cnot_minimal_z
from pft import is_pi_2_phase, is_pi_4_phase, is_pi_phase

POS = 0
ORIENT = 1
DIR = 2
TYPE = 3


def process_boundary_nodes(g, layer_nodes, qubit_xy, tracker, cube_objects):
    boundary_nodes = [v for v in layer_nodes if g.type(v) == VertexType.BOUNDARY]
    for v in boundary_nodes:
        q = g.qubit(v)
        if q not in qubit_xy:
            continue
        x, y = qubit_xy[q]
        z = tracker.next_free_z[q]
        pos = (x, y, z)

        boundary_cube = Cube(pos=pos, orientation="000", dir=None, type="0")
        cube_objects.append(boundary_cube)

        tracker.record_placed([pos])


def process_h_nodes(g, layer_nodes, qubit_xy, tracker, cube_objects):
    h_nodes = [v for v in layer_nodes if g.type(v) == VertexType.H_BOX]
    for v in h_nodes:
        q = g.qubit(v)
        if q not in qubit_xy:
            continue
        x, y = qubit_xy[q]
        z = tracker.next_free_z[q]
        pos = (x, y, z)

        h_cube = h_routing(pos)
        cube_objects.append(h_cube)

        tracker.record_placed([pos])


def process_s_nodes(g, layer_nodes, qubit_xy, tracker, orientations, cube_objects):
    s_nodes = [
        v for v in layer_nodes
        if g.type(v) == VertexType.Z and is_pi_2_phase(g.phase(v))
    ]
    for v in s_nodes:
        q = g.qubit(v)
        if q not in qubit_xy:
            continue
        x, y = qubit_xy[q]
        z = tracker.next_free_z[q]
        pos = (x, y, z)
        orient = orientations.get(v, "XZX")

        s_result = s_routing(pos, orient, "Z", tracker)

        if isinstance(s_result, dict):
            used_positions = [c.pos for c in s_result["cubes"]]
            cube_objects.extend(s_result["cubes"])
            cube_objects.extend(s_result["pipes"])
        else:
            print(f"Warning: S-routing failed for vertex {v} on qubit {q}")
            used_positions = [pos]

        tracker.record_placed(used_positions)

def process_t_nodes(g, layer_nodes, qubit_xy, tracker, orientations, regions,
                     cultivation_registry, cube_objects, t_positions,
                     max_z_climb=500):
    t_nodes = [
        v for v in layer_nodes
        if g.type(v) == VertexType.Z and is_pi_4_phase(g.phase(v))
    ]
    if not t_nodes:
        return

    ring_2d = regions["bus"]

    pending = []
    for v in t_nodes:
        q = g.qubit(v)
        if q not in qubit_xy:
            continue
        x, y = qubit_xy[q]
        pending.append({
            "v": v, "q": q, "xy": (x, y),
            "orient": orientations.get(v, "XZX"),
            "next_free_z": tracker.next_free_z[q],
        })
    if not pending:
        return
    pending.sort(key=lambda n: n["next_free_z"])

    z = pending[0]["next_free_z"]
    active = []
    climbs = 0

    while pending or active:
        if climbs > max_z_climb:
            stuck = [n["q"] for n in active] + [n["q"] for n in pending]
            raise RuntimeError(
                f"T-node scheduling exceeded max_z_climb={max_z_climb} at z={z}. "
                f"Qubits still unrouted: {stuck}"
            )
        climbs += 1

        newly_eligible = [n for n in pending if n["next_free_z"] <= z]
        if newly_eligible:
            pending = [n for n in pending if n not in newly_eligible]
            active.extend(newly_eligible)

        still_active = []
        for node in active:
            result = _route_one_t_node(node, z, ring_2d, tracker, cultivation_registry)
            if result is None:
                still_active.append(node)
                continue

            cube_objects.append(("path", result["path"], result["chosen_start"], result["chosen_goal"]))
            cube_objects.append(Cube(*result["chosen_start"]))
            # goal cube intentionally omitted -- cultivation column drawing owns that cell

            used_positions = [result["start_pos"]] + result["path"]
            t_positions.update(used_positions)
            tracker.record_placed(used_positions)

        active = still_active
        z += 1


def _route_one_t_node(node, z, ring_2d, tracker, cultivation_registry):
    """Two-hop route for a single T-node at level z. Returns None on failure."""
    hop1_path, hop1_start, hop1_goal = a_star_2d(
        start_pos=(node["xy"][0], node["xy"][1], z),
        start_orient=node["orient"],
        start_type="Z",
        goal_positions={(bx, by, z) for (bx, by) in ring_2d},
        tracker=tracker,
    )
    if hop1_path is None:
        return None

    bus_pos = (hop1_goal[POS][0], hop1_goal[POS][1])
    hop2 = route_bus_to_ready_site(
        bus_pos=bus_pos,
        orient=node["orient"],
        tracker=tracker,
        cultivation_registry=cultivation_registry,
        z_attempt=z,
    )
    if hop2 is None:
        return None

    start_pos = (node["xy"][0], node["xy"][1], z)
    full_path = list(hop1_path) + list(hop2["path"])[1:]
    hop2["site"].consume(z)

    return {
        "path": full_path,
        "chosen_start": hop1_start,
        "chosen_goal": hop2["chosen_goal"],
        "start_pos": start_pos,
    }


def process_cnot_nodes(g, layer_nodes, qubit_xy, tracker, orientations, cube_objects):
    cnot_z_nodes = [
        v for v in layer_nodes if g.type(v) == VertexType.Z and g.phase(v) == 0
    ]

    for z_v in cnot_z_nodes:
        q_z = g.qubit(z_v)
        if q_z not in qubit_xy:
            continue

        x_v = None
        for nbr in g.neighbors(z_v):
            if (
                g.type(nbr) == VertexType.X
                and g.qubit(nbr) != q_z
                and nbr in layer_nodes
            ):
                x_v = nbr
                break

        if x_v is None:
            continue

        q_x = g.qubit(x_v)
        if q_x not in qubit_xy:
            continue

        pos_z = (qubit_xy[q_z][0], qubit_xy[q_z][1], tracker.next_free_z[q_z])
        pos_x = (qubit_xy[q_x][0], qubit_xy[q_x][1], tracker.next_free_z[q_x])

        orient_z = orientations.get(z_v, "XZX")
        orient_x = orientations.get(x_v, "XZZ")

        result = route_cnot_minimal_z(
            pos_z=pos_z,
            pos_x=pos_x,
            orient_z=orient_z,
            orient_x=orient_x,
            tracker=tracker,
        )

        used_positions = [pos_z, pos_x]

        if result["path"]:
            cube_objects.append(
                ("path", result["path"], result["chosen_start"], result["chosen_goal"])
            )
            used_positions.extend(result["path"])
            cube1 = Cube(*result["chosen_start"])
            cube2 = Cube(*result["chosen_goal"])
            cube_objects.append(cube1)
            cube_objects.append(cube2)
        else:
            print(f"Warning: CNOT routing failed between qubit {q_z} and {q_x}")

        tracker.record_placed(used_positions)


def process_t_nodes_distillation(g, layer_nodes, qubit_xy, tracker, orientations, regions,
                                  factory_registry, cube_objects, t_positions,
                                  max_z_climb=500):
    t_nodes = [
        v for v in layer_nodes
        if g.type(v) == VertexType.Z and is_pi_4_phase(g.phase(v))
    ]
    if not t_nodes:
        return

    ring_2d = regions["bus"]

    pending = []
    for v in t_nodes:
        q = g.qubit(v)
        if q not in qubit_xy:
            continue
        x, y = qubit_xy[q]
        pending.append({
            "v": v, "q": q, "xy": (x, y),
            "orient": orientations.get(v, "XZX"),
            "next_free_z": tracker.next_free_z[q],
        })
    if not pending:
        return
    pending.sort(key=lambda n: n["next_free_z"])

    z = pending[0]["next_free_z"]
    active = []
    climbs = 0

    while pending or active:
        if climbs > max_z_climb:
            stuck = [n["q"] for n in active] + [n["q"] for n in pending]
            raise RuntimeError(
                f"T-node scheduling exceeded max_z_climb={max_z_climb} at z={z}. "
                f"Qubits still unrouted: {stuck}"
            )
        climbs += 1

        newly_eligible = [n for n in pending if n["next_free_z"] <= z]
        if newly_eligible:
            pending = [n for n in pending if n not in newly_eligible]
            active.extend(newly_eligible)

        still_active = []
        for node in active:
            # re-derive fresh each node, so a consumption earlier in this
            # same z-level is immediately reflected -- mirrors cultivation's
            # per-node ready_sites(z) call
            ready = factory_registry.ready_factories(z)
            if not ready:
                still_active.append(node)
                continue

            result = _route_one_t_node_distillation(node, z, ring_2d, tracker, ready)
            if result is None:
                still_active.append(node)
                continue

            cube_objects.append(("path", result["path"], result["chosen_start"], result["chosen_goal"]))
            cube_objects.append(Cube(*result["chosen_start"]))

            used_positions = [result["start_pos"]] + result["path"]
            t_positions.update(used_positions)
            tracker.record_placed(used_positions)

        active = still_active
        z += 1


def _route_one_t_node_distillation(node, z, ring_2d, tracker, ready_factories):
    """Two-hop route for a single T-node at level z. Returns None on failure."""
    hop1_path, hop1_start, hop1_goal = a_star_2d(
        start_pos=(node["xy"][0], node["xy"][1], z),
        start_orient=node["orient"],
        start_type="Z",
        goal_positions={(bx, by, z) for (bx, by) in ring_2d},
        tracker=tracker,
    )
    if hop1_path is None:
        return None

    bus_pos = (hop1_goal[POS][0], hop1_goal[POS][1])
    hop2 = route_bus_to_ready_factory(
        bus_pos=bus_pos,
        orient=node["orient"],
        tracker=tracker,
        ready_factories=ready_factories,
        z_attempt=z,
    )
    if hop2 is None:
        return None

    start_pos = (node["xy"][0], node["xy"][1], z)
    full_path = list(hop1_path) + list(hop2["path"])[1:]
    hop2["factory"].consume(z)

    return {
        "path": full_path,
        "chosen_start": hop1_start,
        "chosen_goal": hop2["chosen_goal"],
        "start_pos": start_pos,
    }