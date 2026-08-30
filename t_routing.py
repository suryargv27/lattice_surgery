import heapq
import itertools
import matplotlib.pyplot as plt
from visualize import Cube, Pipe, set_axes_equal
from utils import (
    OPPOSITE,
    TYPE,
    POS,
    ORIENT,
    DIR,
    directions,
    directions_2d,
    add,
    negate,
    abs_tuple,
    SEARCH_TRANSITIONS,
    draw_path,
)


def heuristic_2d(pos, goal_positions):
    """
    Admissible 2D Manhattan distance heuristic to any goal position
    (ignoring Z distance since Z is constant).
    """
    px, py, _ = pos
    return min(abs(px - gx) + abs(py - gy) for gx, gy, _ in goal_positions)


def get_start_neighbors_2d(start, tracker, goal_positions=frozenset()):
    neighbours = []
    usable_face = OPPOSITE[start[TYPE]]

    for i in range(2):
        if start[ORIENT][i] == usable_face:
            for step in (2 * i, 2 * i + 1):
                pos = add(start[POS], directions[step])
                if tracker.is_obstacle(pos) and pos not in goal_positions:
                    continue
                neighbours.append((pos, start[ORIENT], directions[step], None))

    return neighbours


def get_neighbours_2d(state, tracker, goal_positions=frozenset()):
    neighbours = []
    current_dir = state[DIR]
    current_orientation = state[ORIENT]

    for dir in directions_2d:
        if dir == negate(current_dir):
            continue

        pos = add(state[POS], dir)
        if tracker.is_obstacle(pos) and pos not in goal_positions:
            continue

        new_orientation = SEARCH_TRANSITIONS[(current_orientation, current_dir, dir)]
        neighbours.append((pos, new_orientation, dir, None))

    return neighbours


def get_generic_start_neighbors_2d(start, tracker, goal_positions=frozenset()):
    """
    Unrestricted first-step neighbor generation for a start position with
    no type/orientation constraint (e.g. a bus point) -- all 4 cardinal
    directions are legal, no backtrack check needed since there's no
    incoming direction yet.
    """
    neighbours = []
    for dir in directions_2d:
        pos = add(start[POS], dir)
        if tracker.is_obstacle(pos) and pos not in goal_positions:
            continue
        # orientation carried through unchanged -- bus points have no
        # meaningful orientation semantics, so just pass start_orient along
        neighbours.append((pos, start[ORIENT], dir, None))

    return neighbours


def a_star_2d(
    start_pos,
    start_orient,
    start_type,
    goal_positions,
    tracker,
    restrict_start_orientation=True,
):
    counter = itertools.count()
    open_set = []
    came_from = {}
    g_score = {}
    path_positions = {}

    start_state = (start_pos, start_orient, None, start_type)
    g_score[start_state] = 0
    path_positions[start_state] = {start_pos}

    if restrict_start_orientation:
        initial_neighbours = get_start_neighbors_2d(
            start_state, tracker, goal_positions
        )
    else:
        initial_neighbours = get_generic_start_neighbors_2d(
            start_state, tracker, goal_positions
        )

    for neighbour in initial_neighbours:
        g = 1
        if g < g_score.get(neighbour, float("inf")):
            g_score[neighbour] = g
            came_from[neighbour] = start_state
            path_positions[neighbour] = path_positions[start_state] | {neighbour[POS]}

            h = heuristic_2d(neighbour[POS], goal_positions)
            heapq.heappush(open_set, (g + h, g, next(counter), neighbour))

    expanded = 0
    MAX_EXPANDED = 50000

    while open_set:
        expanded += 1
        if expanded > MAX_EXPANDED:
            print("Exceeded max expanded nodes in 2D search")
            return None, None, None

        _, current_g, _, current = heapq.heappop(open_set)

        if current_g != g_score[current]:
            continue

        if current[POS] in goal_positions:
            path = []
            curr = current
            while curr in came_from:
                path.append(curr[POS])
                curr = came_from[curr]
            path.append(curr[POS])

            chosen_start = curr
            chosen_goal = (current[POS], "TTT", current[DIR], "T")
            return path[::-1], chosen_start, chosen_goal

        for neighbour in get_neighbours_2d(current, tracker, goal_positions):
            if neighbour[POS] in path_positions[current]:
                continue

            tentative_g = current_g + 1
            if tentative_g >= g_score.get(neighbour, float("inf")):
                continue

            g_score[neighbour] = tentative_g
            came_from[neighbour] = current
            path_positions[neighbour] = path_positions[current] | {neighbour[POS]}

            h = heuristic_2d(neighbour[POS], goal_positions)
            f_score = tentative_g + h
            heapq.heappush(open_set, (f_score, tentative_g, next(counter), neighbour))

    return None, None, None


def route_bus_to_ready_site(bus_pos, orient, tracker, cultivation_registry, z_attempt):
    ready = cultivation_registry.ready_sites(z_attempt)
    if not ready:
        return None

    x, y = bus_pos
    start_pos = (x, y, z_attempt)
    goal_positions = {(*s.pos, z_attempt) for s in ready}

    path, chosen_start, chosen_goal = a_star_2d(
        start_pos=start_pos,
        start_orient=orient,  # arbitrary/unused since restriction is off
        start_type="Z",  # arbitrary/unused since restriction is off
        goal_positions=goal_positions,
        tracker=tracker,
        restrict_start_orientation=False,  # generic A* from the bus point
    )

    if path is None:
        return None

    goal_xy = (chosen_goal[POS][0], chosen_goal[POS][1])
    site = next(s for s in ready if s.pos == goal_xy)

    return {
        "path": path,
        "chosen_start": chosen_start,
        "chosen_goal": chosen_goal,
        "site": site,
    }

def route_bus_to_ready_factory(bus_pos, orient, tracker, ready_factories, z_attempt):
    """
    Distillation analog of route_bus_to_ready_site. Instead of one goal
    point per ready site, each ready factory contributes up to 8 output
    cells as candidate goals -- generic A* from the bus point picks the
    nearest reachable one across all ready factories.

    ready_factories: list of DistillationFactory objects already filtered
    to those available at z_attempt (i.e. the caller already called
    registry.ready_factories(z_attempt) -- this function does no
    availability checking itself).
    """
    if not ready_factories:
        return None

    x, y = bus_pos
    start_pos = (x, y, z_attempt)

    goal_positions = {
        (ox, oy, z_attempt)
        for f in ready_factories
        for (ox, oy) in f.outputs
    }
    if not goal_positions:
        return None

    path, chosen_start, chosen_goal = a_star_2d(
        start_pos=start_pos,
        start_orient=orient,  # arbitrary/unused since restriction is off
        start_type="Z",       # arbitrary/unused since restriction is off
        goal_positions=goal_positions,
        tracker=tracker,
        restrict_start_orientation=False,  # generic A* from the bus point
    )

    if path is None:
        return None

    goal_xy = (chosen_goal[POS][0], chosen_goal[POS][1])
    factory = next(f for f in ready_factories if goal_xy in f.outputs)

    return {
        "path": path,
        "chosen_start": chosen_start,
        "chosen_goal": chosen_goal,
        "factory": factory,
    }