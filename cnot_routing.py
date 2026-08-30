import heapq
import itertools
import matplotlib.pyplot as plt
from visualize import Cube, Pipe, set_axes_equal
from utils import OPPOSITE, TYPE, POS, ORIENT, DIR, directions, add, negate, abs_tuple, SEARCH_TRANSITIONS, draw_path

# ==============================================================================
# 6. NEIGHBOR GENERATION (single start, single goal)
# ==============================================================================
def get_start_neighbors(start, tracker, goal_pos, goal_orient, goal_type):
    neighbours = []
    usable_face = OPPOSITE[start[TYPE]]

    for i in range(3):
        if start[ORIENT][i] == usable_face:
            for step in (2 * i, 2 * i + 1):
                pos = add(start[POS], directions[step])
                dir_ = directions[step]
                candidate = (pos, start[ORIENT], dir_, None)

                if tracker.is_obstacle(pos):
                    if not check_goal(candidate, goal_pos, goal_orient, goal_type):
                        continue

                neighbours.append(candidate)

    return neighbours


def get_neighbours(state, tracker, goal_pos, goal_orient, goal_type):
    neighbours = []
    current_dir = state[DIR]
    current_orientation = state[ORIENT]

    for dir in directions:
        if dir == negate(current_dir):
            continue

        pos = add(state[POS], dir)
        new_orientation = SEARCH_TRANSITIONS[(current_orientation, current_dir, dir)]
        candidate = (pos, new_orientation, dir, None)

        if tracker.is_obstacle(pos):
            if not check_goal(candidate, goal_pos, goal_orient, goal_type):
                continue

        neighbours.append(candidate)

    return neighbours

# ==============================================================================
# 7. SINGLE-GOAL A* SEARCH
# ==============================================================================
def check_goal(current, goal_pos, goal_orient, goal_type):
    """
    Checks if the current state has reached the single goal position and
    meets all orientation/face matching criteria.
    """
    if current[POS] != goal_pos:
        return False

    if goal_type == "0":  # Generic target, no orientation requirement
        return True

    current_abs_dir = abs_tuple(current[DIR])
    entry_axis = current_abs_dir.index(1)
    entry_face = goal_orient[entry_axis]

    if entry_face != OPPOSITE[goal_type]:
        return False

    for i in range(3):
        if i == entry_axis:
            continue
        if current[ORIENT][i] != goal_orient[i]:
            return False

    return True


def heuristic(pos, goal_pos):
    """Admissible Manhattan distance heuristic to the single goal position."""
    px, py, pz = pos
    gx, gy, gz = goal_pos
    return abs(px - gx) + abs(py - gy) + abs(pz - gz)


def a_star_single(
    start_pos,
    start_orient,
    start_type,
    goal_pos,
    goal_orient,
    goal_type,
    tracker,
    max_expanded=50000,
):
    """
    Single-start, single-goal oriented A* search. Returns (path, start_state,
    goal_state) on success, or (None, None, None) if no path is found or the
    expansion budget is exceeded.
    """
    counter = itertools.count()
    open_set = []
    came_from = {}
    g_score = {}
    path_positions = {}

    start_state = (start_pos, start_orient, None, start_type)
    g_score[start_state] = 0
    path_positions[start_state] = {start_pos}

    for neighbour in get_start_neighbors(
        start_state, tracker, goal_pos, goal_orient, goal_type
    ):
        g = 1
        if g < g_score.get(neighbour, float("inf")):
            g_score[neighbour] = g
            came_from[neighbour] = start_state
            path_positions[neighbour] = path_positions[start_state] | {neighbour[POS]}
            h = heuristic(neighbour[POS], goal_pos)
            heapq.heappush(open_set, (g + h, g, next(counter), neighbour))

    expanded = 0

    while open_set:
        expanded += 1
        if expanded > max_expanded:
            return None, None, None

        _, current_g, _, current = heapq.heappop(open_set)
        if current_g != g_score[current]:
            continue

        if check_goal(current, goal_pos, goal_orient, goal_type):
            path = []
            curr = current
            while curr in came_from:
                path.append(curr[POS])
                curr = came_from[curr]
            path.append(curr[POS])

            corrected_goal = (current[POS], goal_orient, current[DIR], goal_type)
            return path[::-1], curr, corrected_goal

        for neighbour in get_neighbours(
            current, tracker, goal_pos, goal_orient, goal_type
        ):
            if neighbour[POS] in path_positions[current]:
                continue

            tentative_g = current_g + 1
            if tentative_g >= g_score.get(neighbour, float("inf")):
                continue

            g_score[neighbour] = tentative_g
            came_from[neighbour] = current
            path_positions[neighbour] = path_positions[current] | {neighbour[POS]}

            h = heuristic(neighbour[POS], goal_pos)
            heapq.heappush(
                open_set, (tentative_g + h, tentative_g, next(counter), neighbour)
            )

    return None, None, None


def route_cnot_minimal_z(
    pos_z,
    pos_x,
    orient_z,
    orient_x,
    tracker,
    restarts=10,
    goal_ceiling=3,
    max_expanded_per_attempt=50000,
):
    """
    Routes a CNOT pair, searching for the lowest-z valid goal for a given
    fixed start, and only raising the start z if every goal in the current
    band fails.

    For restart round k (0-indexed):
        start_z = higher_cube_base_z + k
        goal_z tried in strictly ascending order: goal_z_floor, goal_z_floor+1,
            ..., start_z + goal_ceiling
        Each (start_z, goal_z) pair is a single-start/single-goal A* call.
        The FIRST goal_z that succeeds is accepted immediately (guaranteeing
        the lowest reachable goal for that start_z; no lower goal_z in this
        round's band could have worked, since all lower ones were already
        tried and failed).

    If every goal_z in round k fails, k increments (start_z rises by 1, goal
    band grows by 1 at the top) and the whole ascending goal sweep repeats.

    Stops at `restarts` rounds; reports failure with full attempt history
    if nothing succeeded. `restarts` and `goal_ceiling` alone bound the
    total search space — no separate global z ceiling is needed here.

    Parameters
    ----------
    pos_z, pos_x : (x, y, z) next-free-z positions for the Z-type and X-type
        cubes of this CNOT pair.
    orient_z, orient_x : orientation strings for the Z and X cubes.
    obstacles : combined obstacle set (tracker.obstacles).
    restarts : number of restart rounds (default 3).
    goal_ceiling : extra z-headroom above start_z included in the goal sweep
        each round (default 3).
    max_expanded_per_attempt : expansion budget passed to each individual
        a_star_single call.

    Returns
    -------
    dict with keys "path", "chosen_start", "chosen_goal", "start_pos",
    "start_type", "start_orient", "goal_pos", "goal_type", "goal_orient",
    and "attempts" (full diagnostic trail of every (start_z, goal_z) tried).
    On total failure, "path" is None.
    """
    if pos_z[2] >= pos_x[2]:
        start_base_pos, start_orient, start_type = pos_z, orient_z, "Z"
        goal_base_pos, goal_orient, goal_type = pos_x, orient_x, "X"
    else:
        start_base_pos, start_orient, start_type = pos_x, orient_x, "X"
        goal_base_pos, goal_orient, goal_type = pos_z, orient_z, "Z"

    gx, gy, goal_z_floor = goal_base_pos
    sx, sy, start_z_base = start_base_pos

    attempts = []

    for k in range(restarts):
        start_z = start_z_base + k
        start_pos = (sx, sy, start_z)
        goal_band_top = start_z + goal_ceiling

        round_found = False
        for goal_z in range(goal_z_floor, goal_band_top + 1):
            goal_pos = (gx, gy, goal_z)

            path, chosen_start, chosen_goal = a_star_single(
                start_pos=start_pos,
                start_orient=start_orient,
                start_type=start_type,
                goal_pos=goal_pos,
                goal_orient=goal_orient,
                goal_type=goal_type,
                tracker=tracker,
                max_expanded=max_expanded_per_attempt,
            )

            attempts.append({
                "round": k,
                "start_z": start_z,
                "goal_z": goal_z,
                "status": "success" if path else "no_path",
            })

            if path:
                round_found = True
                return {
                    "path": path,
                    "chosen_start": chosen_start,
                    "chosen_goal": chosen_goal,
                    "start_pos": start_pos,
                    "start_type": start_type,
                    "start_orient": start_orient,
                    "goal_pos": goal_pos,
                    "goal_type": goal_type,
                    "goal_orient": goal_orient,
                    "attempts": attempts,
                }

        if not round_found:
            continue  # advance to next restart round

    return {
        "path": None,
        "chosen_start": None,
        "chosen_goal": None,
        "start_pos": None,
        "goal_pos": None,
        "attempts": attempts,
    }



# ==============================================================================
# 9. EXECUTION ENTRY POINT
# ==============================================================================
if __name__ == "__main__":

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    pos_z = (0, 0, 2)
    pos_x = (0, 5, 0)
    orient_z, orient_x = "XZX", "XZZ"

    obstacles = {
        (0, 5, 0),
        (0, 0, 1),
    }

    result = route_cnot_minimal_z(
        pos_z=pos_z,
        pos_x=pos_x,
        orient_z=orient_z,
        orient_x=orient_x,
        obstacles=obstacles,
    )

    if result["path"] is not None:
        print(f"Path found from {result['start_pos']} to {result['goal_pos']}")
        draw_path(result["path"], result["chosen_start"], result["chosen_goal"], ax)
        c1 = Cube(*result["chosen_start"])
        c1.draw(ax)
        c2 = Cube(*result["chosen_goal"])
        c2.draw(ax)
    else:
        print("No path found across all restarts")
        for a in result["attempts"]:
            print(a)

    set_axes_equal(ax)
    ax.set_box_aspect([1, 1, 1])
    plt.show()