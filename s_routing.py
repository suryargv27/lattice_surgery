import random
from visualize import Cube, Pipe
import matplotlib.pyplot as plt
from visualize import set_axes_equal

X = "X"
Z = "Z"
O = "0"

OPPOSITE = {
    X: Z,
    Z: X,
}

def s_routing(pos, orientation, cube_type, tracker, max_attempts=50):
    """
    Attempts to create an S-routing structure (2 new cubes: pos_across and
    pos_sss) starting from `pos`. If neither step direction is free at the
    current base z, the base is climbed to z+1 and retried, up to
    `max_attempts` times.

    On success, draws the base cube at whatever z it finally settled on,
    plus pos_across and pos_sss, plus connecting pipes, and returns a dict
    describing everything placed (including the FINAL base position used,
    since it may differ from the original `pos` passed in).

    Returns "no routing possible" (str) if no valid placement was found
    within max_attempts climbs.
    """
    usable_face = OPPOSITE[cube_type]

    usable_axis = None
    for axis_idx in [0, 1]:
        if orientation[axis_idx] == usable_face:
            usable_axis = axis_idx
            break

    if usable_axis is None:
        raise ValueError(
            f"No usable X or Y axis found for orientation '{orientation}' and type '{cube_type}'"
        )

    x, y, z = pos

    def get_positions(base_z, step):
        if usable_axis == 0:
            pos_across = (x + step, y, base_z)
            pos_sss = (x + step, y, base_z + 1)
        else:
            pos_across = (x, y + step, base_z)
            pos_sss = (x, y + step, base_z + 1)
        return pos_across, pos_sss

    for attempt in range(max_attempts):
        base_z = z + attempt
        base_pos = (x, y, base_z)

        valid_steps = []
        for step in (1, -1):
            pos_across, pos_sss = get_positions(base_z, step)
            if not tracker.is_obstacle(pos_across) and not tracker.is_obstacle(pos_sss):
                valid_steps.append(step)

        if valid_steps:
            step_val = random.choice(valid_steps)
            pos_across, pos_sss = get_positions(base_z, step_val)

            base_cube = Cube(base_pos, orientation, None, cube_type)
            cube_across = Cube(pos_across, orientation, None, cube_type)
            cube_sss = Cube(pos_sss, "SSS", None, "S")

            pipe1 = Pipe(pos1=base_pos, pos2=pos_across, orientation=orientation)
            pipe2 = Pipe(pos1=pos_sss, pos2=pos_across, orientation=orientation)

            return {
                "base_pos": base_pos,
                "cubes": [base_cube, cube_across, cube_sss],
                "pipes": [pipe1, pipe2],
            }

    return "no routing possible"

if __name__ == "__main__":

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Define start parameters
    start_pos = (0, 0, 0)
    start_orient = "ZXX"  # usable face 'Z' is at X axis (index 0)
    start_type = Z  # OPPOSITE[Z] -> 'X' or OPPOSITE[X] -> 'Z'

    # Draw initial start cube
    c_start = Cube(start_pos, start_orient, None, start_type)
    c_start.draw(ax)

    # Perform S-routing
    result = s_routing(start_pos, start_orient, start_type, ax)
    print(
        f"S-routing generated using {result['chosen_axis']} axis, direction {result['direction']}"
    )

    set_axes_equal(ax)
    ax.set_box_aspect([1, 1, 1])
    plt.show()
