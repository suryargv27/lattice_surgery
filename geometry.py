def get_bounding_square_ring(qubit_xy, offset=2):
    """
    Computes a square ring of positions located `offset` steps outside
    the bounding box of all qubit (x, y) coordinates.
    """
    xs = [pos[0] for pos in qubit_xy.values()]
    ys = [pos[1] for pos in qubit_xy.values()]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    min_x_outer = min_x - offset
    max_x_outer = max_x + offset
    min_y_outer = min_y - offset
    max_y_outer = max_y + offset

    ring_positions = set()

    for x in range(min_x_outer, max_x_outer + 1):
        ring_positions.add((x, min_y_outer))
        ring_positions.add((x, max_y_outer))

    for y in range(min_y_outer, max_y_outer + 1):
        ring_positions.add((min_x_outer, y))
        ring_positions.add((max_x_outer, y))

    return ring_positions


def compute_bounding_volume(positions):
    """
    Given a set/list of (x, y, z) positions, returns a dict with the
    bounding-box dimensions (extent along each axis, inclusive of both
    ends) and the resulting volume (product of the three extents).
    Returns None if positions is empty.
    """
    if not positions:
        return None

    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    zs = [p[2] for p in positions]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)

    dx = max_x - min_x + 1
    dy = max_y - min_y + 1
    dz = max_z - min_z + 1

    return {
        "min": (min_x, min_y, min_z),
        "max": (max_x, max_y, max_z),
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "volume": dx * dy * dz,
        "cell_count": len(positions),
    }