# column_fill.py
from visualize import Cube, Pipe


def build_qubit_column_index(qubit_xy, cube_objects):
    xy_to_q = {xy: q for q, xy in qubit_xy.items()}
    columns = {q: {} for q in qubit_xy}

    for item in cube_objects:
        if not isinstance(item, Cube):
            continue  # skip Pipe objects and ("path", ...) tuples
        cx, cy, cz = item.pos
        q = xy_to_q.get((cx, cy))
        if q is not None:
            columns[q][cz] = item

    return columns

def fill_qubit_columns(qubit_xy, tracker, cube_objects):
    columns = build_qubit_column_index(qubit_xy, cube_objects)
    new_objects = []

    for q, (x, y) in qubit_xy.items():
        col = columns[q]
        boundary_zs = sorted(z for z, cube in col.items() if cube.type == "0")
        if len(boundary_zs) < 2:
            continue
        z_start, z_end = boundary_zs[0], boundary_zs[-1]

        orientation, ctype = "XZX", "Z"
        prev_cube = col.get(z_start)

        for z in range(z_start + 1, z_end + 1):
            existing = col.get(z)

            if existing is not None:
                cube_here = existing
            else:
                cube_here = Cube(pos=(x, y, z), orientation=orientation, dir=None, type=ctype)
                new_objects.append(cube_here)
                col[z] = cube_here
                tracker.record_placed([(x, y, z)])

            # incoming pipe uses the orientation in effect BEFORE any flip
            pipe = Pipe(pos1=prev_cube.pos, pos2=cube_here.pos, orientation=orientation)
            new_objects.append(pipe)

            # flip AFTER drawing the incoming pipe, so outgoing pipe/cubes get the new one
            if existing is not None and existing.type == "H":
                if orientation == "XZX":
                    orientation, ctype = "ZXX", "Z"
                else:
                    orientation, ctype = "XZX", "Z"

            prev_cube = cube_here

    return new_objects