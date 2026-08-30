class QubitTracker:
    """Manages qubit Z-levels and placed geometry obstacles.
    Qubit columns and extra reserved columns (cultivation sites,
    distillation factory footprints, etc.) are treated as infinite
    obstacles along z."""

    def __init__(self, qubit_xy, cultivation_sites=None, extra_columns=None):
        self.qubit_xy = qubit_xy
        self.qubit_columns = set(qubit_xy.values())
        self.next_free_z = {q: 0 for q in qubit_xy}
        self.placed_obstacles = set()

        reserved = set()
        if cultivation_sites:
            reserved |= set(cultivation_sites)
        if extra_columns:
            reserved |= set(extra_columns)
        self.extra_columns = reserved

    def is_column_obstacle(self, pos):
        xy = (pos[0], pos[1])
        return xy in self.qubit_columns or xy in self.extra_columns

    def is_obstacle(self, pos):
        return self.is_column_obstacle(pos) or pos in self.placed_obstacles

    def record_placed(self, positions):
        for pos in positions:
            self.placed_obstacles.add(pos)

        for q, (qx, qy) in self.qubit_xy.items():
            column_zs = [pos[2] for pos in positions if pos[0] == qx and pos[1] == qy]
            if column_zs:
                max_used_z = max(column_zs)
                new_free_z = max_used_z + 1
                if new_free_z > self.next_free_z[q]:
                    self.next_free_z[q] = new_free_z