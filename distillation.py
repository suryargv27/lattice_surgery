from visualize import Cube, Pipe

T_LATENCY = 12  # z-levels a factory needs, deterministically, to produce a magic state


class DistillationFactory:
    def __init__(self, center, cells, outputs, start_z=0):
        self.center = center
        self.cells = set(cells)
        self.outputs = set(outputs)

        self.start_z = start_z
        self.history = []
        self._begin_attempt(start_z)

    def _begin_attempt(self, start_z):
        self.start_z = start_z
        self.history.append(("start_distilling", start_z))
        self.history.append(("became_ready", start_z + T_LATENCY))

    def is_ready(self, z_now):
        return (z_now - self.start_z) >= T_LATENCY

    def consume(self, z_now):
        assert self.is_ready(z_now), (
            f"Attempted to consume factory at {self.center} at z={z_now} "
            f"while start_z={self.start_z} (not ready)."
        )
        self.history.append(("consumed", z_now))
        self._begin_attempt(z_now + 1)


class DistillationFactoryRegistry:
    def __init__(self, factories_result, start_z=0):
        """
        factories_result: the "factories" list from solve_max_factories(),
        each dict with "center", "cells", "outputs".
        """
        self.factories = [
            DistillationFactory(
                center=f["center"],
                cells=f["cells"],
                outputs=f["outputs"],
                start_z=start_z,
            )
            for f in factories_result
        ]

    def ready_factories(self, z_now):
        """Factories currently available to serve as a hop-2 goal."""
        return [f for f in self.factories if f.is_ready(z_now)]

    def factory_owning_output(self, xy):
        for f in self.factories:
            if xy in f.outputs:
                return f
        return None

def _attempts_from_history(history):
    """
    Groups a factory's flat (event, z) history into per-attempt triples:
    (start_z, ready_z, consumed_z_or_None). Mirrors cultivation's
    (start_z, ready_at_z) history list, but distillation also needs
    consumed_z to know where the NEXT attempt's start_z will be -- unlike
    cultivation, where next_start_z is read directly from the next
    attempt's own start_z in the list.
    """
    attempts = []
    start_z = ready_z = None

    for event, z in history:
        if event == "start_distilling":
            start_z = z
            ready_z = None
        elif event == "became_ready":
            ready_z = z
        elif event == "consumed":
            attempts.append((start_z, ready_z, z))
            start_z = ready_z = None

    if start_z is not None:
        # attempt still in progress (mid-distilling or ready-but-unconsumed)
        attempts.append((start_z, ready_z, None))

    return attempts


def build_factory_column(factory, final_z):
    """
    Walks the factory's per-attempt (start_z, ready_z, consumed_z) triples
    and emits Cube/Pipe objects for all 9 footprint cells per z-slice,
    reproducing cultivation.py's build_cultivation_column exactly, widened
    from a single column to the 3x3 block.

    Per attempt (start_z, ready_z, consumed_z):
      - magenta cubes: start_z .. ready_z - 1
      - magenta pipes: start_z .. ready_z (last magenta pipe bridges into
        the first purple cube at ready_z)
      - purple cubes:  ready_z .. next_start_z - 1
      - purple pipes:  ready_z .. next_start_z - 1 (consecutive purple
        cubes only)

    next_start_z is consumed_z + 1 (the next attempt's start_z), or
    final_z + 1 if this is the last/open attempt.

    The chain breaks between attempts -- no pipe from the last purple cube
    of one attempt to the first magenta cube of the next.

    Everything is trimmed at final_z.
    """
    attempts = _attempts_from_history(factory.history)
    objects = []

    for i, (start_z, ready_z, consumed_z) in enumerate(attempts):
        if start_z > final_z:
            break

        if ready_z is None:
            # still distilling, never became ready within recorded history --
            # treat the whole remainder up to final_z as magenta, no purple phase
            ready_z = final_z + 1

        next_start_z = (consumed_z + 1) if consumed_z is not None else (final_z + 1)

        magenta_cube_end = min(ready_z, final_z + 1)      # cubes: [start_z, magenta_cube_end)
        magenta_pipe_end = min(ready_z + 1, final_z + 1)  # pipes: [start_z, magenta_pipe_end)
        purple_cube_end = min(next_start_z, final_z + 1)  # cubes: [ready_z, purple_cube_end)

        for (cx, cy) in factory.cells:
            magenta_cubes = {}
            for z in range(start_z, magenta_cube_end):
                pos = (cx, cy, z)
                cube = Cube(pos=pos, orientation="TTT", dir=None, type=None)
                magenta_cubes[z] = cube
                objects.append(cube)

            purple_cubes = {}
            for z in range(ready_z, purple_cube_end):
                pos = (cx, cy, z)
                cube = Cube(pos=pos, orientation="RRR", dir=None, type=None)
                purple_cubes[z] = cube
                objects.append(cube)

            all_cubes = {**magenta_cubes, **purple_cubes}

            for z in range(start_z, magenta_pipe_end - 1):
                if z in all_cubes and (z + 1) in all_cubes:
                    pipe = Pipe(pos1=all_cubes[z].pos, pos2=all_cubes[z + 1].pos, orientation="TTT")
                    objects.append(pipe)

            for z in range(ready_z, purple_cube_end - 1):
                if z in purple_cubes and (z + 1) in purple_cubes:
                    pipe = Pipe(pos1=purple_cubes[z].pos, pos2=purple_cubes[z + 1].pos, orientation="RRR")
                    objects.append(pipe)

    return objects


def build_all_factory_columns(registry, final_z):
    objects = []
    for factory in registry.factories:
        objects.extend(build_factory_column(factory, final_z))
    return objects