import math
import random
from visualize import Cube, Pipe


class CultivationSite:
    """
    A single cultivation site. Models PureMagic's exponential completion
    time (Section III-C): duration ~ Exponential(lambda) / code_distance.

    No separate readiness flag -- readiness is always derived by comparing
    the current z against ready_at_z. History is a plain list of
    (start_z, ready_at_z) pairs, one per cultivation attempt.
    """

    def __init__(self, pos, code_distance=17, lam=0.00227, start_z=0):
        self.pos = pos  # (x, y), fixed for the site's lifetime
        self.code_distance = code_distance
        self.lam = lam

        self.start_z = start_z
        self.ready_at_z = None
        self.history = []  # list of (start_z, ready_at_z) pairs

        self._begin_attempt(start_z)

    def _sample_duration(self):
        """Exponential(lambda) sample, scaled to logical cycles by code distance."""
        u = random.random()
        t_cycles = -math.log(1.0 - u) / self.lam
        duration = t_cycles / self.code_distance
        return max(1, round(duration))

    def _begin_attempt(self, z_now):
        self.start_z = z_now
        self.ready_at_z = z_now + self._sample_duration()
        self.history.append((self.start_z, self.ready_at_z))

    def is_ready(self, z_now):
        return z_now >= self.ready_at_z

    def consume(self, z_now):
        """Pick up the T state; site immediately restarts cultivation."""
        assert self.is_ready(z_now), (
            f"Attempted to consume site at {self.pos} before ready "
            f"(ready_at_z={self.ready_at_z}, z_now={z_now})"
        )
        self._begin_attempt(z_now + 1)


class CultivationRegistry:
    def __init__(self, cultivation_positions, code_distance=17, lam=0.00227, start_z=0):
        self.sites = [
            CultivationSite(pos, code_distance=code_distance, lam=lam, start_z=start_z)
            for pos in cultivation_positions
        ]

    def ready_sites(self, z_now):
        """Sites whose CURRENT attempt is ready by z_now."""
        return [s for s in self.sites if s.is_ready(z_now)]


def build_cultivation_column(site, final_z):
    """
    Walks the site's (start_z, ready_at_z) history and emits Cube/Pipe
    objects.

    Per attempt (start_z, ready_z):
      - magenta cubes: start_z .. ready_z - 1
      - magenta pipes: start_z .. ready_z (so the last magenta pipe bridges
        into the first purple cube at ready_z)
      - purple cubes:  ready_z .. next_start_z - 1
      - purple pipes:  ready_z .. next_start_z - 1 (connecting consecutive
        purple cubes only)

    The chain breaks between attempts -- no pipe from the last purple cube
    of one attempt to the first magenta cube of the next.

    Everything is trimmed at final_z.
    """
    x, y = site.pos
    objects = []

    for i, (start_z, ready_z) in enumerate(site.history):
        if start_z > final_z:
            break

        next_start_z = (
            site.history[i + 1][0] if i + 1 < len(site.history) else final_z + 1
        )

        magenta_cube_end = min(ready_z, final_z + 1)      # cubes: [start_z, magenta_cube_end)
        magenta_pipe_end = min(ready_z + 1, final_z + 1)  # pipes: [start_z, magenta_pipe_end)
        purple_cube_end = min(next_start_z, final_z + 1)  # cubes: [ready_z, purple_cube_end)

        # -- magenta cubes --
        magenta_cubes = {}
        for z in range(start_z, magenta_cube_end):
            pos = (x, y, z)
            cube = Cube(pos=pos, orientation="TTT", dir=None, type=None)
            magenta_cubes[z] = cube
            objects.append(cube)

        # -- purple cubes --
        purple_cubes = {}
        for z in range(ready_z, purple_cube_end):
            pos = (x, y, z)
            cube = Cube(pos=pos, orientation="RRR", dir=None, type=None)
            purple_cubes[z] = cube
            objects.append(cube)

        all_cubes = {**magenta_cubes, **purple_cubes}

        # -- magenta pipes: start_z .. ready_z (may bridge into first purple cube) --
        for z in range(start_z, magenta_pipe_end - 1):
            if z in all_cubes and (z + 1) in all_cubes:
                pipe = Pipe(pos1=all_cubes[z].pos, pos2=all_cubes[z + 1].pos, orientation="TTT")
                objects.append(pipe)

        # -- purple pipes: consecutive purple cubes only --
        for z in range(ready_z, purple_cube_end - 1):
            if z in purple_cubes and (z + 1) in purple_cubes:
                pipe = Pipe(pos1=purple_cubes[z].pos, pos2=purple_cubes[z + 1].pos, orientation="RRR")
                objects.append(pipe)

    return objects


def build_all_cultivation_columns(cultivation_registry, final_z):
    objects = []
    for site in cultivation_registry.sites:
        objects.extend(build_cultivation_column(site, final_z))
    return objects