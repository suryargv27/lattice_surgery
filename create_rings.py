import matplotlib.pyplot as plt


def chebyshev_neighbors(pos):
    x, y = pos
    return [
        (x + dx, y + dy)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if not (dx == 0 and dy == 0)
    ]


def manhattan_neighbors(pos):
    x, y = pos
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def compute_layout_regions(qubit_xy):
    """
    qubit_xy: dict {qubit_index: (x, y)} from the layout step.

    Returns dict with:
      data       : set of (x, y) -- the data qubits themselves
      ancilla    : set of (x, y) -- full Chebyshev-1 dilation shell around data
      bus        : set of (x, y) -- ancilla cells touching open space
      routing    : set of (x, y) -- remaining ancilla cells (interior)
      cultivation: set of (x, y) -- Manhattan-1 neighbors of bus, outside
                   everything claimed so far
    """
    data = set(qubit_xy.values())

    # Step 2: ancilla = Chebyshev-1 neighbors of ALL data cells, minus data itself
    ancilla = set()
    for cell in data:
        for nb in chebyshev_neighbors(cell):
            if nb not in data:
                ancilla.add(nb)

    # Step 3: split ancilla into bus (touches open space) vs routing (interior)
    claimed_so_far = data | ancilla
    bus = set()
    routing = set()
    for cell in ancilla:
        touches_open_space = any(
            nb not in claimed_so_far for nb in chebyshev_neighbors(cell)
        )
        if touches_open_space:
            bus.add(cell)
        else:
            routing.add(cell)

    # Step 4: cultivation = Manhattan-1 neighbors of bus, outside data/ancilla
    claimed_all = data | ancilla
    cultivation = set()
    for cell in bus:
        for nb in manhattan_neighbors(cell):
            if nb not in claimed_all:
                cultivation.add(nb)

    return {
        "data": data,
        "ancilla": ancilla,
        "bus": bus,
        "routing": routing,
        "cultivation": cultivation,
    }


def plot_layout_regions(regions):
    fig, ax = plt.subplots(figsize=(9, 9))

    def scatter(cells, color, label, size=60):
        if not cells:
            return
        xs = [c[0] for c in cells]
        ys = [c[1] for c in cells]
        ax.scatter(xs, ys, c=color, label=label, s=size)

    scatter(regions["data"], "black", "data qubits")
    scatter(regions["routing"], "orange", "routing (ancilla interior)")
    scatter(regions["bus"], "red", "bus (ancilla outer edge)")
    scatter(regions["cultivation"], "magenta", "cultivation sites")

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.set_title("Layout regions: data / routing / bus / cultivation")
    plt.show()


# Usage:
# qubit_xy = simulated_annealing_layout(g, step=2, iterations=20000, seed=42)
# regions = compute_layout_regions(qubit_xy)
# plot_layout_regions(regions)