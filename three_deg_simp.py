import pyzx as zx
import math


def simplify_graph(g):
    g = g.copy()
    zx.id_simp(g)
    zx.spider_simp(g)
    zx.id_simp(g)

    # Split vertices with degree > 3
    while True:
        # Find first vertex with degree > 3
        v = None
        for candidate in g.vertices():
            if (
                g.type(candidate) != zx.VertexType.BOUNDARY
                and g.vertex_degree(candidate) > 3
            ):
                v = candidate
                break

        if v is None:
            break

        # Get vertex properties
        neighbours = list(g.neighbors(v))

        # SORT NEIGHBOURS BY ROW VALUE
        neighbours.sort(key=lambda neighbor: g.row(neighbor))

        phase = g.phase(v)
        vtype = g.type(v)
        row = g.row(v)
        qubit = g.qubit(v)

        # Calculate split sizes for maximum degree 3
        m = len(neighbours)

        if m <= 3:
            n = 1
        else:
            n = m - 2

        if n == 1:
            capacities = [3]
        elif n == 2:
            capacities = [2, 2]
        else:
            capacities = [2] + [1] * (n - 2) + [2]

        assert sum(capacities) >= m

        # Fill capacities from left to right
        split_sizes = []
        remaining = m

        for cap in capacities:
            take = min(cap, remaining)
            split_sizes.append(take)
            remaining -= take

        assert remaining == 0

        # Save edge types
        edge_types = {}
        for neighbor in neighbours:
            edge = g.edge(v, neighbor)
            edge_types[neighbor] = g.edge_type(edge)

        # Remove original vertex
        g.remove_vertex(v)

        # Shift subsequent vertices down to make room
        shift_amount = n - 1
        for other in list(g.vertices()):
            if g.qubit(other) == qubit and g.row(other) >= row:
                g.set_row(other, g.row(other) + shift_amount)

        # Create new vertices
        new_vertices = []
        for i in range(n):
            phase_value = phase if i == 0 else 0
            new_v = g.add_vertex(ty=vtype, row=row + i, qubit=qubit, phase=phase_value)
            new_vertices.append(new_v)

        # Assign neighbors sequentially based on split sizes
        start_idx = 0
        for i, size in enumerate(split_sizes):
            chunk = neighbours[start_idx : start_idx + size]
            target_v = new_vertices[i]

            for neighbor in chunk:
                edge_type = edge_types[neighbor]
                g.add_edge((target_v, neighbor), edgetype=edge_type)

            start_idx += size

        # Add chain edges between new vertices
        for i in range(len(new_vertices) - 1):
            v1 = new_vertices[i]
            v2 = new_vertices[i + 1]
            g.add_edge((v1, v2))

    # Pack rows after all splitting is done
    g.pack_circuit_rows()
    g.normalize()

    return g.copy()


# g = zx.generate.cnots(3, 10)
# h = simplify_graph(g)

# zx.draw(g, labels=True)
# zx.draw(h, labels=True)
