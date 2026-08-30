import pyzx as zx

def create_hboxes(g):
    """
    Finds all Hadamard edges in the graph, replaces each with an H-box vertex,
    and connects the H-box to the original endpoints using simple edges.
    """
    g = g.copy()

    # Iterate over a list copy of all edges in the graph
    for e in list(g.edges()):
        if g.edge_type(e) == zx.EdgeType.HADAMARD:
            u, v = g.edge_st(e)

            # Remove the original Hadamard edge
            g.remove_edge(e)

            # Calculate midpoint row/qubit for clean visual representation
            q_mid = (g.qubit(u) + g.qubit(v)) / 2.0
            r_mid = (g.row(u) + g.row(v)) / 2.0

            # Add an H-box vertex
            h_box = g.add_vertex(ty=zx.VertexType.H_BOX, qubit=q_mid, row=r_mid)

            # Connect the H-box to original endpoints via simple edges
            g.add_edge((u, h_box), edgetype=zx.EdgeType.SIMPLE)
            g.add_edge((h_box, v), edgetype=zx.EdgeType.SIMPLE)

    return g

from fractions import Fraction

def split_3pi4_and_5pi4_nodes(g):
    """
    Finds all green (Z-type) nodes with phase 3pi/4 or 5pi/4 and splits them:
    - 3pi/4  --> pi/2 node followed by a pi/4 node
    - 5pi/4  --> pi node followed by a pi/4 node
    """
    g = g.copy()
    
    # Target phases in PyZX (fractions of pi)
    phase_3pi4 = Fraction(3, 4)
    phase_5pi4 = Fraction(5, 4)
    
    # Iterate over a list copy of all vertices
    for v in list(g.vertices()):
        if g.type(v) == zx.VertexType.Z:
            v_phase = g.phase(v) % 2  # Normalize phase mod 2*pi
            
            if v_phase == phase_3pi4 or v_phase == phase_5pi4:
                # Determine phase breakdown
                if v_phase == phase_3pi4:
                    first_phase = Fraction(1, 2)  # pi/2
                else:
                    first_phase = Fraction(1, 1)  # pi
                
                second_phase = Fraction(1, 4)     # pi/4
                
                # 1. Update existing node v to carry the first phase
                g.set_phase(v, first_phase)
                
                # 2. Find output/right neighbor to insert new node cleanly
                q = g.qubit(v)
                r = g.row(v)
                
                # Retrieve all neighbors to rewire right-side connections
                neighbors = list(g.neighbors(v))
                
                # Create new node at row r + 0.5 on the same qubit line
                v_new = g.add_vertex(
                    ty=zx.VertexType.Z,
                    qubit=q,
                    row=r + 0.5,
                    phase=second_phase
                )
                
                # Identify right-side neighbors (row > r)
                right_neighbors = [n for n in neighbors if g.row(n) > r]
                
                # If no right neighbor found, pick any neighbor other than inputs
                if not right_neighbors and len(neighbors) > 0:
                    right_neighbors = [neighbors[-1]]
                
                # Re-route right-side edges through v_new
                for nbr in right_neighbors:
                    e = g.edge(v, nbr)
                    e_type = g.edge_type(e)
                    
                    g.remove_edge(e)
                    g.add_edge((v_new, nbr), edgetype=e_type)
                
                # Connect original node v to new node v_new with a simple edge
                g.add_edge((v, v_new), edgetype=zx.EdgeType.SIMPLE)

    return g


def simplify_green_spiders(g):
    """
    Traverses each qubit line left-to-right.

    - Boundaries, Red nodes (X-type), and Hadamard nodes (H_BOX) are treated as separators.
    - Collects degree-2 green nodes into a fusing group.
    - Ignores (bypasses) green nodes with degree > 2 while keeping the group open.
    - When a separator is hit:
        1. Combines the phases of all degree-2 green nodes in the group.
        2. Assigns the combined sum to the FIRST node in the group.
        3. Sets the phases of all remaining nodes in the group to ZERO.
        4. Resets the group and continues searching.
    """
    g = g.copy()
    unique_qubits = sorted(list(set(g.qubits().values())))

    for q in unique_qubits:
        if q < 0:
            continue

        # Fetch all vertices on qubit q and sort left-to-right by row
        q_nodes = [v for v in g.vertices() if g.qubit(v) == q]
        q_nodes.sort(key=lambda v: g.row(v))

        fusing_group = []

        for v in q_nodes:
            v_type = g.type(v)

            # Check if this node is a SEPARATOR:
            # - Boundary node
            # - Red spider (X-type)
            # - Hadamard node (H_BOX)
            is_separator = (
                v_type == zx.VertexType.BOUNDARY
                or v_type == zx.VertexType.X
                or v_type == zx.VertexType.H_BOX
            )

            if is_separator:
                # Process and collapse the current group if it contains nodes
                _process_fusing_group(g, fusing_group)
                # Reset fusing group for the next segment
                fusing_group = []

            else:
                # Node is a Green spider (Z-type)
                # If degree is 2, add it to the active fusing group
                if g.vertex_degree(v) == 2:
                    fusing_group.append(v)
                # If degree > 2 (e.g., CNOT control), ignore it and continue searching

        # Process any remaining group at the end of the qubit line
        _process_fusing_group(g, fusing_group)
    
    g = split_3pi4_and_5pi4_nodes(g)
    zx.id_simp(g)

    return g


def _process_fusing_group(g, fusing_group):
    """
    Combines phases of all nodes in fusing_group.
    Assigns the total phase to fusing_group[0] and sets the rest to 0.
    """
    if len(fusing_group) <= 1:
        return  # Nothing to combine for 0 or 1 node

    # 1. Calculate sum of all phases in the group
    total_phase = sum(g.phase(v) for v in fusing_group) % 2

    # 2. Assign total phase to the first node
    g.set_phase(fusing_group[0], total_phase)

    # 3. Set phases of remaining nodes in the group to ZERO
    for v in fusing_group[1:]:
        g.set_phase(v, 0)


from collections import defaultdict, deque
import pyzx as zx

def assign_layers(g):
    """
    Tightly packs nodes into layers while ensuring CNOT-connected cross-qubit
    nodes are strictly assigned to the exact same layer.
    """
    vertices = list(g.vertices())
    
    # 1. Group CNOT / Cross-qubit neighbors into Equivalence Classes (Super-Nodes)
    parent = {v: v for v in vertices}
    
    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    # Merge nodes connected by cross-qubit edges (CNOTs)
    for u in vertices:
        for v in g.neighbors(u):
            if g.qubit(u) != g.qubit(v):
                union(u, v)

    # Group vertices by their super-node root
    super_nodes = defaultdict(list)
    for v in vertices:
        super_nodes[find(v)].append(v)

    # 2. Build DAG dependencies between Super-Nodes based on same-qubit left-to-right ordering
    super_dag = defaultdict(set)
    in_degree = defaultdict(int)
    
    # Initialize in-degree for all super nodes
    for s_root in super_nodes:
        in_degree[s_root] = 0

    for u in vertices:
        for v in g.neighbors(u):
            if g.qubit(u) == g.qubit(v) and g.row(u) < g.row(v):
                root_u = find(u)
                root_v = find(v)
                if root_u != root_v and root_v not in super_dag[root_u]:
                    super_dag[root_u].add(root_v)

    for root_u in super_dag:
        for root_v in super_dag[root_u]:
            in_degree[root_v] += 1

    # 3. Compute Longest-Path Layering (ASAP scheduling)
    super_layer = {}
    queue = deque([s_root for s_root, deg in in_degree.items() if deg == 0])

    for s_root in queue:
        super_layer[s_root] = 0

    while queue:
        curr_s = queue.popleft()
        curr_l = super_layer[curr_s]

        for nxt_s in super_dag[curr_s]:
            # Push layer forward to satisfy maximum predecessor depth
            super_layer[nxt_s] = max(super_layer.get(nxt_s, 0), curr_l + 1)
            
            in_degree[nxt_s] -= 1
            if in_degree[nxt_s] == 0:
                queue.append(nxt_s)

    # 4. Map super-node layers back to individual graph vertices
    layer = {}
    for s_root, members in super_nodes.items():
        l_val = super_layer.get(s_root, 0)
        for v in members:
            layer[v] = l_val

    max_layer = max(layer.values())
    for v in g.outputs():
        layer[v] = max_layer

    # 5. Set rows in PyZX Graph
    for v in g.vertices():
        if v in layer:
            g.set_row(v, layer[v])

    return layer


import pyzx as zx

def assign_simple_layers(g):
    """
    Traverses the graph qubit by qubit line.
    For each qubit line:
      - Starts with the input boundary node at layer 0 (row 0).
      - Increments the layer count by 1 for each subsequent node on that line.
      - Sets each vertex's row value to its layer index.

    Returns:
      layer: dict mapping vertex -> assigned layer (row) index.
    """
    layer = {}
    
    # Identify unique qubit lines
    qubits = sorted(list(set(g.qubits().values())))

    for q in qubits:
        if q < 0:
            continue

        # Get all vertices on qubit q, sorted by their current row
        q_nodes = [v for v in g.vertices() if g.qubit(v) == q]
        q_nodes.sort(key=lambda v: g.row(v))

        # Assign strictly increasing layer indices along the qubit line
        for l_idx, v in enumerate(q_nodes):
            layer[v] = l_idx
            g.set_row(v, l_idx)

    return layer


