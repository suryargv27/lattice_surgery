from fractions import Fraction
import pyzx as zx
from pyzx.utils import VertexType

def is_pi_phase(phase):
    """Check if phase is equivalent to pi (1.0 in PyZX fraction)."""
    if phase is None:
        return False
    return (Fraction(phase) % 2) == 1

def is_pi_2_phase(phase):
    """Check if phase is equivalent to pi/2 (0.5 or 1.5 in PyZX fraction)."""
    if phase is None:
        return False
    frac = Fraction(phase) % 2
    return frac in (Fraction(1, 2), Fraction(3, 2))

def is_pi_4_phase(phase):
    """Check if phase is equivalent to pi/4 (T-gate family)."""
    if phase is None:
        return False
    frac = Fraction(phase) % 2
    return frac.denominator == 4


def track_pauli_frames(g):
    """
    Performs Pauli frame tracking on a PyZX graph where Hadamards are explicit H_BOX nodes.
    
    Returns:
        pauli_record: dict mapping qubit -> [X_bit, Z_bit]
        pending_x_t_nodes: list of T-gate node IDs that encountered an active X-frame.
    """
    qubits = sorted(list(set(g.qubits().values())))
    
    # Initialize Pauli records for all qubits to I: [0, 0]
    # Format: {qubit: [X_bit, Z_bit]}
    pauli_record = {q: [0, 0] for q in qubits if q >= 0}
    
    # Track T-gate node IDs that require a pending X-gate flush
    pending_x_t_nodes = []

    # Step 1: Detect explicit Pauli pi-nodes, update frame, and remove nodes
    vertices_to_remove = []
    for v in list(g.vertices()):
        q = g.qubit(v)
        vtype = g.type(v)
        phase = g.phase(v)

        if q >= 0 and is_pi_phase(phase):
            if vtype == VertexType.Z:      # Pauli Z (Green pi)
                pauli_record[q][1] ^= 1
                vertices_to_remove.append(v)
            elif vtype == VertexType.X:    # Pauli X (Red pi)
                pauli_record[q][0] ^= 1
                vertices_to_remove.append(v)

    # Cleanly remove absorbed Pauli vertices and re-connect neighbors directly
    for v in vertices_to_remove:
        nbrs = list(g.neighbors(v))
        if len(nbrs) == 2:
            g.remove_vertex(v)
            g.add_edge((nbrs[0], nbrs[1]))

    # Step 2: Sweep remaining vertices topologically left-to-right (by row index)
    sorted_vertices = sorted(list(g.vertices()), key=lambda v: g.row(v))

    for v in sorted_vertices:
        q = g.qubit(v)
        if q < 0:
            continue

        vtype = g.type(v)
        phase = g.phase(v)

        # -------------------------------------------------------------
        # 1. H_BOX Node Check: Swaps X and Z frame components
        # -------------------------------------------------------------
        if vtype == VertexType.H_BOX:
            pauli_record[q][0], pauli_record[q][1] = pauli_record[q][1], pauli_record[q][0]
            continue

        # -------------------------------------------------------------
        # 2. CNOT Cross-Qubit Propagation
        # -------------------------------------------------------------
        for nbr in g.neighbors(v):
            nbr_q = g.qubit(nbr)
            if nbr_q >= 0 and nbr_q != q and g.row(v) <= g.row(nbr):
                if vtype == VertexType.Z and g.type(nbr) == VertexType.X:
                    # Control = q (Z), Target = nbr_q (X)
                    pauli_record[nbr_q][0] ^= pauli_record[q][0]  # Target X ^= Control X
                    pauli_record[q][1] ^= pauli_record[nbr_q][1]  # Control Z ^= Target Z

        # -------------------------------------------------------------
        # 3. S-Gate (pi/2 phase): Z ^= X
        # -------------------------------------------------------------
        if is_pi_2_phase(phase):
            pauli_record[q][1] = pauli_record[q][0] ^ pauli_record[q][1]

        # -------------------------------------------------------------
        # 4. T-Gate (pi/4 phase): Z passes through; record & flush X
        # -------------------------------------------------------------
        elif is_pi_4_phase(phase):
            # Z-frame passes through unchanged
            
            # If there is an active X-frame, record T-gate node ID and flush X
            if pauli_record[q][0] == 1:
                pending_x_t_nodes.append(v)
                pauli_record[q][0] = 0  # Flush / Clear X-frame

    return pauli_record, pending_x_t_nodes