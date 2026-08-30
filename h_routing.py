from pyzx.utils import VertexType
from visualize import Cube

def assign_node_orientations(g):
    """
    Walks each qubit line left-to-right (by row) and assigns a face
    orientation string to every node.

    Rules:
      - BOUNDARY (input/output) nodes: fixed orientation "000"
      - H_BOX nodes: fixed orientation "HHH"
      - X nodes: start at "XZZ", flipping to "ZXZ" after each H_BOX seen
                 on that qubit line (and back to "XZZ" after the next, etc.)
      - Z nodes: start at "XZX", flipping to "ZXX" after each H_BOX seen
                 on that qubit line (and back to "XZX" after the next, etc.)

    Returns:
      orientation: dict mapping vertex -> orientation string
    """
    orientation = {}
    qubits = sorted({g.qubit(v) for v in g.vertices() if g.qubit(v) >= 0})

    for q in qubits:
        verts = [v for v in g.vertices() if g.qubit(v) == q]
        verts.sort(key=g.row)

        flipped = False

        for v in verts:
            v_type = g.type(v)

            if v_type == VertexType.BOUNDARY:
                orientation[v] = "000"

            elif v_type == VertexType.H_BOX:
                orientation[v] = "HHH"
                flipped = not flipped  # toggle for subsequent nodes on this line

            elif v_type == VertexType.X:
                orientation[v] = "ZXZ" if flipped else "XZZ"

            elif v_type == VertexType.Z:
                orientation[v] = "ZXX" if flipped else "XZX"

            else:
                # fallback for any unexpected vertex type
                orientation[v] = "000"

    return orientation

def h_routing(pos):
    """
    Places and draws an H-type cube at the specified position.
    
    Parameters:
        pos (tuple): (x, y, z) coordinate tuple.
        ax (matplotlib axis): The 3D subplot axis to draw on.
        
    Returns:
        Cube: The created H-type cube instance.
    """
    h_cube = Cube(
        pos=pos,
        orientation="HHH",
        dir=None,
        type="H",
    )
    
    
    return h_cube
