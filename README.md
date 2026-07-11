# 3D Spatial Routing and Embedding of ZX-Graphs


## Usage
To execute the 3D pipeline on an OpenQASM benchmark file (e.g., a 16-qubit GHZ state or Bernstein-Vazirani circuit), run:

```bash
python main.py
```

### Modifying Benchmarks
You can swap out or experiment with alternative quantum circuits inside `main.py`:
```python
# Change circuit source in main.py
circ = zx.Circuit.from_qasm_file("benchmark/your_circuit.qasm")
```

---

## Sample Metrics & Output

When execution finishes, the pipeline yields analytical details and opens an interactive 3D visualization window:

```text
Routed : 2 -> 14
Routed : 5 -> 18
Success: True

Bounding box
  X: [0, 6]  size = 7
  Y: [0, 6]  size = 7
  Z: [0, 24]  size = 25
Bounding box volume = 1225
```

- **Red Cubes/Faces:** $X$-spider nodes/boundaries.
- **Blue Cubes/Faces:** $Z$-spider nodes/boundaries.
- **Yellow Pipes:** Hadamard edge connections.

---



# Detailed Algorithm Workflow

## 1. Graph Simplification & Spider Splitting

Before embedding into a physical 3D grid, the raw quantum circuit is converted into a graph using PyZX. It passes through initial ZX-calculus identity and spider simplifications to contract redundant structures.

Because physical hardware topologies or routing grids cannot accommodate arbitrary $N$-degree connectivity at a single coordinate, the algorithm restricts the graph's maximum vertex degree.

```
  High-Degree Spider (e.g., Degree 6)           Bounded Chain Decomposition (Degree 3)
           \   |   /                                       \   /
            \  |  /                                         \ /
           ---(v)---          ==========>                   (v1)---(v2)---(v3)
            /  |  \                                                /   \   |
           /   |   \                                              /     \  |

```

* **Target Constraints:** The algorithm decomposes vertices into a chain of nodes with a maximum degree of either 3 (`three_deg_simp.py`) or 4 (`four_deg_simplify.py`).


* **Chain Generation:** For a high-degree vertex $v$, its neighbors are sorted sequentially by their timeline row value.


* **Properties Preservation:** The original vertex is removed. It is replaced by a linear sequence of $n$ new vertices linked by chain edges. The original gate phase is retained exclusively on the first node in the chain ($\text{phase value} = \text{phase}$ if $i == 0$ else $0$) to maintain semantic equality.



---

## 2. Spatial 3D Grid Assignment

Once the graph satisfies the degree constraints, it maps the abstract topological nodes to localized 3D voxel coordinates:

* **2D Qubit Layout ($X, Y$ Floorplan):** Qubits are assigned to a square 2D floor grid. The layout width $k$ is calculated based on the maximum number of qubits: $k = \lceil\sqrt{Q}\rceil$. The spatial $x$ and $y$ positions use a spacing scale factor `STEP`:



$$x = \text{STEP} \times (q // k)$$


$$y = \text{STEP} \times (q \% k)$$


* **Temporal Axis ($Z$-unrolling):** The execution timeline flows along the $Z$-axis. Graph vertices are grouped dynamically by their sequence layers. As nodes in a layer are physically placed at the current $Z$-level, the algorithm increments the layer index to the next available vertical ceiling ($\text{max z} + 1$) to ensure routing clearance.



---

## 3. Stateful Orientation-Aware A* Search

Unlike standard 3D pathfinders that only search for a physical coordinate $(x, y, z)$, this routing engine must preserve specific quantum parity alignments. Paths represent topological interactions, meaning wires entering or exiting a node must match the appropriate color face of a ZX-spider ($X$-red vs. $Z$-blue).

### The Search State

Every step in the A* queue evaluates an enhanced 4-tuple state space:


$$\text{State} = (\text{Position}, \text{Orientation}, \text{Direction}, \text{Type})$$


Where:

* **Position:** The physical $(x,y,z)$ coordinates in the grid.


* **Orientation:** A string tracking the alignment of the 3 fundamental structural axes (e.g., `"XZZ"` or `"XZX"`).


* **Direction:** The vector along which the path arrived at this voxel.


* **Type:** Tracks the vertex behavior ($X$-spider, $Z$-spider, or `0` for neutral/boundary nodes).



### Search Transitions and Rotations

As the pathfinder progresses through neighboring coordinates, the orientation of the tracking frame dynamically shifts.

* When changing direction (e.g., executing a right-angle bend from the $+X$ direction to the $+Z$ direction), the algorithm updates the orientation metadata.


* It swaps the tracking indices using a `SEARCH_TRANSITIONS` lookup table to mirror physical face rotation:



```python
# Swapping the face tracking properties when the path bends
o[old_axis], o[new_axis] = o[new_axis], o[old_axis]

```

### Goal Validation

A path cannot simply terminate at the target position; it must hit the correct interface type. The algorithm checks that the path's entry axis matches the exact opposite properties requested by the target node's type configuration ($X$-spider connects to $Z$-faces, and vice versa).

---

## 4. Multi-Stage Routing Execution Loop

The overall pipeline maps out paths by strictly tracking obstacles across two separate stages in `main.py`:

1. **Gate / Inter-Qubit Routing:** The layout loops through the logical circuit layers. For every interaction edge between different qubits, the A* engine finds an optimal 3D path through the grid. Once a path is established, its intermediate points are added to an `occupied` blacklist to prevent future routes from causing cross-talk or physical collisions.


2. **Straight Line Qubit Wires:** After resolving cross-qubit gates, the engine handles long-term state preservation along a single qubit. These connections act as direct structural columns that run vertically along the $Z$-axis, connecting a qubit's past interaction layer to its next layer.