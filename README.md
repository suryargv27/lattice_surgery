# 3D Spatial Routing and Embedding of ZX-Graphs

A comprehensive quantum circuit routing and 3D spatial embedding framework for lattice surgery operations. This project implements advanced algorithms for mapping logical ZX-graph representations onto physical 3D grids with support for Pauli frame tracking, multi-stage routing, and magic state distillation.

## Overview

This codebase provides:
- **ZX-Graph Simplification** via PyZX with configurable degree constraints (3 or 4)
- **3D Spatial Grid Assignment** with 2D qubit floorplan and Z-axis time-unrolling
- **Stateful A* Pathfinding** with orientation tracking and parity preservation
- **Multi-Stage Routing** for inter-qubit gates and qubit wire connections
- **Pauli Frame Tracking** for implicit gate compensation
- **Visualization Framework** with interactive 3D Matplotlib rendering
- **Magic State Distillation** and **Cultivation** mechanics for T-gate support

**Language Composition:**
- Python: 69.6%
- OpenQASM: 30.4%

## Quick Start

### Basic Usage

To execute the 3D pipeline on an OpenQASM benchmark file:

```bash
python main.py
```

### Modifying Benchmarks

Swap out or experiment with alternative quantum circuits inside `main.py`:

```python
# Change circuit source in main.py
circ = zx.Circuit.from_qasm_file("benchmark/your_circuit.qasm")
```

### Alternative Entry Points

The repository includes specialized entry points:

```bash
# Clifford circuit routing
python main_clifford.py

# Magic state distillation pipeline
python main_distillation.py

# Cultivation-based T-gate support
python main_cultivation.py
```

---

## Sample Output

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

**Visualization Legend:**
- **Red Cubes/Faces:** X-spider nodes/boundaries
- **Blue Cubes/Faces:** Z-spider nodes/boundaries
- **Yellow Pipes:** Hadamard edge connections

---

## Core Modules

### Graph Processing & Simplification

#### `pft.py` - Pauli Frame Tracking
Performs Pauli frame tracking on PyZX graphs with explicit Hadamard nodes. Tracks X and Z frame bits across qubits, handles CNOT propagation, and identifies T-gate nodes requiring X-frame compensation.

**Key Functions:**
- `track_pauli_frames()` - Core frame tracking algorithm
- `is_pi_phase()`, `is_pi_2_phase()`, `is_pi_4_phase()` - Phase detection helpers

#### `simplify.py` - Degree Reduction
Decomposes high-degree vertices into chains of bounded-degree nodes (configurable max degree of 3 or 4). Preserves all gate phases and maintains graph semantics.

**Key Components:**
- Vertex degree analysis and decomposition
- Chain generation with sequential neighbor ordering
- Phase retention on original vertices

### Spatial Embedding

#### `layout.py` - 2D Qubit Layout
Assigns qubits to a 2D square floor grid with computed spacing. Calculates optimal grid dimensions based on qubit count and performs mapping to (X, Y) coordinates.

**Key Calculations:**
- Grid width: `k = ⌈√Q⌉`
- Position: `x = STEP × (q div k)`, `y = STEP × (q mod k)`

#### `geometry.py` - Spatial Utilities
Provides bounding box computations, spatial ring generation, and volume calculations for embedded circuits.

**Key Functions:**
- `get_bounding_square_ring()` - Computes perimeter positions
- `compute_bounding_volume()` - Calculates bounding box dimensions

#### `tracker.py` - Obstacle Management
Manages qubit Z-levels and tracks placed geometry obstacles. Maintains free space tracking for both qubit columns and extra reserved regions (cultivation sites, distillation factories).

**Key Class:**
- `QubitTracker` - Obstacle tracking and availability checking

### Routing Engines

#### `cnot_routing.py` - CNOT/Multi-Qubit Gate Routing
Implements stateful orientation-aware A* search for inter-qubit gate connections. Maintains 4-tuple state space: `(Position, Orientation, Direction, Type)`.

**Key Features:**
- Orientation-aware pathfinding
- Face rotation tracking via `SEARCH_TRANSITIONS` lookup
- Goal validation with type matching

**Key Functions:**
- `a_star_3d()` - Main 3D A* pathfinding
- `get_neighbors_3d()` - State transition generation

#### `t_routing.py` - T-Gate/Single-Qubit Routing
Specialized 2D routing for single-qubit connections within fixed Z layers. Used for magic state T-gate injection and consumption.

**Key Functions:**
- `a_star_2d()` - 2D A* with orientation constraints
- `heuristic_2d()` - Manhattan distance heuristic

#### `s_routing.py` - S-Routing Framework
Implements S-routing structures for specific quantum operations. Includes 2-cube and 3-cube placement strategies with automatic Z-level climbing for obstacle avoidance.

#### `h_routing.py` - Hadamard Gate Routing
Specialized routing for Hadamard operations, including routing and visualization helpers.

### Layer Processing

#### `layer_processing.py` - Circuit Layer Management
Orchestrates the multi-stage routing pipeline. Processes logical circuit layers sequentially, routing inter-qubit gates then single-qubit wires.

**Pipeline Stages:**
1. **Gate Routing:** A* finds optimal paths for each interaction edge
2. **Wire Routing:** Direct columns along single-qubit timelines

**Key Functions:**
- `process_all_layers()` - Main orchestration
- Layer-by-layer execution with obstacle updates

#### `column_fill.py` - Qubit Column Filling
Fills columns connecting boundary nodes along qubit timelines. Handles orientation flips for Hadamard nodes and maintains continuity.

**Key Functions:**
- `build_qubit_column_index()` - Maps coordinates to qubits
- `fill_qubit_columns()` - Fills gaps in columns

### Layout Regions

#### `create_rings.py` - Layout Region Computation
Partitions 2D qubit space into functional regions: data qubits, routing ancilla, bus sites, and cultivation zones.

**Regions:**
- **Data:** Logical qubits
- **Routing:** Interior ancilla for cross-qubit connections
- **Bus:** Outer ancilla touching open space
- **Cultivation:** Magic state preparation sites

**Key Functions:**
- `compute_layout_regions()` - Region partitioning
- `plot_layout_regions()` - Region visualization

### T-Gate Support Mechanics

#### `cultivation.py` - Cultivation-Based T-Gates
Implements cultiva ion sites for sequential T-gate magic state generation. Each site occupies a 1×1 qubit column.

**Key Class:**
- `CultivationSite` - Per-site tracking

#### `distillation.py` - Distillation Factory System
Implements 3×3 factory footprints for batch magic state production with latency tracking and consumption scheduling.

**Key Components:**
- `DistillationFactory` - Factory state and history
- `DistillationFactoryRegistry` - Multi-factory coordination
- Automatic ready state management
- Per-attempt attempt history tracking

### Utilities & Visualization

#### `utils.py` - Vectorial Operations
Core mathematical utilities for 3D pathfinding:
- **Vectors:** 3D tuple addition, negation, absolute values
- **Orientations:** 8-orientation domain with transition lookups
- **Path Drawing:** Matplotlib rendering of calculated paths

**Constants:**
- `ORIENTATIONS` - All 8 possible orientations (e.g., "XXX", "XZZ")
- `SEARCH_TRANSITIONS` - Orientation updates for path bends
- `DRAW_TRANSITIONS` - Visual rendering adjustments

#### `visualize.py` - 3D Visualization
Matplotlib-based 3D rendering engine with geometry primitives:
- **Cube** - Voxel representation with type-based coloring
- **Pipe** - Edge connections between voxels
- Axis balancing and 3D visualization setup

**Key Classes:**
- `Cube` - Cube geometry with type/color mapping
- `Pipe` - Pipe/edge geometry between positions

#### `reporting.py` - Analysis & Output
Generates performance reports and orchestrates visualization:
- Volume statistics and bounding box analysis
- Integration with cultivation/distillation registries
- Final 3D circuit visualization

**Key Functions:**
- `print_volume_report()` - Performance metrics
- `visualize_circuit()` - Main visualization orchestration

#### `spectral_layout.py` - Advanced Layout Optimization
Implements spectral graph partitioning and optimized qubit placement using advanced graph algorithms.

---

## Algorithm Workflow

### 1. Graph Simplification & Spider Splitting

Before embedding, the raw quantum circuit is converted to a PyZX graph and simplified:
1. Identity and spider reductions via ZX-calculus
2. Hadamard consolidation
3. Degree constraint application (max degree 3 or 4)

High-degree spiders are decomposed into chains:
```
  High-Degree Spider (Degree 6)         Bounded Chain (Degree 3)
           \   |   /                              \   /
            \  |  /              ====>            \ /
           ---(v)---                           (v1)---(v2)---(v3)
            /  |  \                              /   \   |
           /   |   \                            /     \  |
```

**Target Constraints:** Degree-3 or degree-4 decomposition

### 2. Spatial 3D Grid Assignment

Map abstract topological nodes to 3D voxel coordinates:

**2D Qubit Layout (X, Y Floorplan):**
- Qubits assigned to square floor grid
- Layout width: `k = ⌈√Q⌉`
- Position: `x = STEP × (q div k)`, `y = STEP × (q mod k)`

**Temporal Axis (Z-unrolling):**
- Execution timeline flows along Z-axis
- Vertices grouped dynamically by sequence layers
- Nodes placed at increasing Z as timeline progresses

### 3. Stateful Orientation-Aware A* Search

Enhanced 3D pathfinding preserving quantum parity alignments:

**Search State:**
```
State = (Position, Orientation, Direction, Type)
```

- **Position:** Physical (x, y, z) coordinates
- **Orientation:** String tracking axis alignment (e.g., "XZZ", "XZX")
- **Direction:** Vector along which path arrived at voxel
- **Type:** Vertex behavior (X-spider, Z-spider, neutral)

**Search Transitions:**
As pathfinder progresses, orientation shifts dynamically:
- Direction changes trigger orientation updates
- Tracking indices swapped via `SEARCH_TRANSITIONS` lookup
- Mirrors physical face rotation

```python
# Swapping face tracking properties when path bends
o[old_axis], o[new_axis] = o[new_axis], o[old_axis]
```

**Goal Validation:**
Path must terminate at correct interface type with matching entry axis.

### 4. Multi-Stage Routing Execution Loop

Two-stage pipeline in the main execution:

**Stage 1 - Gate/Inter-Qubit Routing:**
- Loop through logical circuit layers
- For each interaction edge between qubits, A* finds optimal 3D path
- Update obstacle tracker with placed paths

**Stage 2 - Straight Line Qubit Wires:**
- Handle long-term state preservation along single qubit
- Connect boundary nodes along each qubit column

---

- Visualization updates are tested with sample benchmarks
- Performance regressions are addressed
