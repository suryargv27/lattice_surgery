# Lattice Surgery Compilation Pipeline: Technical Overview

This codebase compiles a ZX-calculus circuit representation into a 3D spatiotemporal lattice-surgery layout for fault-tolerant quantum computation, with two magic-state supply strategies (cultivation vs. distillation). The pipeline has five conceptual stages.

## 1. Circuit Preprocessing (ZX-calculus)

Starting from a PyZX graph $g$ (random Clifford+T circuit via `zx.generate.cliffordT`), three normalizations occur:

- **`create_hboxes`**: Hadamard edges (implicit basis changes) are reified as explicit `H_BOX` vertices, so all graph edges become "simple," making the geometry-layer routing logic edge-type-agnostic.
- **`simplify_green_spiders`**: Along each qubit's wire, maximal runs of degree-2 Z-spiders are fused by phase addition mod $2\pi$ — a direct application of the ZX spider-fusion rule $\alpha \cdot \beta \to \alpha+\beta$. Phases landing on $3\pi/4$ or $5\pi/4$ are split via `split_3pi4_and_5pi4_nodes` into a Clifford component ($\pi/2$ or $\pi$) followed by a $\pi/4$ residual, since the downstream lattice-surgery gadgets are built around canonical T-gates ($\pi/4$).
- **`assign_layers`**: Vertices are grouped into union-find equivalence classes across CNOT (cross-qubit) edges — since a 2-qubit interaction must occupy one synchronized time-slice — then longest-path (ASAP) scheduled via topological layering of the resulting DAG of super-nodes. This gives the minimum-depth valid schedule.

## 2. Pauli Frame Tracking

`track_pauli_frames` implements a stabilizer/Pauli-frame propagation pass: explicit $\pi$-phase spiders are absorbed as classically-tracked $X$/$Z$ frame flips (removing the vertices), then a left-to-right sweep propagates frames through:
- $H$: swap $X\leftrightarrow Z$ frame bits,
- CNOT (Z-control/X-target boundary): $X_{\text{target}} \mathrel{\oplus}= X_{\text{control}}$, $Z_{\text{control}} \mathrel{\oplus}= Z_{\text{target}}$,
- $S$ ($\pi/2$): $Z \mathrel{\oplus}= X$ (frame update for the phase gate),
- $T$ ($\pi/4$): if an active $X$-frame is present, the correction must be physically applied (T is not Pauli-covariant in general), so the node is flagged and the frame is flushed.

This is standard "Pauli frame" bookkeeping to avoid inserting real Pauli-correction circuitry.

## 3. T-depth Analysis

`compute_t_depth` extracts the T-only dependency sub-DAG: an edge $t_i \to t_j$ exists iff a path exists in $g$ from $t_i$ to $t_j$ with no intervening T-node (found by reverse BFS that stops expansion at other T-nodes). Longest-path layering over this sub-DAG gives the **T-depth**, and the widest layer gives the **minimum number of concurrent magic-state factories/cultivation sites** needed to avoid serializing T-gate consumption — a standard resource-estimation metric in surface-code compilation (cf. Litinski's game-of-surface-codes framework).

## 4. Spectral Data-Qubit Placement

This is the most mathematically distinctive part (`spectral_layout.py`). Given the CNOT interaction matrix $W \in \mathbb{R}^{n\times n}$ (weighted by gate count) and T-demand vector $t$, qubits are placed on a 2D lattice to approximately minimize routing cost:

1. **Candidate extraction**: `valid_data_candidates` restricts placement to the largest sublattice parity class $(x,y) \bmod 2$ whose full Moore (8-)neighborhood lies in $S$ — this guarantees enough clearance around each data qubit for later routing/ancilla space.
2. **Compact induced subgraph**: `compact_bfs_set` BFS-grows a connected region $D$ of size $n$ over the *candidate graph* (edges = distance-2 cardinal steps), seeded near the candidate centroid — approximating a compact (low-diameter) placement region.
3. **Spectral embedding**: `spectral_embedding` forms the graph Laplacian $L = \mathrm{diag}(W\mathbf{1}) - W$ and takes the first two non-trivial eigenvectors of $L$ (Fiedler-type embedding) as a 2D layout $Z$ that places highly-interacting qubits near each other — this is the classical spectral graph-drawing heuristic, minimizing $\sum_{ij} W_{ij}\|z_i - z_j\|^2$ subject to orthonormality constraints.
4. **Procrustes alignment + assignment**: $Z$ and the physical coordinates $D$ are each centered/scaled to unit RMS radius, PCA-aligned via SVD, and then matched over the dihedral group of the square (8 sign/swap reflections) to find the best orientation. For each, the **Hungarian algorithm** (`linear_sum_assignment`) solves the exact linear assignment problem $\min_\sigma \sum_i \|z_i - d_{\sigma(i)}\|^2$, and the globally lowest-cost orientation/assignment is kept.
5. **Lattice partitioning** (`build_lattice_sets`): a one-ring Moore dilation around $D$ splits the remaining grid into `bus` (boundary-adjacent, routing exit points) and `ancilla`/interior routing cells, leaving the rest as generic `S`.

## 5. Magic-State Resource Placement (two variants)

Both are **exact combinatorial optimization** problems solved via CP-SAT (Google OR-Tools), maximizing the count of placed resource sites subject to a routing-reachability constraint — solved with a "solve → validate → no-good cut" refinement loop (or a faster pre-conflict-pruned single-solve variant in `factory_solver.solve_max_factories_fast`).

- **Cultivation** (`cultivation.py`): 1×1 sites, each independently reachable to `bus` via 4-connected BFS through unoccupied $S$-cells. Site readiness is modeled as $T_{\text{ready}} = T_{\text{cycles}}/d$ where $T_{\text{cycles}}\sim\mathrm{Exp}(\lambda)$ — an exponential completion-time model (referencing "PureMagic" cultivation), scaled by code distance $d$.
- **Distillation** (`factory_solver.py`/`distillation.py`): 3×3 factory footprints with 8 boundary "output" cells as BFS sources; readiness is deterministic, $T_{\text{ready}} = T_{\text{LATENCY}} = 12$ cycles per factory cycle. The fast solver precomputes single-factory isolated routes, converts spatial cell-overlaps into pairwise ILP conflict constraints *before* the first solve (bounded via a Manhattan-radius spatial hash), so the common case requires only one CP-SAT call rather than iterative cutting-plane rounds.

Both use an integer program: binary $x_i$ per candidate, non-overlap constraints $\sum_{i \ni \text{cell}} x_i \le 1$, objective $\max \sum_i x_i$, with reachability re-validated per solution (adding no-good cuts $\sum_{i\in\text{selected}} x_i \le |\text{selected}|-1$ on failure).

## 6. Geometric Realization

`layer_processing.py` walks topological layers and, per vertex type, emits `Cube`/`Pipe` primitives at $(x,y,z)$ positions (z = logical time):
- **Boundary/H/S nodes**: placed directly or via a small local gadget (`s_routing` fuses an $S$-ancilla with two auxiliary cubes).
- **T-nodes**: two-hop A* routing — first a 2D search from the qubit to any `bus` cell (`a_star_2d`, oriented-face-aware neighbor generation respecting the cube's "usable face" $= \mathrm{OPPOSITE}[\text{type}]$), then bus→ready-resource A* to the nearest ready cultivation site or factory output, consuming it (`site.consume`/`factory.consume`, which restarts its stochastic/deterministic timer).
- **CNOT nodes**: full 3D A* (`route_cnot_minimal_z`) with a restart schedule that sweeps `start_z` upward only when the entire `goal_z` band at the current start fails, guaranteeing the minimum-$z$ solution found is optimal for that start rung.

All A* variants use orientation-transition tables (`SEARCH_TRANSITIONS`/`DRAW_TRANSITIONS`) precomputed over the 8 possible face-orientation strings × 6 directions, encoding how a lattice-surgery "pipe" rotates the logical operator support as it moves through 3D space — this is the combinatorial heart of enforcing valid lattice-surgery topology (no illegal face/pipe mismatches).

## 7. Validation & Reporting

`reconstruct_zx.py` inverts the geometry back into a PyZX graph (one spider per unique `(x,y,z)`, typed by inferring the "odd-one-out" face from the orientation string), enabling `zx.compare_tensors` to numerically verify the compiled 3D structure implements the same unitary as the original circuit — a correctness check via tensor-network contraction equivalence, used in `main_clifford.py`'s sanity check for the Clifford-only pipeline.

---

**In summary**: ZX normalization → Pauli-frame/T-depth analysis → spectral-graph-theoretic qubit placement (Laplacian embedding + Procrustes + Hungarian assignment) → exact ILP-based ancilla-resource packing with reachability constraints → oriented 3D A* lattice-surgery routing → tensor-network round-trip verification.
