# Embodied AI & Robotics Data — 8-Month Transition Curriculum
## Frontend Track: for the Senior Angular / TypeScript Engineer

**Target role:** Member of Technical Staff, Robotics Research & Data Infrastructure (micro1 and similar AI data labs)
**Realistic first landing spot:** Robotics Data Tooling / Data Platform Engineer at a robotics or AI-data company
**Assumed background:** Senior Angular — TypeScript, RxJS, component architecture, NgRx, build tooling, testing, real-time UIs
**Assumed effort:** 12–15 focused hours/week
**Total length:** 8 months (Month 0 bridge + 7 modules)

---

## Table of Contents

1. [Read this first: honest positioning](#1-read-this-first-honest-positioning)
2. [Your real advantage](#2-your-real-advantage-and-it-is-a-real-one)
3. [Your real gaps](#3-your-real-gaps)
4. [Month 0 — Python, NumPy & Linear Algebra Bridge](#month-0--python-numpy--linear-algebra-bridge)
5. [Module 1 — Spatial Mathematics & Kinematics](#module-1--spatial-mathematics--kinematics)
6. [Module 2 — Robotics Data Protocols & the Dataset Inspector](#module-2--robotics-data-protocols--the-dataset-inspector)
7. [Module 3 — Machine Learning Foundations in PyTorch](#module-3--machine-learning-foundations-in-pytorch)
8. [Module 4 — Policy Learning & VLA Models](#module-4--policy-learning--vla-models)
9. [Module 5 — Physics Simulation & Synthetic Data](#module-5--physics-simulation--synthetic-data)
10. [Module 6 — 3D Vision & Perception](#module-6--3d-vision--perception)
11. [Module 7 — Capstone: Robot Data Studio](#module-7--capstone-robot-data-studio)
12. [Timeline & milestones](#timeline--milestones)
13. [Appendix A — Hardware notes](#appendix-a--hardware--environment-notes)
14. [Appendix B — TypeScript → Python phrasebook](#appendix-b--typescript--python-phrasebook)
15. [Appendix C — Reading list](#appendix-c--reading-list)
16. [Appendix D — Interview prep & positioning](#appendix-d--interview-prep--positioning)
17. [Appendix E — The compressed 6-month variant](#appendix-e--the-compressed-6-month-variant)

---

## 1. Read this first: honest positioning

The micro1 posting asks for "2–5+ years in robotics, computer vision, multimodal learning, embodied AI, or applied ML research." A senior Angular engineer has none of those, and — unlike a backend/data engineer — cannot lean on adjacent production-pipeline credit either. **Eight months of part-time study will not close that gap to the point where you are a first-choice candidate for that specific research role.**

What it *can* do is make you a strong candidate for the role immediately adjacent to it, which is a real and growing job:

> **Robotics data tooling / data platform engineer** — the person who builds the systems that collection operators, annotators, and researchers actually touch: teleop interfaces, dataset browsers, trajectory viewers, annotation studios, quality dashboards, evaluation report UIs.

This is not a consolation prize. Consider what these companies actually run on:

- **Foxglove** — the standard robotics visualisation tool. It is a TypeScript/React web application.
- **Rerun** — the fast-growing robotics/CV logging viewer, with a web (wasm) frontend.
- **LeRobot's dataset visualiser** — a web app.
- **Scale AI, Encord, Segments.ai, Labelbox** — annotation platforms; their core product is frontend engineering over multimodal 3D data.
- **micro1 itself** — is a *platform* company. Read their own description: "Our platform identifies and vets top talent through an AI recruiter, enabling high-quality expert contributions at scale." The human-data-contribution layer *is* a software product, and someone has to build it.

So the strategy is: **enter through tooling, then move toward research.** Land in the robotics data org as the person who builds the instruments, spend 12–18 months absorbing the domain from the inside with real datasets and real researchers next to you, then move to a research-facing seat. That path is well-trodden and far more likely to work than trying to out-credential a robotics PhD from a standing start.

Everything in this curriculum is designed to serve both goals at once — you learn the same robotics substance as the backend track, but your portfolio artefacts are weighted toward the thing you can already do better than almost any robotics candidate.

**One caveat about Angular specifically.** The robotics web-tooling ecosystem is overwhelmingly React (Foxglove, most annotation platforms, most internal dashboards). Your Angular seniority transfers *conceptually* in full — components, DI, reactive state, change detection, testing discipline are the same ideas — but the hiring keyword is usually React. Build the capstone in Angular if that's where you're fastest, then either port one meaningful piece to React or be ready to say, credibly, "the framework is the least interesting part of this; here is the same component in React." Do not let framework labels be the reason you're filtered out.

---

## 2. Your real advantage (and it is a real one)

Three things you already have that most candidates for robotics-data roles do not:

### 2.1 You think natively in asynchronous time-varying streams

This is the single best-kept secret of your background. The central hard problem in robotics data is: *multiple asynchronous sensor streams, arriving at different rates, with independent clocks, that must be aligned into coherent frames.*

You have been solving a structurally identical problem in RxJS for years. The mapping is close enough to be genuinely useful:

| RxJS concept | Robotics data equivalent |
|---|---|
| `Observable<SensorReading>` | A sensor topic / channel |
| `combineLatest` | Nearest-value join across sensors |
| `withLatestFrom` | Sampling slave sensors against a master clock (usually the camera) |
| `sampleTime` / `auditTime` | Resampling to a canonical frame rate |
| `bufferCount(H, 1)` | Sliding action-chunk windows for training |
| `pairwise` | Computing deltas (velocities, delta-poses) |
| Backpressure / `observeOn` | Dataloader throughput and prefetch |
| Marble diagrams | Exactly how you should be reasoning about sensor alignment |
| `scan` | Stateful accumulation (integrating velocity into position) |
| Cold vs hot observables | Replay-from-log vs live-from-robot |

**But learn the one place the intuition misleads you**, because it's an interview-grade distinction:

`combineLatest` emits whenever *any* source emits, pairing the newest value from each. That is a *latest-value* join — correct for a live UI, **wrong for offline dataset construction**. It silently pairs a camera frame with a joint reading that may be 30 ms stale, with the staleness varying frame to frame, and it produces a different result depending on arrival order rather than timestamp order. Offline you need a **time-based join**: for each camera timestamp, interpolate every other channel to that exact time, and record the interpolation distance so you can reject frames where the nearest sample was too far away.

That distinction — *event-order joins versus timestamp joins* — is the difference between a dataset that trains and one that quietly teaches a policy to act on stale observations. You will build the correct version in Module 2.

### 2.2 You can build the instruments

Robotics researchers are, as a population, poor frontend engineers, and they know it. They spend enormous amounts of time squinting at matplotlib plots and printing arrays to terminals because nobody built them something better. A person who can ship a *good* interactive tool for inspecting a 50-GB multimodal dataset — synchronised video scrub, 3D trajectory overlay, per-frame quality flags, keyboard-driven episode triage — is disproportionately valuable and disproportionately rare.

### 2.3 You have production software discipline

Typed interfaces, dependency injection, testing, CI, versioned APIs, reviewable code. Research code is famously none of these things. Bringing schema-as-contract thinking to a dataset pipeline is a genuine contribution, not just tidiness.

---

## 3. Your real gaps

Be clear-eyed. Relative to a backend/Python engineer attempting the same transition, you are starting further back on:

| Gap | Severity | Where it's addressed |
|---|---|---|
| Python fluency (idioms, packaging, tooling) | High | Month 0 |
| NumPy / vectorised array thinking | High — this is the big one | Month 0 |
| Linear algebra (matrices, transforms, decompositions) | High | Month 0 + Module 1 |
| Calculus/gradients intuition | Medium | Month 0 + Module 3 |
| PyTorch, training loops, ML debugging | High | Module 3 (a full extra month vs the backend track) |
| Linux/CLI, SSH, remote GPU workflow | Medium | Month 0 |
| Data engineering at rest (columnar formats, chunking, I/O) | Medium | Module 2 |
| Statistics for experiments (CIs, ablations) | Medium | Module 4 |

**The NumPy gap deserves special mention.** As a TypeScript developer your instinct for "do something to every element" is `array.map()`. In numerical Python, writing a Python loop over a 10-million-element array is 100–1000× slower than the vectorised equivalent, and vectorised code *looks nothing like* the loop. This is not a syntax difference; it is a different way of expressing computation, and it takes weeks to become natural. Month 0 is built around it. Do not skip it — every subsequent module assumes it.

---

## Month 0 — Python, NumPy & Linear Algebra Bridge

> **6 weeks · Objective:** Reach the Python and numerical-computing fluency that the rest of this curriculum assumes. Nothing robotics-specific happens here, and that is fine.

This is the module the backend track doesn't need. Give it the full six weeks.

### Week-by-week

| Week | Focus | Output |
|---|---|---|
| 1 | Python language: types, dataclasses, comprehensions, context managers, modules, `uv`/`ruff`/`pytest` | Port a real TS utility of yours to typed Python with full tests |
| 2 | NumPy I: arrays, dtypes, shapes, indexing, slicing, views vs copies | 40-exercise notebook, no `for` loops allowed |
| 3 | NumPy II: broadcasting, axis semantics, reductions, `einsum`, stacking | Vectorise five loop-based functions; benchmark the speedup |
| 4 | Linear algebra: vectors, matrix multiply as composition, basis/frames, transpose, inverse, determinant, orthogonality | Implement 2D/3D transforms from scratch; visualise in Three.js |
| 5 | Calculus & gradients: derivative as slope, chain rule, gradient descent by hand | Hand-rolled gradient descent fitting a line, then a small MLP with manual backprop |
| 6 | Environment & workflow: WSL2, Linux CLI, SSH, `tmux`, Jupyter, matplotlib, remote GPU basics | Working Linux dev environment + a rented-GPU dry run |

### Key concepts

**Python for someone fluent in TypeScript.** You will be productive in days; the trap is writing TypeScript in Python syntax. Specifically:

- Type hints are *not* enforced at runtime. They're for you, your editor, and `mypy`. Treat them as mandatory anyway — you're building schema-defining tooling, and untyped Python rots faster than untyped JS.
- `@dataclass` (or Pydantic models) is your `interface` + constructor. Pydantic in particular gives you runtime validation, which is exactly what a data schema needs.
- There is no `null`/`undefined` split — just `None`. `Optional[X]` is `X | None`.
- Mutable default arguments are a genuine footgun with no TS analogue: `def f(x=[])` shares one list across all calls. Use `field(default_factory=list)`.
- Packaging: `uv` is the modern `npm`; `pyproject.toml` is `package.json`; virtual environments are non-optional and are closer to `node_modules` being per-project than to a global install.
- `pytest` is `jest` with less ceremony. Fixtures ≈ beforeEach with DI.

See **Appendix B** for a fuller phrasebook.

**NumPy: the actual mental shift.**

The core idea is that an array operation applies to the *whole array at once*, and shape rules determine how arrays of different sizes combine. Broadcasting is the hardest part and the most important:

- Shapes are aligned from the **right**. `(100, 3)` and `(3,)` are compatible; `(100, 3)` and `(100,)` are not.
- A dimension of size 1 is stretched to match; a missing leading dimension is treated as 1.
- `arr[:, None]` inserts an axis — this is how you make `(100,)` into `(100, 1)` so it broadcasts against `(100, 3)`.
- Almost every "shapes not aligned" error is solved by printing `.shape` at each step. Do this constantly; it is not a sign of weakness.

**Linear algebra, framed usefully.** You need surprisingly little, but you need it solid:

- A **matrix multiply is a composition of transformations**, not a grid of numbers. `A @ B` means "do B, then do A."
- A **rotation matrix** is an orthonormal basis written in columns — its columns are where the x, y, z axes end up.
- **Transpose of a rotation is its inverse** (because it's orthonormal). This is why the SE(3) inverse in Module 1 is cheap.
- A **homogeneous 4×4 matrix** packs rotation and translation so that composition is a single multiply.
- The **determinant** of a rotation is +1; if yours drifts, you have accumulated numerical error and need re-orthonormalisation.

If you want visual intuition fast: 3Blue1Brown's *Essence of Linear Algebra*, then immediately implement what you saw. Watching is not learning.

### Applied lab

```python
"""m0-bridge/vectorize.py — the loop-to-vectorized shift, with the timings that prove it."""
import time

import numpy as np


def transform_points_loop(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """The TypeScript instinct: iterate and map. Correct, and far too slow."""
    out = np.empty_like(points)
    for i in range(points.shape[0]):
        homogeneous = np.array([points[i, 0], points[i, 1], points[i, 2], 1.0])
        out[i] = (T @ homogeneous)[:3]
    return out


def transform_points_vectorized(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """The NumPy way: express the whole operation once.

    points: (N, 3)   T: (4, 4)   ->  (N, 3)

    points @ T[:3, :3].T   rotates every point   -> (N, 3) @ (3, 3) = (N, 3)
    + T[:3, 3]             broadcasts the translation across all N rows
    """
    return points @ T[:3, :3].T + T[:3, 3]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(1_000_000, 3))
    T = np.eye(4)
    T[:3, 3] = [1.0, 2.0, 3.0]

    t0 = time.perf_counter(); a = transform_points_loop(pts[:50_000], T); t1 = time.perf_counter()
    t2 = time.perf_counter(); b = transform_points_vectorized(pts, T);   t3 = time.perf_counter()

    print(f"loop      (50k pts): {t1 - t0:.3f}s")
    print(f"vectorized (1M pts): {t3 - t2:.3f}s")
    assert np.allclose(a, b[:50_000])
```

Run it. Internalise the ratio. Then never write the first version again.

```python
"""m0-bridge/broadcasting_drills.py — the shape rules, made concrete."""
import numpy as np

positions = np.random.rand(100, 3)      # 100 points in 3D
centroid = positions.mean(axis=0)        # (3,)   -- axis=0 collapses rows

# Broadcasting: (100, 3) - (3,) aligns from the right -> fine.
centered = positions - centroid

# Per-point distance to origin. axis=1 collapses columns -> (100,)
distances = np.linalg.norm(positions, axis=1)

# WRONG: (100, 3) / (100,) does not align from the right.
# scaled = positions / distances            # ValueError
# RIGHT: insert an axis so (100, 3) / (100, 1) broadcasts.
unit = positions / distances[:, None]

# Pairwise distances between all 100 points, no loops:
# (100, 1, 3) - (1, 100, 3) -> (100, 100, 3) -> norm over last axis -> (100, 100)
pairwise = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)

assert unit.shape == (100, 3)
assert pairwise.shape == (100, 100)
assert np.allclose(np.diag(pairwise), 0)
```

### Deliverables

1. **A ported utility library** — take a real TypeScript utility module you've written, port it to Python with type hints, dataclasses, and `pytest` tests at equivalent coverage. This calibrates your Python against a codebase you already understand.
2. **A vectorisation notebook** — 40 exercises solved without a single Python `for` loop over array elements, with before/after timings on five of them.
3. **A transforms-from-scratch demo** — implement 2D and 3D rotation/translation/scale in pure NumPy, then render the result in a small Three.js page so you can *see* the composition order matter. Building the visual is your advantage; use it from day one.
4. **A working Linux dev environment** — WSL2 with Python, CUDA-enabled PyTorch verified against your GPU, and one successful SSH session to a rented cloud GPU running `nvidia-smi`. Do the rented-GPU dry run now, in Month 0, not in Month 4 under time pressure.

### Self-check

- Explain broadcasting rules from memory, then predict the output shape of `np.ones((8, 1, 6)) * np.ones((7, 1))`.
- What does `axis=0` mean for a `(100, 3)` array, and why?
- Why is `A @ B` "do B then A"?
- Rewrite a nested double loop over pairs of points as a single vectorised expression.
- What's wrong with `def parse(rows, seen=set())`?

---

## Module 1 — Spatial Mathematics & Kinematics

> **Month 1 · Objective:** Rigid-body transformations you can compose in your head, plus the ability to see them — because you can build the viewer.

### Week-by-week

| Week | Focus | Output |
|---|---|---|
| 1 | Rotations: matrices, Euler & gimbal lock, quaternions, axis-angle, 6D representations | Conversion library + property tests |
| 2 | SE(3): homogeneous transforms, frame graphs, the `T_target_source` discipline | Frame-graph resolver |
| 3 | Kinematics: URDF structure, forward kinematics, IK, the Jacobian, singularities | FK implemented against a real URDF |
| 4 | **Leverage week:** interactive frame visualiser in Angular + Three.js | `frame-viz` web tool |

### Key concepts & theory

**Rotation representations and their trade-offs**

- **Rotation matrix** `R ∈ SO(3)` — 9 numbers, 3 degrees of freedom. Composes by multiplication, never singular, but redundant and drifts from orthonormality under repeated numerical operations.
- **Euler angles** — 3 numbers, human-readable, but order-dependent (`xyz` ≠ `zyx`) and subject to **gimbal lock**, where two axes align and you lose a degree of freedom. Never store these in a dataset. Angular developers meet this exact problem in Three.js; it's the same thing.
- **Quaternion** `q = w + xi + yj + zk` — 4 numbers, no gimbal lock, cheap composition and interpolation. Two catches: it must stay unit-norm, and it **double-covers** rotation (`q` and `−q` are the same rotation). If you don't canonicalise the sign, a neural network regressing quaternions sees a discontinuous target and trains badly.
- **Axis-angle / Rodrigues** — 3 numbers, compact, ambiguous at ±π.
- **6D representation** (first two columns of the rotation matrix, re-orthonormalised) — the representation modern policies actually regress, precisely because quaternions and Euler angles are *discontinuous* as functions of rotation and neural networks fit continuous functions. **This is the highest-value fact in the module:** your choice of representation in the dataset schema directly determines whether the model can learn it.

**SE(3) and frame discipline**

- Homogeneous 4×4 matrices let you compose rotation and translation in one multiply: `T_a_c = T_a_b @ T_b_c`.
- Adopt `T_target_source` naming universally and read right-to-left; adjacent indices cancel. This convention alone prevents most frame bugs.
- The inverse is cheap and should be hand-written: `R.T` and `-R.T @ t`.
- A robot rig is a **frame graph**: world → base → each joint → end-effector → gripper, plus base → fixed camera and end-effector → wrist camera. Resolving an arbitrary transform is a graph traversal. As an Angular developer this should feel like a dependency graph, and you should implement it as one.

**Kinematics**

- **URDF** describes the robot as links and joints — it is a tree, serialised as XML. Parsing it and walking the chain is ordinary software engineering; don't be intimidated by the robotics vocabulary.
- **Forward kinematics:** compose the per-joint transforms down the chain. Deterministic, exact, easy.
- **Inverse kinematics:** given a desired end-effector pose, find joint angles. Non-unique (elbow up or down), sometimes unsolvable (out of reach), usually solved numerically via damped least squares on the Jacobian.
- **Jacobian** `J(q)` maps joint velocities to end-effector twist. At a **singularity** it loses rank — the arm physically cannot move in some direction, and IK solvers produce enormous joint velocities. This matters to you as a data person: singularity approaches show up in logs as action spikes, and a naive validator will flag them as sensor errors when they're actually kinematics.

**Time-series synchronisation — where your RxJS background pays**

Real rigs stream at mismatched rates: RGB 30 Hz, depth 30 Hz but phase-offset, joint encoders 100–1000 Hz, IMU 200 Hz, force/torque 500+ Hz. Independent clocks drift. Drivers buffer unpredictably.

The correct offline approach: **choose a canonical clock** (almost always the camera, since you cannot interpolate pixels), then for each canonical timestamp, interpolate every other channel to that exact time — linear for positions and velocities, **SLERP** for quaternions, zero-order hold for discrete signals like gripper open/close. Record the interpolation distance per frame and reject frames where the nearest real sample was too far away.

Never lerp raw quaternion components: the result isn't unit-norm and the implied angular velocity is wrong. SLERP exists for exactly this.

### Applied engineering & code lab

```python
"""m1-spatial/transforms.py — SE(3) utilities."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def se3(translation, rotation: Rotation) -> np.ndarray:
    """Build a 4x4 homogeneous transform."""
    T = np.eye(4)
    T[:3, :3] = rotation.as_matrix()
    T[:3, 3] = np.asarray(translation, dtype=float)
    return T


def invert(T: np.ndarray) -> np.ndarray:
    """Inverse of a homogeneous transform. Uses R.T == R^-1 for orthonormal R."""
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def slerp(q_start: np.ndarray, q_end: np.ndarray, alpha: float) -> np.ndarray:
    """Spherical linear interpolation, scipy quaternion order [x, y, z, w].

    Note: scipy exposes Slerp as its own class. `Rotation` has no `.slerp`
    method, despite a great many code samples online claiming otherwise.
    """
    keys = Rotation.from_quat(np.stack([q_start, q_end]))
    return Slerp([0.0, 1.0], keys)(float(alpha)).as_quat()


def canonicalize_quat(q: np.ndarray) -> np.ndarray:
    """Resolve the q/-q double cover by forcing a non-negative scalar part."""
    q = np.asarray(q, dtype=float)
    return -q if q[..., 3] < 0 else q


def to_6d(R: np.ndarray) -> np.ndarray:
    """Continuous 6D rotation representation: first two columns of R."""
    return R[:3, :2].T.reshape(6)


def from_6d(v: np.ndarray) -> np.ndarray:
    """Recover a rotation matrix from the 6D representation (Gram-Schmidt)."""
    a1, a2 = v[:3], v[3:]
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - (b1 @ a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)
```

```python
"""m1-spatial/frame_graph.py — resolve transforms by graph traversal.

This is a dependency graph. You have built these before.
"""
from collections import deque

import numpy as np


class FrameGraph:
    def __init__(self) -> None:
        self._edges: dict[str, dict[str, np.ndarray]] = {}

    def add(self, target: str, source: str, T: np.ndarray) -> None:
        """Register T_target_source; the reverse edge is added automatically."""
        self._edges.setdefault(target, {})[source] = T
        self._edges.setdefault(source, {})[target] = invert(T)

    def resolve(self, target: str, source: str) -> np.ndarray:
        """Find T_target_source by BFS over the frame graph."""
        if target == source:
            return np.eye(4)
        queue = deque([(source, np.eye(4))])
        seen = {source}
        while queue:
            node, acc = queue.popleft()
            for nxt, T in self._edges.get(node, {}).items():
                if nxt in seen:
                    continue
                composed = T @ acc          # T_nxt_node @ T_node_source
                if nxt == target:
                    return composed
                seen.add(nxt)
                queue.append((nxt, composed))
        raise KeyError(f"no transform path from {source!r} to {target!r}")
```

**Leverage week — build the viewer.** A minimal Angular + Three.js component that renders coordinate frames and lets you scrub a trajectory. This takes you two days and gives you an intuition pump that people without your background simply don't have:

```typescript
// m1-spatial/viz/src/app/frame-viewer.component.ts
import { Component, ElementRef, Input, OnDestroy, OnInit, ViewChild } from '@angular/core';
import * as THREE from 'three';

export interface Pose {
  /** seconds, on the canonical clock */
  t: number;
  /** metres, [x, y, z] */
  position: [number, number, number];
  /** quaternion, [x, y, z, w] — same order as scipy */
  quaternion: [number, number, number, number];
}

@Component({
  selector: 'app-frame-viewer',
  standalone: true,
  template: `<div #canvasHost class="viewer"></div>`,
  styles: [`.viewer { width: 100%; height: 480px; display: block; }`],
})
export class FrameViewerComponent implements OnInit, OnDestroy {
  @ViewChild('canvasHost', { static: true }) canvasHost!: ElementRef<HTMLDivElement>;

  /** Full end-effector trajectory for one episode. */
  @Input({ required: true }) trajectory!: Pose[];

  private renderer!: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera!: THREE.PerspectiveCamera;
  private eeAxes = new THREE.AxesHelper(0.1);
  private frameId = 0;

  ngOnInit(): void {
    const host = this.canvasHost.nativeElement;
    this.camera = new THREE.PerspectiveCamera(50, host.clientWidth / host.clientHeight, 0.01, 50);
    this.camera.position.set(0.8, 0.8, 0.8);
    this.camera.lookAt(0, 0, 0);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(host.clientWidth, host.clientHeight);
    host.appendChild(this.renderer.domElement);

    // World origin, end-effector frame, and the swept path.
    this.scene.add(new THREE.AxesHelper(0.2));
    this.scene.add(this.eeAxes);
    this.scene.add(this.buildPathLine());

    this.renderer.setAnimationLoop(() => this.renderer.render(this.scene, this.camera));
  }

  /** Scrub to a frame — wire this to a slider or to video currentTime. */
  seek(index: number): void {
    const pose = this.trajectory[Math.max(0, Math.min(index, this.trajectory.length - 1))];
    this.eeAxes.position.fromArray(pose.position);
    // Three.js Quaternion is (x, y, z, w) — matching scipy's convention, not w-first.
    this.eeAxes.quaternion.fromArray(pose.quaternion);
    this.frameId = index;
  }

  private buildPathLine(): THREE.Line {
    const points = this.trajectory.map((p) => new THREE.Vector3(...p.position));
    return new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(points),
      new THREE.LineBasicMaterial({ color: 0x4f8cff }),
    );
  }

  ngOnDestroy(): void {
    this.renderer?.setAnimationLoop(null);
    this.renderer?.dispose();
  }
}
```

### Deliverables

1. **`spatial-py`** — tested SE(3)/rotation library including 6D conversions and quaternion canonicalisation, with property tests (round-trip conversions, composition associativity, `invert(T) @ T == I`, SLERP endpoint and constant-angular-velocity properties).
2. **`frame-graph`** — the BFS resolver above, with a URDF parser that populates it from a real robot description file.
3. **`frame-viz`** — the Angular/Three.js viewer, published to GitHub Pages, loading a real trajectory. **This is your first differentiated artefact — ship it publicly in Month 1.**

### Self-check

- Why is a quaternion a poor regression target for a neural network, and what do modern policies use instead?
- Given `T_world_base`, `T_base_cam`, `T_cam_object`, write `T_world_object`.
- Why can't you linearly interpolate a gripper open/close signal?
- Two sensors are offset by a constant 40 ms. How do you detect that from the data alone?
- Your validator flags a burst of huge joint velocities. Name two causes that are *not* sensor faults.

---

## Module 2 — Robotics Data Protocols & the Dataset Inspector

> **Month 2 · Objective:** Learn the formats the field actually uses, build a correct time-based synchroniser, and ship a web dataset inspector that is better than what most labs have internally.

This is your strongest module. Spend the extra energy here.

### Week-by-week

| Week | Focus | Output |
|---|---|---|
| 1 | LeRobot format, Open X-Embodiment, RLDS; action space conventions | Load and dissect a real LeRobot dataset |
| 2 | Storage engineering: HDF5, Zarr, Parquet, MCAP, WebDataset; chunking and I/O | Format benchmark write-up |
| 3 | Time-based synchronisation (the correct version of `combineLatest`) | `traj-sync` + validator |
| 4 | **Leverage week:** the web dataset inspector | `dataset-inspector` shipped |

### Key concepts & theory

**The formats**

- **Open X-Embodiment (OXE)** — 1M+ trajectories pooled from 20+ robot platforms and 60+ source datasets. The valuable thing to study is not the data but the *normalisation problem*: every contributing lab used different action spaces, control rates, camera counts, coordinate conventions and gripper semantics. Reconciling that is literally the job you're aiming at. Read the appendix where they describe the harmonisation decisions.
- **RLDS** — the TFDS-based serialisation OXE originally shipped in. Episode = sequence of steps; step = `{observation, action, reward, is_first, is_last, is_terminal}`.
- **LeRobot** (Hugging Face) — the current community default. Tabular frames in Parquet plus separately encoded video:
  - `observation.images.<camera>` — MP4/AV1 encoded, decoded lazily per frame
  - `observation.state` — proprioception vector
  - `action` — commanded action vector
  - `episode_index`, `frame_index`, `index`, `timestamp`, `task_index`
  - `meta/info.json` (fps, feature shapes, video specs), `meta/episodes.jsonl`, `meta/tasks.jsonl`, `meta/stats.json` (per-feature statistics used for normalisation at train time)
  - Understand the trade: 50 episodes of raw frames is ~100 GB; video-encoded it's ~2 GB. The cost is that random frame access requires keyframe-aware seeking, which often dominates dataloader time. **This is a frontend-adjacent problem you can reason about better than most** — it's the same problem as scrubbing video in a browser, and the same solution (keyframe interval tuning, prefetch windows).

**Storage formats and when each wins**

| Format | Strength | Use when |
|---|---|---|
| **HDF5** | Hierarchical, chunked, one file per episode | Local episode archives; the ALOHA/ACT ecosystem |
| **Zarr** | Chunked like HDF5 but cloud-native and parallel-write-safe | Large generation runs, S3/GCS-backed |
| **Parquet** | Columnar; excellent for tabular state/action + metadata | LeRobot's tabular half; analytics across episodes |
| **MCAP** | ROS 2-native, self-describing schemas, append-only, seekable | Real-robot logging; anything touching Foxglove |
| **WebDataset / TFRecord** | Sharded sequential records, saturates network I/O | Distributed multi-node training |

Failure modes to know: HDF5 has no safe concurrent write; Zarr chunk sizes mismatched to the access pattern destroy throughput; MCAP without registered schemas becomes unreadable later; WebDataset offers no random access, so shuffling must be buffer-based.

**Action space formulations**

- **Absolute joint positions** — easy to learn, embodiment-specific, doesn't transfer.
- **Joint velocities** — needs a well-defined control rate; latency-sensitive.
- **End-effector absolute pose** — transfers better across similar arms.
- **End-effector delta pose** `(Δx, Δy, Δz, Δrot)` — the most common VLA action space; scale-sensitive, so normalisation statistics matter enormously.
- **Action chunking** — predicting `H` future actions at once (H ≈ 8–100) rather than one, which reduces compounding error and smooths motion. Your storage layout must therefore support efficient windowed reads. (In RxJS terms: your dataset must cheaply serve `bufferCount(H, 1)`.)
- **Normalisation** — min/max vs mean/std vs quantile. Percentile clipping (1st/99th) is standard, because one teleop glitch otherwise sets the scale for the whole dataset.

**What actually goes wrong in real robot data**

Dropped and duplicated frames; timestamps going backwards; action spikes from teleop disconnects; frozen sensor channels returning an identical buffer for 40 frames; episodes labelled success that are failures; mismatched episode lengths across modalities; and the classic dataset-killer — a silent unit change mid-collection (millimetres to metres, degrees to radians).

### Applied engineering & code lab

**The correct synchroniser — the thing your RxJS instinct gets subtly wrong:**

```python
"""m2-protocols/sync.py — timestamp-based alignment, not latest-value joins.

RxJS combineLatest pairs the newest value from each stream whenever ANY stream
emits. That is an event-ORDER join: correct for a live dashboard, wrong for
building a dataset. Offline, we interpolate every channel to the canonical
clock and record how far the nearest real sample was, so bad frames can be
rejected rather than silently accepted.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


@dataclass
class Channel:
    name: str
    timestamps: np.ndarray            # (N,) seconds, strictly increasing
    values: np.ndarray                # (N, D)
    kind: str = "linear"              # 'linear' | 'zoh' | 'slerp'


def align(
    channels: list[Channel],
    canonical_times: np.ndarray,
    max_gap_s: float = 0.02,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Resample every channel onto the canonical clock.

    Returns (aligned_values_by_name, valid_mask). Frames where any channel's
    nearest real sample was further than `max_gap_s` away are marked invalid.
    """
    aligned: dict[str, np.ndarray] = {}
    valid = np.ones(canonical_times.shape[0], dtype=bool)

    for ch in channels:
        if np.any(np.diff(ch.timestamps) <= 0):
            raise ValueError(f"channel {ch.name!r} has non-monotonic timestamps")

        # How far is each canonical time from the nearest real sample?
        idx = np.searchsorted(ch.timestamps, canonical_times)
        idx = np.clip(idx, 1, len(ch.timestamps) - 1)
        gap = np.minimum(
            np.abs(canonical_times - ch.timestamps[idx - 1]),
            np.abs(ch.timestamps[idx] - canonical_times),
        )
        valid &= gap <= max_gap_s

        t = np.clip(canonical_times, ch.timestamps[0], ch.timestamps[-1])
        if ch.kind == "slerp":
            aligned[ch.name] = Slerp(ch.timestamps, Rotation.from_quat(ch.values))(t).as_quat()
        elif ch.kind == "zoh":
            j = np.clip(np.searchsorted(ch.timestamps, t, side="right") - 1, 0, len(ch.values) - 1)
            aligned[ch.name] = ch.values[j]
        else:
            aligned[ch.name] = np.stack(
                [np.interp(t, ch.timestamps, ch.values[:, c]) for c in range(ch.values.shape[1])],
                axis=-1,
            )

    return aligned, valid


def estimate_clock_offset(
    ts_a: np.ndarray, sig_a: np.ndarray,
    ts_b: np.ndarray, sig_b: np.ndarray,
    search_s: float = 0.5, step_s: float = 0.001,
) -> float:
    """Estimate the constant offset between two clocks by cross-correlating a
    shared signal (e.g. a deliberate motion spike visible to both sensors).

    Returns the offset to ADD to ts_b to align it with ts_a.
    """
    grid = np.arange(ts_a[0], ts_a[-1], step_s)
    a = np.interp(grid, ts_a, sig_a)
    a = (a - a.mean()) / (a.std() + 1e-9)

    offsets = np.arange(-search_s, search_s, step_s)
    scores = []
    for off in offsets:
        b = np.interp(grid, ts_b + off, sig_b)
        b = (b - b.mean()) / (b.std() + 1e-9)
        scores.append(float((a * b).mean()))
    return float(offsets[int(np.argmax(scores))])
```

**The validator** (extend this through every later module):

```python
"""m2-protocols/validate.py — automated episode auditing."""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class AuditReport:
    episode: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def audit_episode(
    timestamps: np.ndarray,
    actions: np.ndarray,
    qpos: np.ndarray,
    joint_limits: np.ndarray,      # (n_joints, 2)
    fps: int,
    name: str = "episode",
) -> AuditReport:
    r = AuditReport(episode=name)

    dt = np.diff(timestamps)
    if np.any(dt <= 0):
        r.errors.append(f"non-monotonic timestamps at {np.flatnonzero(dt <= 0)[:5].tolist()}")
    gaps = np.flatnonzero(dt > 1.5 / fps)
    if gaps.size:
        r.errors.append(f"{gaps.size} frame gaps > 1.5x nominal period")

    # Robust spike detection (median absolute deviation, not std -- std is
    # itself corrupted by the spikes you are trying to find).
    da = np.abs(np.diff(actions, axis=0))
    mad = np.median(np.abs(da - np.median(da, axis=0)), axis=0) + 1e-8
    spikes = np.flatnonzero((da / (1.4826 * mad) > 8).any(axis=1))
    if spikes.size:
        r.warnings.append(f"{spikes.size} action spikes (>8 robust sigma)")

    oob = (qpos < joint_limits[:, 0]).any(axis=1) | (qpos > joint_limits[:, 1]).any(axis=1)
    if oob.any():
        r.errors.append(f"{int(oob.sum())} frames outside joint limits")

    jerk = np.diff(qpos, n=3, axis=0) * (fps ** 3)
    if np.abs(jerk).max() > 5e3:
        r.warnings.append(f"peak jerk {np.abs(jerk).max():.0f} rad/s^3 over threshold")

    return r
```

**Leverage week — the dataset inspector.** A FastAPI backend serving episode data, and an Angular frontend giving you:

- Episode list with per-episode quality flags from the validator, sortable and filterable
- Synchronised video playback across all cameras, scrubber-linked
- 3D trajectory view (reuse `frame-viz` from Module 1) that seeks with the video
- Per-frame action/state plots with the current frame marked
- Visual markers on the timeline where the validator flagged problems
- Keyboard-driven triage: mark episode good/bad/needs-review, next/previous

```python
"""m2-protocols/server/main.py — minimal read API for the inspector."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Robot Dataset Inspector API")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:4200"], allow_methods=["*"])


class EpisodeSummary(BaseModel):
    id: str
    task: str
    num_frames: int
    fps: int
    cameras: list[str]
    quality: str            # 'pass' | 'warn' | 'fail'
    issues: list[str]


class FrameWindow(BaseModel):
    """Actions/state for a frame range — the plots and 3D view consume this."""
    start: int
    timestamps: list[float]
    ee_position: list[list[float]]
    ee_quaternion: list[list[float]]
    actions: list[list[float]]
    flags: list[str]


@app.get("/episodes", response_model=list[EpisodeSummary])
def list_episodes() -> list[EpisodeSummary]:
    return STORE.summaries()


@app.get("/episodes/{episode_id}/window", response_model=FrameWindow)
def get_window(episode_id: str, start: int = 0, count: int = 500) -> FrameWindow:
    if episode_id not in STORE:
        raise HTTPException(404, "unknown episode")
    return STORE.window(episode_id, start, count)
```

```typescript
// m2-protocols/inspector/src/app/episode.service.ts
import { HttpClient } from '@angular/common/http';
import { Injectable, computed, signal } from '@angular/core';
import { toObservable } from '@angular/core/rxjs-interop';
import { switchMap, shareReplay } from 'rxjs/operators';

export interface EpisodeSummary {
  id: string; task: string; numFrames: number; fps: number;
  cameras: string[]; quality: 'pass' | 'warn' | 'fail'; issues: string[];
}

@Injectable({ providedIn: 'root' })
export class EpisodeService {
  private http = new HttpClient(null as never); // inject() in real code

  readonly selectedId = signal<string | null>(null);
  readonly currentFrame = signal(0);

  /** Windowed fetch keyed on selection — the classic switchMap cancellation pattern. */
  readonly window$ = toObservable(this.selectedId).pipe(
    switchMap((id) =>
      this.http.get<FrameWindow>(`/api/episodes/${id}/window`, {
        params: { start: 0, count: 500 },
      }),
    ),
    shareReplay({ bufferSize: 1, refCount: true }),
  );

  /** Derived: is the current frame flagged by the validator? */
  readonly currentFlags = computed(() => this.windowSignal()?.flags[this.currentFrame()] ?? '');
}
```

### Deliverables

1. **`traj-sync`** — the timestamp-based aligner above, with automatic clock-offset estimation, emitting schema-conformant HDF5 or MCAP. Include a written comparison of *why* a latest-value join is wrong here; this is a strong technical-blog post and a strong interview answer.
2. **`traj-audit`** — the validator, extended to scan a directory and emit a dataset-level report.
3. **`oxe2lerobot`** — an ETL converting one Open X-Embodiment source dataset into LeRobot format, with every assumption documented.
4. **`dataset-inspector`** — the web tool. **This is your headline artefact for the first half of the curriculum.** Record a demo video. Compare it honestly to Foxglove and Rerun in the README — position it as a *dataset curation and triage* layer, not a general-purpose visualiser, because that framing is both true and more interesting.

### Self-check

- Why is `combineLatest` the wrong join for offline dataset construction?
- Why does LeRobot store video rather than raw frames, and what does that cost at training time?
- You're merging two datasets, one in millimetres with absolute poses, one in metres with delta poses. List every step to unify them.
- What is action chunking, and how does it constrain your storage layout?
- Name three dataset defects a schema validator catches, and three it cannot.

---

## Module 3 — Machine Learning Foundations in PyTorch

> **Month 3 · Objective:** Go from zero to being able to write, train, and *debug* a neural network. This module does not exist in the backend track; you need it.

Do not skip ahead to robot policies. A month spent here makes Module 4 straightforward; skipping it makes Module 4 impossible.

### Week-by-week

| Week | Focus | Output |
|---|---|---|
| 1 | Tensors, autograd, the training loop, optimisers, losses | MLP on a toy regression task, written from scratch |
| 2 | CNNs, image encoders, transfer learning, ResNet as a feature extractor | Image classifier fine-tuned from a pretrained backbone |
| 3 | Sequence models: attention and transformers from first principles | Minimal transformer implemented and trained |
| 4 | The craft: overfitting a single batch, LR schedules, normalisation, experiment tracking, reproducibility | Debugging playbook + W&B-tracked runs |

### Key concepts & theory

**The training loop is the whole thing.** Five steps, repeated: forward pass → compute loss → zero gradients → backward pass → optimiser step. Everything else is elaboration. Write it by hand until it's muscle memory; do not start from a framework that hides it.

**Autograd as a concept you already know.** A computation graph that records operations and replays them backwards to compute derivatives. If you've ever reasoned about a reactive dependency graph recomputing downstream values, the shape is familiar — the difference is the traversal goes *backwards* and accumulates derivatives via the chain rule.

**Tensors vs NumPy arrays.** Nearly the same API, plus: device placement (`.to('cuda')`), gradient tracking (`requires_grad`), and batching conventions. The persistent beginner bug is shape mismatch; the persistent intermediate bug is a tensor accidentally left on the wrong device or still carrying a gradient into a place it shouldn't.

**Image encoders.** A CNN maps an image to a feature vector. For robot policies you'll usually use a pretrained ResNet-18 or a small ViT as a frozen or lightly-finetuned encoder. Understand: receptive field, why spatial-softmax layers are common in visuomotor policies (they output *coordinates* rather than pooled features, which is what a manipulation policy actually needs), and why ImageNet-pretrained features help even for robot data.

**Attention and transformers.** Enough to read the ACT and diffusion-policy papers: queries/keys/values, multi-head attention, positional encodings, encoder vs decoder, causal masking. Implement a tiny one — reading about attention teaches much less than writing it.

**The debugging craft — the part nobody writes down**

This is where engineers with strong software discipline outperform, so lean in:

1. **Overfit a single batch first.** If your model cannot drive loss to ~0 on 8 samples, you have a bug, not a data problem. This single habit saves weeks.
2. **Check shapes at every boundary.** Assert them. Treat shape as a type.
3. **Verify the data pipeline independently** — visualise what actually reaches the model after augmentation. Wrong normalisation and accidentally shuffled labels are extremely common and invisible in the loss curve.
4. **Seed everything, log everything.** Config, git SHA, seed, data version. An unreproducible experiment is not an experiment.
5. **Learning rate is the most important hyperparameter** by a wide margin. Sweep it first, an order of magnitude at a time.
6. **Watch for silent NaN.** Once loss goes NaN, everything downstream is garbage; assert on it.

### Applied engineering & code lab

```python
"""m3-ml/training_loop.py — the loop, unhidden, with the debugging habits built in."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def overfit_single_batch(model: nn.Module, batch, steps: int = 300, lr: float = 1e-3) -> float:
    """Sanity check #1: can the model memorise 8 examples?

    If final loss is not near zero, stop and find the bug. Do not proceed to
    a full training run. This check has a higher return on time than any
    other single habit in ML engineering.
    """
    x, y = batch
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for step in range(steps):
        pred = model(x)
        assert pred.shape == y.shape, f"shape mismatch: {pred.shape} vs {y.shape}"
        loss = loss_fn(pred, y)
        assert torch.isfinite(loss), f"non-finite loss at step {step}"
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return float(loss)


def train_epoch(model, loader: DataLoader, opt, loss_fn, device, clip: float = 1.0) -> float:
    model.train()
    total, seen = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        pred = model(x)                       # 1. forward
        loss = loss_fn(pred, y)               # 2. loss
        opt.zero_grad(set_to_none=True)       # 3. clear stale gradients
        loss.backward()                       # 4. backward
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()                            # 5. update

        total += float(loss) * x.size(0)
        seen += x.size(0)
    return total / seen


@torch.no_grad()
def evaluate(model, loader, loss_fn, device) -> float:
    model.eval()                              # disables dropout / uses running BN stats
    total, seen = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        total += float(loss_fn(model(x), y)) * x.size(0)
        seen += x.size(0)
    return total / seen
```

```python
"""m3-ml/repro.py — reproducibility scaffolding. Set this up once, use it forever."""
import os
import random
import subprocess

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    # trades speed for determinism


def run_metadata(config: dict) -> dict:
    """Attach to every experiment log. An experiment you cannot rerun is an anecdote."""
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"],
                                capture_output=True, text=True).stdout.strip())
    return {"git_sha": sha, "git_dirty": dirty, "config": config,
            "torch": torch.__version__, "cuda": torch.version.cuda}
```

### Deliverables

1. **A from-scratch MLP** trained with a hand-written training loop (no Lightning, no trainer abstraction), including the single-batch overfit check.
2. **A fine-tuned image encoder** — ResNet-18 pretrained backbone adapted to a small dataset, with a comparison against training from scratch. Understand *why* the pretrained one wins with little data; you'll use this argument about robot data too.
3. **A minimal transformer** — implemented and trained on a toy sequence task, with attention maps visualised. (Visualising attention is another place your frontend skill produces a better artefact than the norm.)
4. **A debugging playbook** — write down, in your own words, the checklist you now use when a model won't train. This becomes genuinely useful reference material and demonstrates the kind of thinking that gets hired.

### Self-check

- Write the five steps of a training loop from memory and explain why `zero_grad` is needed.
- Your loss is flat from step 0. Give five candidate causes in order of likelihood.
- What does `model.eval()` change, and what happens if you forget it?
- Why does an ImageNet-pretrained encoder help on robot images that look nothing like ImageNet?
- What's the difference between a training loss of 0.001 and a *useful* model?

---

## Module 4 — Policy Learning & VLA Models

> **Month 4 · Objective:** Train real robot policies, and — more importantly — learn to reason about what data would make them better. This is the module that converts you from "tooling engineer" to "data researcher."

### Week-by-week

| Week | Focus | Output |
|---|---|---|
| 1 | Behavioural cloning; covariate shift; DAgger; action chunking | BC baseline on LeRobot PushT |
| 2 | ACT (Action Chunking Transformer); temporal ensembling | ACT trained, loss curves logged |
| 3 | Diffusion Policy; multimodal action distributions | Diffusion Policy trained; compared to BC |
| 4 | VLAs (RT-2, OpenVLA, π₀); **the data ablation experiment** | `RESULTS.md` with the ablation table |

### Key concepts & theory

**Behavioural cloning and its characteristic failure.** BC is supervised regression: `a_t = π(o_t)`. Trivial to implement, and it fails in a specific, learnable way.

**Covariate shift / compounding error:** the policy is trained on the *expert's* state distribution. Any small error moves it slightly off that distribution, where its predictions are worse, which moves it further off. Error compounds roughly as `O(εT²)` over horizon `T`. This is the central problem of imitation learning, and — critically for you — **every mitigation is a data decision:**

- **DAgger** — iteratively collect expert corrections on the states the policy actually visits.
- **Recovery data** — deliberately perturb the robot mid-task and record a human recovering. Small amounts have outsized effect.
- **Action chunking** — predict `H` steps at once; fewer decision points means less compounding.
- **Temporal ensembling** — average overlapping predicted chunks for smoothness.
- **Augmentation** — random crop, colour jitter, camera-pose jitter.

**Multimodality of human demonstrations.** If half your demonstrators go left around an obstacle and half go right, an L2-regression policy learns the *mean* — straight into the obstacle. This is the clearest motivation for diffusion policies, and it is fundamentally a data-curation observation: the same dataset that breaks one architecture is fine for another. Being able to say that fluently is what a data researcher sounds like.

**Architectures worth knowing by name**

| Model | Core idea | Why a data person cares |
|---|---|---|
| **BC-MLP / BC-RNN** | Regression baseline | Your control condition in every experiment |
| **ACT** | Transformer + CVAE predicting action chunks | Standard ALOHA-family baseline; cheap to train |
| **Diffusion Policy** | Denoising diffusion over action sequences | Handles multimodal demonstrations; strong default |
| **RT-1 / RT-2** | Discretised action tokens; RT-2 co-trains on web VQA | Showed web data improves *physical* generalisation |
| **OpenVLA** | 7B open VLA trained on OXE; LoRA-finetunable | Your realistic hands-on VLA |
| **π₀ / π₀.₅** | Flow-matching action expert on a VLM backbone | Current frontier reference |
| **World models** | Learn dynamics/prediction rather than actions | "What data teaches physics?" is a real interview question |

**Evaluation — where most candidates are weak and you can be strong**

- Validation loss is a poor proxy for task success. Report both, and say so explicitly.
- **Success rate** over N trials with a *stated* success criterion, plus a binomial confidence interval. `9/10` and `90/100` are not the same claim, and people constantly present them as if they were.
- Decompose generalisation into named axes: unseen object poses, unseen lighting, unseen object appearance, distractor objects, longer horizons. Each is a separate column.
- Smoothness metrics: jerk, path-length ratio, action saturation.
- **Failure taxonomy:** never-grasped / grasped-then-dropped / wrong-object / collided / timed-out. A benchmark reporting only a success percentage tells a researcher nothing about what data to collect next. A failure taxonomy tells them exactly. This is the single most valuable evaluation habit in this curriculum.

### Applied engineering & code lab

```python
"""m4-policy/chunked_bc.py — chunked behavioural cloning."""
import torch
import torch.nn as nn


class ChunkedBCPolicy(nn.Module):
    """Visual features + proprioception -> a chunk of H future actions.

    Chunking is the cheapest meaningful improvement over vanilla BC: fewer
    decision points means less compounding error.
    """

    def __init__(self, visual_dim: int, state_dim: int, action_dim: int, horizon: int):
        super().__init__()
        self.horizon, self.action_dim = horizon, action_dim
        self.net = nn.Sequential(
            nn.Linear(visual_dim + state_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, action_dim * horizon),
        )

    def forward(self, visual: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        x = torch.cat([visual, state], dim=-1)
        return self.net(x).view(x.shape[0], self.horizon, self.action_dim)


@torch.no_grad()
def temporal_ensemble(chunk_buffer: list[torch.Tensor], k: float = 0.01) -> torch.Tensor:
    """ACT-style exponentially-weighted average of overlapping chunk predictions.

    chunk_buffer[i] is the prediction for *this* timestep made i steps ago.
    """
    w = torch.exp(-k * torch.arange(len(chunk_buffer), dtype=torch.float32))
    w = w / w.sum()
    return (torch.stack(chunk_buffer) * w[:, None]).sum(0)
```

```python
"""m4-policy/metrics.py — evaluation that actually informs data collection."""
from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.stats import beta


@dataclass
class EvalResult:
    axis: str
    trials: int
    successes: int
    failures: Counter

    @property
    def rate(self) -> float:
        return self.successes / self.trials if self.trials else float("nan")

    def confidence_interval(self, alpha: float = 0.05) -> tuple[float, float]:
        """Clopper-Pearson exact interval. 9/10 and 90/100 are NOT the same claim."""
        k, n = self.successes, self.trials
        lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
        hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
        return float(lo), float(hi)

    def to_row(self) -> str:
        lo, hi = self.confidence_interval()
        top = ", ".join(f"{m}:{c}" for m, c in self.failures.most_common(3))
        return f"| {self.axis} | {self.successes}/{self.trials} | {self.rate:.0%} | [{lo:.0%}, {hi:.0%}] | {top} |"


def smoothness(positions: np.ndarray, fps: int) -> dict:
    """Trajectory quality independent of task success."""
    jerk = np.diff(positions, n=3, axis=0) * (fps ** 3)
    path = np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()
    straight = np.linalg.norm(positions[-1] - positions[0])
    return {
        "peak_jerk": float(np.abs(jerk).max()),
        "mean_jerk": float(np.abs(jerk).mean()),
        "path_efficiency": float(straight / path) if path > 0 else 0.0,
    }
```

**The experiment that matters — run this in Week 4.**

Train the *same* policy on systematically varied data:

| Condition | Episodes | Question answered |
|---|---|---|
| Full dataset | 200 | Baseline |
| Half | 100 | Where is the data-scaling knee? |
| Quarter | 50 | |
| Full minus the worst 10% by your Module 2 validator | 180 | **Does curation beat volume?** |
| Full + augmentation | 200 | Does augmentation substitute for data? |
| Full, with 50 ms of injected sensor misalignment | 200 | **What does bad synchronisation actually cost?** |

That last row is *your* experiment. Nobody else running this curriculum will have built a synchroniser good enough to deliberately break it in a controlled way. "I quantified the success-rate cost of unmodelled sensor latency in training data" is a sentence that gets you an interview.

### Deliverables

1. **Trained ACT and/or Diffusion Policy** on a LeRobot benchmark, with logged loss curves and evaluated success rates.
2. **`policy-viz`** — extend your Module 1/2 viewer to overlay predicted action chunks on ground-truth trajectories over the camera feed. Frame this as a *data debugging tool*: seeing where the policy diverges tells you where the data is thin.
3. **`RESULTS.md`** — the ablation table above with confidence intervals, failure taxonomy, plots, and an honest limitations section.

### Self-check

- Explain covariate shift to an engineer with no ML background, in three sentences.
- Why does an L2 policy fail on multimodal demonstrations, and how does diffusion fix it?
- You have budget for 100 more teleop episodes. How do you choose which 100?
- What's wrong with reporting "90% success rate" and nothing else?

---

## Module 5 — Physics Simulation & Synthetic Data

> **Month 5 · Objective:** Generate domain-randomised synthetic datasets at scale, reproducibly.

### Week-by-week

| Week | Focus | Output |
|---|---|---|
| 1 | MuJoCo & MJCF; loading a robot; scripted control | Tabletop scene with a 7-DoF arm |
| 2 | Scripted pick-and-place with IK; offscreen rendering | Working episode generator |
| 3 | Domain randomisation: visual and dynamics; config schema | `randomization_rules.yaml` + sampler |
| 4 | Scale-out: parallel generation, export, generation QA | 5k-episode synthetic dataset |

### Key concepts & theory

**Simulators.** **MuJoCo** (Apache-2.0, DeepMind) is your engine: fast, accurate soft-contact solver, CPU-first, trivial install, excellent Python bindings. **MuJoCo MJX** gives JAX/GPU-parallel rollouts. **NVIDIA Isaac Sim / Isaac Lab** offers photorealistic RTX rendering and GPU physics — learn the vocabulary, but see Appendix A: it will not run on your GPU. Also know the names PyBullet, Genesis, SAPIEN, and Drake.

**Description formats.** **URDF** is the ROS standard (links, joints, inertials, visual/collision geometry; no closed chains, weak contact parameters). **MJCF** is MuJoCo's XML with richer contact and actuator parameters. **USD** is Isaac Sim's composable scene graph. URDF → MJCF conversion is routine but lossy in exactly the parameters that matter for contact-rich manipulation.

**Domain randomisation.**

- *Visual:* lighting position/colour/intensity, textures, background clutter, distractors, camera intrinsics and extrinsics, sensor noise, motion blur, exposure.
- *Dynamics:* mass, inertia, friction (sliding/torsional/rolling), joint damping, actuator gains, control latency, observation noise.
- **The key insight:** DR works by making reality look like one more sample from the training distribution. Too narrow → no transfer. Too wide → the policy learns an over-conservative average and underperforms. The width of the randomisation distribution is a hyperparameter to tune and report, not a setting to guess once.

**The sim-to-real gap, specifically:** contact and friction modelling; rendering realism; unmodelled actuator dynamics and latency; deformables and liquids; and real depth-sensor noise characteristics that no simulator reproduces by default.

**Reproducibility of generated data — your software-engineering edge.** Every episode records its seed and full randomisation parameter vector. The generation config is versioned and hashed into the dataset metadata. Regeneration from a seed is bit-identical, or you cannot debug the pipeline.

### Applied engineering & code lab

```python
"""m5-sim/generator.py — reproducible domain-randomised generation."""
from dataclasses import asdict, dataclass

import mujoco
import numpy as np


@dataclass
class RandomizationSpec:
    """Serialised into every episode's metadata."""
    friction_range: tuple = (0.5, 1.5)
    mass_scale_range: tuple = (0.8, 1.2)
    light_pos_jitter: float = 0.3
    camera_pos_jitter: float = 0.02
    camera_euler_jitter_deg: float = 2.0


class MujocoDataGenerator:
    def __init__(self, xml_path: str, spec: RandomizationSpec, width=640, height=480):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.spec = spec

    def randomize(self, seed: int) -> dict:
        """Apply DR from an explicit seed; return sampled parameters as metadata."""
        rng = np.random.default_rng(seed)
        params: dict = {"seed": int(seed), **asdict(self.spec)}

        friction = rng.uniform(*self.spec.friction_range, size=self.model.ngeom)
        self.model.geom_friction[:, 0] = friction
        params["friction"] = friction.tolist()

        scale = rng.uniform(*self.spec.mass_scale_range, size=self.model.nbody)
        self.model.body_mass[:] = self.model.body_mass * scale
        params["mass_scale"] = scale.tolist()

        self.model.geom_rgba[:, :3] = rng.uniform(0.1, 0.9, size=(self.model.ngeom, 3))
        if self.model.nlight:
            self.model.light_pos[:] += rng.normal(
                0, self.spec.light_pos_jitter, size=self.model.light_pos.shape
            )
        return params

    def rollout(self, controller, max_steps: int, seed: int) -> dict:
        """Generate one episode. `controller(data) -> ctrl vector`."""
        mujoco.mj_resetData(self.model, self.data)
        params = self.randomize(seed)
        frames, qpos, qvel, actions, times = [], [], [], [], []
        for _ in range(max_steps):
            ctrl = controller(self.data)
            self.data.ctrl[:] = ctrl
            mujoco.mj_step(self.model, self.data)
            self.renderer.update_scene(self.data, camera="hand_eye_cam")
            frames.append(self.renderer.render())
            qpos.append(self.data.qpos.copy())
            qvel.append(self.data.qvel.copy())
            actions.append(np.asarray(ctrl).copy())
            times.append(float(self.data.time))
        return {
            "images": np.stack(frames), "qpos": np.stack(qpos), "qvel": np.stack(qvel),
            "actions": np.stack(actions), "timestamp": np.asarray(times),
            "randomization": params,
        }
```

### Deliverables

1. **Headless MuJoCo pick-and-place environment** with a scripted IK controller that reliably completes the task.
2. **`synth-gen`** — parallel generator producing 5,000 episodes with per-episode DR, exporting in your Module 1 schema, with a verified seed → bit-identical-dataset guarantee.
3. **A DR ablation** — train the Module 4 policy on narrow DR, wide DR, and no DR; evaluate all three on a held-out simulated "test world" with parameters *outside* every training range. This is a legitimate stand-in for sim-to-real without owning a robot.
4. **Extend the inspector** to render synthetic episodes side by side with their randomisation parameters. Being able to visually scan 5,000 generated episodes for degenerate ones is a real capability.

### Self-check

- Why can too much domain randomisation hurt?
- What is lost converting URDF → MJCF for a contact-rich task?
- Sim success 95%, real success 20%. Five candidate causes, ranked.
- How do you make a 5,000-episode generation run bit-reproducible?

---

## Module 6 — 3D Vision & Perception

> **Month 6 · Objective:** Turn raw RGB-D into calibrated 3D, and build the quality filters that decide whether a frame is usable — with a viewer, because you can.

### Week-by-week

| Week | Focus | Output |
|---|---|---|
| 1 | Pinhole model, intrinsics, distortion, calibration | Calibration + undistortion tool |
| 2 | Depth → point cloud; extrinsics; hand-eye calibration | Back-projection implemented from scratch |
| 3 | Point cloud processing; ICP registration; multi-view merge | Two-view merged cloud |
| 4 | **Leverage week:** perception QA + browser point-cloud viewer | `depth-qa` + WebGL cloud viewer |

### Key concepts & theory

**Camera geometry**

- **Intrinsics** `K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]` map camera-frame 3D points to pixels.
- **Extrinsics** `[R | t]` give the rigid transform from world/base to camera. For a wrist camera this is `T_ee_cam`, found by **hand-eye calibration** (solving `AX = XB`).
- **Distortion:** radial `(k1, k2, k3)`, tangential `(p1, p2)`. Undistort before any geometric reasoning.
- **Back-projection:** `X = (u − cx)·d / fx`, `Y = (v − cy)·d / fy`, `Z = d`.
- **Depth units are a perennial bug source:** 16-bit millimetres vs float metres, depth-to-colour registration, driver `depth_scale`. Assert, never assume.
- **Depth modalities and their signatures:** structured light (fails in sunlight), active stereo (needs texture), ToF (multipath artefacts at corners), LiDAR (sparse). Each fails distinctively, and each failure is programmatically detectable — which is the point for a data-quality role.

**3D representations.** Point clouds (`(N, 3)` + colour/normals; PointNet → PointNet++ → PointNeXt); voxel grids and TSDFs; meshes; NeRF; and **3D Gaussian Splatting**, now used to build simulation assets and augmented viewpoints from real scans. Registration: ICP (point-to-point, point-to-plane) refined from a global FPFH+RANSAC initialisation — ICP is a *local* method and needs a decent starting guess.

**Perception data quality checks** — depth validity ratio per frame; flying pixels at discontinuities; specular dropout on metal; **occlusion of the target object by the robot's own arm** (the most common silent killer in wrist-camera data); motion blur via variance-of-Laplacian; exposure clipping; point-cloud sparsity in the region of interest.

### Applied engineering & code lab

```python
"""m6-perception/pointcloud.py — RGB-D to point cloud, and the quality filters."""
import numpy as np
import open3d as o3d


def backproject(depth_m: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Depth image to an (H, W, 3) point map, vectorized. No loops.

    This is the Month 0 broadcasting lesson applied to a real problem.
    """
    h, w = depth_m.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    x = (u - cx) * depth_m / fx
    y = (v - cy) * depth_m / fy
    return np.stack([x, y, depth_m], axis=-1)


def rgbd_to_point_cloud(color: np.ndarray, depth_m: np.ndarray, K: np.ndarray,
                        depth_trunc: float = 3.0) -> o3d.geometry.PointCloud:
    h, w = depth_m.shape
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(color, dtype=np.uint8)),
        o3d.geometry.Image(np.ascontiguousarray(depth_m, dtype=np.float32)),
        depth_scale=1.0, depth_trunc=depth_trunc, convert_rgb_to_intensity=False,
    )
    intr = o3d.camera.PinholeCameraIntrinsic(w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)


def register_views(source, target, voxel: float = 0.005, init=None) -> np.ndarray:
    """Point-to-plane ICP refinement, returning T_target_source."""
    src, tgt = source.voxel_down_sample(voxel), target.voxel_down_sample(voxel)
    for pc in (src, tgt):
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 4, max_nn=30))
    result = o3d.pipelines.registration.registration_icp(
        src, tgt, voxel * 3, np.eye(4) if init is None else init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    return result.transformation


def depth_quality(depth_m: np.ndarray, roi=None) -> dict:
    d = depth_m if roi is None else depth_m[roi[0]:roi[1], roi[2]:roi[3]]
    valid = np.isfinite(d) & (d > 0)
    gy, gx = np.gradient(np.where(valid, d, np.nan))
    grad = np.hypot(np.nan_to_num(gy), np.nan_to_num(gx))
    return {
        "valid_ratio": float(valid.mean()),
        "median_depth_m": float(np.median(d[valid])) if valid.any() else float("nan"),
        "flying_pixel_ratio": float((grad > 0.05).mean()),
        "dropout_ratio": float((~valid).mean()),
    }


def is_target_occluded(depth_m: np.ndarray, target_mask: np.ndarray,
                       expected_depth_m: float, tol: float = 0.03) -> bool:
    """True when something (usually the arm) sits in front of the target region."""
    obs = depth_m[target_mask]
    obs = obs[np.isfinite(obs) & (obs > 0)]
    return True if obs.size == 0 else bool(np.median(obs) < expected_depth_m - tol)
```

**Leverage week — the browser point-cloud viewer.** Stream a decimated point cloud to the frontend and render it with Three.js `Points`. Being able to open a browser tab and scrub through a merged 3D reconstruction of an episode, with quality flags overlaid, is a tool researchers will actually ask you for.

```typescript
// m6-perception/viewer/src/app/point-cloud.component.ts (core rendering logic)
private buildCloud(positions: Float32Array, colors: Float32Array): THREE.Points {
  const geometry = new THREE.BufferGeometry();
  // Interleaved XYZ and RGB, uploaded once — decimate server-side, not here.
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geometry.computeBoundingSphere();

  return new THREE.Points(
    geometry,
    new THREE.PointsMaterial({ size: 0.003, vertexColors: true, sizeAttenuation: true }),
  );
}
```

### Deliverables

1. **`rgbd-fuse`** — merge two RGB-D streams (wrist + fixed camera) into one point cloud in the robot base frame, via global registration + ICP, with a before/after visual.
2. **`depth-qa`** — the per-frame quality filter including arm-occlusion detection, wired into the Module 2 validator.
3. **A browser point-cloud viewer** integrated into the inspector.
4. **A calibration write-up** quantifying residual error and how it propagates into 3D ground-truth annotations. Quantifying annotation error is a research-grade habit that very few candidates demonstrate.

### Self-check

- Given `K` and a depth pixel, write the back-projection by hand.
- Why does ICP need a good initial guess, and what do you do without one?
- Your merged cloud shows the object duplicated in two nearby positions. What's wrong?
- Four distinct physical causes of missing depth pixels, and how you'd detect each.

---

## Module 7 — Capstone: Robot Data Studio

> **Month 7–8 · Objective:** Ship one public system that no robotics PhD and no backend engineer would have built — and that a robotics data lab would genuinely want to use.

### The artefact

**`robot-data-studio`** — *An inspection, validation and curation platform for multimodal robot datasets.*

This is deliberately different from the backend track's capstone. That one is a pipeline. This one is a **pipeline plus the instrument**, and the instrument is where you are irreplaceable.

```
robot-data-studio/
│
├── packages/
│   ├── core/                          # Python: the pipeline and the truth
│   │   ├── schema.py                  # Pydantic models -- single source of truth
│   │   ├── sync.py                    # Timestamp-based alignment (Module 2)
│   │   ├── validator.py               # All quality checks (Modules 2, 5, 6)
│   │   ├── generator.py               # MuJoCo synthetic generation (Module 5)
│   │   ├── converter.py               # LeRobot / MCAP / Zarr export (Module 2)
│   │   └── stats.py                   # Normalisation stats, dataset cards
│   │
│   ├── api/                           # FastAPI: serves episodes, frames, flags
│   │   ├── main.py
│   │   └── openapi.json               # -> generates the TS client
│   │
│   └── studio/                        # Angular: the instrument
│       ├── episode-browser/           # Sortable, filterable, quality-flagged list
│       ├── sync-player/               # Multi-camera synchronised playback
│       ├── trajectory-3d/             # Three.js pose + path + point cloud
│       ├── signal-plots/              # Action/state charts, frame-linked
│       ├── quality-timeline/          # Validator flags rendered on the scrubber
│       ├── triage/                    # Keyboard-driven good/bad/review labelling
│       └── report/                    # Benchmark results viewer
│
├── models/                            # Policy training (Module 4)
├── evaluation/                        # Benchmark runner + failure taxonomy
├── tests/                             # Property tests + golden-file regressions
├── .github/workflows/                 # CI: validate a fixture dataset on every push
├── DATASET_CARD.md
├── RESULTS.md
└── README.md
```

### Requirements

**1. Schema as contract, end to end.** Pydantic models in `core/schema.py` are the single source of truth. Generate the OpenAPI spec from FastAPI, generate the TypeScript client from the OpenAPI spec, and have CI fail if the generated client drifts. **This is a genuinely impressive thing to show a robotics team**, because almost no research codebase has a typed contract from the data schema all the way to the UI, and everyone who has suffered without one recognises the value immediately.

**2. The pipeline.** Ingest or generate multi-episode manipulation data with 2× RGB-D streams, joint telemetry, 6-DoF end-effector actions, gripper state, and task labels — synchronised onto a canonical clock, schema-validated, exportable to LeRobot.

**3. Quality infrastructure, CI-enforced.** Frame continuity; kinematic bounds; jerk thresholds; cross-modal alignment; depth usability and occlusion; schema conformance. All running in GitHub Actions against a small fixture dataset committed to the repo, including a **deliberately corrupted episode that CI must reject** — a test that demonstrates the validator works is more convincing than the validator itself.

**4. The studio.** Everything listed above, with two features that turn it from a viewer into a *curation* tool:
   - **Triage workflow:** keyboard-driven episode review with a persisted verdict, producing a curated subset manifest.
   - **Quality timeline:** validator flags rendered directly on the video scrubber, so a human reviewing 500 episodes can jump straight to the suspicious frames instead of watching everything.

**5. The experiment.** Use the studio to do something measurable. The strongest option, because it closes the loop between your tooling and a research result:

> **"Human-in-the-loop curation beats both raw volume and automated filtering alone."**
>
> Train the same policy on: (a) all N episodes, (b) N episodes minus the worst 10% by automated validator score, (c) N episodes minus 10% removed by *human triage through the studio*, and (d) 0.8N episodes chosen at random. Report success rates with confidence intervals across your generalisation axes.
>
> Then report the thing only you can report: **how long the human triage took**. "Curated 500 episodes in 40 minutes using the studio, versus an estimated 6 hours with matplotlib and ffmpeg" is a product argument *and* a research argument, and it is exactly the kind of contribution a data lab is built to value.

**6. Publication.**

- GitHub repo, permissive licence, quickstart that a stranger runs in under 15 minutes (`docker compose up` ideally).
- **Live demo deployed** — GitHub Pages frontend against a small hosted sample dataset. You can do this and most candidates cannot. A recruiter clicking a link and immediately seeing a working tool is worth more than any README.
- Dataset on Hugging Face in LeRobot format with a complete dataset card.
- A technical blog post. Suggested angle: *"What frontend engineering has to offer robotics data"* — the RxJS/time-join insight, the schema-to-UI type contract, the curation-speed result. That post is a differentiated point of view, not a tutorial rehash, and differentiated points of view get shared.
- A 3–5 minute demo video.

### Mapping back to the job description

| Job requirement | Where the studio demonstrates it |
|---|---|
| "Design new datasets and collection methods" | `generator.py`, randomisation config |
| "Define data schemas, annotations, ground truth" | `schema.py`, `DATASET_CARD.md`, the triage/annotation workflow |
| "…and evaluations" | `evaluation/`, generalisation axes, failure taxonomy |
| "Run experiments to understand what data most improves model performance" | `RESULTS.md` — the curation experiment |
| "Turn research needs into scalable data pipelines and **products**" | The studio is literally a product |
| "Strong technical communication and cross-functional collaboration" | Live demo, dataset card, blog post, video |

Note that "products" is in the actual posting. For a company whose business is a *platform* for expert human data contribution, a candidate who ships the instrument is not a lesser fit — they're a different and arguably scarcer one.

---

## Timeline & Milestones

| Month | Module | Core output | Done when |
|---|---|---|---|
| **0** | Python/NumPy/LinAlg bridge | Ported library, vectorisation notebook, Linux+GPU env | You solve NumPy problems without reaching for a loop |
| **1** | Spatial math & kinematics | `spatial-py`, `frame-graph`, **`frame-viz` (public)** | Frame viewer live on GitHub Pages |
| **2** | Data protocols & inspector | `traj-sync`, `traj-audit`, `oxe2lerobot`, **`dataset-inspector`** | Inspector loads a real LeRobot dataset |
| **3** | ML foundations | MLP from scratch, fine-tuned encoder, mini transformer, debugging playbook | You can debug a model that won't train |
| **4** | Policy learning | Trained ACT/Diffusion Policy, `policy-viz`, ablation table | `RESULTS.md` with CIs published |
| **5** | Simulation & synthetic data | `synth-gen` 5k episodes, DR ablation | Seed → bit-identical dataset verified |
| **6** | 3D perception | `rgbd-fuse`, `depth-qa`, browser cloud viewer | Two views merge with sub-cm residual |
| **7–8** | Capstone | **`robot-data-studio` published + deployed + written up** | A stranger runs it in 15 minutes |

**Weekly cadence**

- Mon/Tue/Thu evenings (2h): theory and reading — one paper per week, notes committed to the repo.
- Wed evening (2h): code lab.
- Sat morning (4–5h): the week's deliverable.
- Sun (1h): write. Commit a progress note. **Public writing every week is non-negotiable** — it compounds into the portfolio and forces clarity you won't reach otherwise.

**Monthly review questions**

1. Did I ship the artefact, or do I have a folder of half-finished notebooks?
2. Can I explain this month's core concept to an engineer with no robotics background?
3. What did I learn that changes how I'd design a dataset?

**Start applying in Month 6, not Month 8.** By Month 6 you have a public dataset inspector, a trained policy with a real ablation result, and a synthetic data pipeline. That's already a stronger portfolio than most applicants to robotics *tooling* roles. Interview processes take 2–3 months; run them in parallel with the capstone and let the interviews tell you what to sharpen.

---

## Appendix A — Hardware & Environment Notes

**Your machine: NVIDIA Quadro P3000 (Pascal, 6 GB).** Plan around this now rather than discovering it in Month 5:

| Task | Feasible locally? | Plan |
|---|---|---|
| Month 0, Modules 1, 2, 6 (Python, math, data formats, Open3D) | ✅ Yes, CPU-bound | No changes needed |
| All frontend work | ✅ Yes | Your home turf |
| MuJoCo simulation + offscreen rendering | ✅ Yes | CPU-first; EGL offscreen works on Pascal |
| Training BC / ACT / small Diffusion Policy | ⚠️ Tight | Small images (96–128 px), batch 8–16, gradient accumulation. Hours, not minutes |
| **NVIDIA Isaac Sim / Isaac Lab** | ❌ No | Needs RTX ray-tracing cores. Learn the concepts; use MuJoCo for all hands-on work |
| Fine-tuning OpenVLA (7B) | ❌ No | Rent an A100/H100 (Lambda, RunPod, Vast.ai, ~$1–3/hr). Budget ~$150–250 across the whole plan |
| 3D Gaussian Splatting | ⚠️ Small scenes only | 6 GB caps scene size; fine for a tabletop |

**Setup**

- Python 3.11 with `uv`; pin everything and commit the lockfile.
- Install order that avoids pain: `numpy`, `scipy`, then `torch` (CUDA 12.x build), then `mujoco`, then `open3d`, then `lerobot`.
- **Use WSL2 for the Python/robotics half.** Much of the ecosystem (ROS 2, MCAP tooling, EGL rendering, LeRobot's video backends) assumes Linux. Keep your Angular/Node work wherever you're already fastest — a split setup is fine and normal.
- MuJoCo headless rendering needs `MUJOCO_GL=egl` on Linux.
- Track experiments from day one — W&B free tier, or disciplined CSV + matplotlib. Untracked experiments are unpublishable experiments.
- **Rented-GPU discipline:** develop and debug at tiny scale locally, then run the full job from a committed config. Never debug on a paid GPU.

---

## Appendix B — TypeScript → Python Phrasebook

| TypeScript / Angular | Python | Notes |
|---|---|---|
| `interface User { id: string }` | `@dataclass class User: id: str` | Or a Pydantic `BaseModel` for runtime validation |
| `type X = A \| B` | `X = A \| B` (3.10+) | Same syntax, unenforced at runtime |
| `string \| null` | `str \| None` / `Optional[str]` | One nullish value, not two |
| `arr.map(f)` | `[f(x) for x in arr]` | For numeric arrays use NumPy vectorisation instead |
| `arr.filter(p)` | `[x for x in arr if p(x)]` | |
| `arr.reduce(f, init)` | `functools.reduce(f, arr, init)` | Usually a plain loop reads better |
| `Object.entries(o)` | `o.items()` | |
| `{...a, ...b}` | `{**a, **b}` | |
| `[...a, ...b]` | `[*a, *b]` | |
| `async/await` | `async/await` | `asyncio` instead of the microtask queue |
| `Promise.all` | `asyncio.gather` | |
| `npm` / `package.json` | `uv` / `pyproject.toml` | |
| `node_modules` | virtualenv (`.venv`) | Per-project, must be activated |
| `jest` | `pytest` | Fixtures ≈ `beforeEach` with DI |
| `eslint` + `prettier` | `ruff` (both jobs) | |
| `tsc --noEmit` | `mypy` / `pyright` | Opt-in, but use it |
| DI via `@Injectable` | Explicit construction, or a small container | Python DI frameworks exist; usually unnecessary |
| `RxJS Observable` | Generators, `asyncio` streams, or callbacks | For *offline* data use arrays; reserve streaming for live |
| `readonly` | Naming convention + `@dataclass(frozen=True)` | Weakly enforced |
| `private` | `_leading_underscore` | Convention only |
| `enum` | `enum.Enum` / `StrEnum` | |
| `never`/exhaustiveness | `assert_never` from `typing_extensions` | |

**Gotchas with no TypeScript analogue**

- **Mutable default arguments.** `def f(x=[])` creates *one* list shared by every call. Always `field(default_factory=list)` or `x=None` then `x = x or []`.
- **Late-binding closures in loops.** `[lambda: i for i in range(3)]` — all three return 2. Use `lambda i=i: i`.
- **Truthiness of arrays.** `if my_numpy_array:` raises. Use `.any()` / `.size` / `is not None`. This will bite you in week one of Month 0.
- **Integer division.** `/` is float division, `//` is floor. `7 / 2 == 3.5`.
- **`is` vs `==`.** `is` is identity, `==` is equality. Use `is` only for `None`, `True`, `False`.
- **Views vs copies in NumPy.** Slicing returns a *view*; mutating it mutates the original. `arr[0:5] = 0` changes `arr`. Use `.copy()` when you mean a copy.

---

## Appendix C — Reading List

One paper a week; read the method and the *data* section, skim the rest.

**Month 0 foundations**
- 3Blue1Brown, *Essence of Linear Algebra* (watch, then implement)
- *Python for Data Analysis* (McKinney) — NumPy chapters only
- NumPy official "Broadcasting" and "Indexing" guides — read them twice

**Spatial & kinematics**
- Modern Robotics (Lynch & Park), Ch. 3 and 4–6 — free lectures suffice
- Zhou et al., *On the Continuity of Rotation Representations in Neural Networks*

**Datasets & infrastructure**
- *Open X-Embodiment: Robotic Learning Datasets and RT-X Models* — read the normalisation appendix closely
- LeRobot dataset format documentation
- *DROID: A Large-Scale In-the-Wild Robot Manipulation Dataset*
- *Universal Manipulation Interface (UMI)*
- *ALOHA* / *Mobile ALOHA*
- Foxglove and Rerun documentation — study these as *product* prior art for your capstone

**ML foundations**
- Karpathy, *Neural Networks: Zero to Hero* — the best possible fit for your background; build every step
- *The Illustrated Transformer* (Alammar), then implement one

**Policies**
- Ross et al., *DAgger*
- Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (ACT)
- Chi et al., *Diffusion Policy*
- Brohan et al., *RT-1*, *RT-2*; Kim et al., *OpenVLA*; Black et al., *π₀*

**Simulation & transfer**
- Tobin et al., *Domain Randomization…*
- Peng et al., *Sim-to-Real Transfer with Dynamics Randomization*
- MuJoCo documentation, modelling and computation chapters

**Perception**
- Qi et al., *PointNet* / *PointNet++*
- Kerbl et al., *3D Gaussian Splatting*
- Open3D registration tutorials

**Evaluation**
- *SIMPLER* / *RoboArena* — on the difficulty of evaluating robot policies fairly

---

## Appendix D — Interview Prep & Positioning

### Your pitch, in one paragraph

> "I've spent [N] years building complex reactive frontends — which means I've spent [N] years reasoning about asynchronous streams with independent clocks, which turns out to be the central problem in robotics data. I retrained into the domain: spatial math, dataset standards, policy learning, simulation, 3D perception. Then I built Robot Data Studio, an inspection and curation platform for multimodal robot datasets, with a typed schema contract running from Pydantic models through the API to the TypeScript client. I used it to run a controlled study showing that human-in-the-loop curation through good tooling beat automated filtering by [X] points of success rate, and took 40 minutes instead of an estimated 6 hours. Robotics teams are curating million-episode datasets with matplotlib. I build the instruments."

That is a specific, defensible, and genuinely uncommon position. It is much stronger than an apologetic "I'm transitioning from frontend."

### Questions to have crisp answers for

1. **"Design a dataset for [task X]."** The core question. Structure: task definition → embodiment and sensors → action space and rate → schema and units → collection protocol → annotation and ground truth → quality gates → evaluation splits → known limitations. Practise until it's a fluent 5-minute answer.
2. **"How would you evaluate whether more data would help?"** Scaling curve on subsets, ablation by data slice, curation-vs-volume comparison, held-out generalisation axes. You'll have run exactly this.
3. **"How do you handle heterogeneous data from 20 different labs?"** The OXE problem. Answer with your converter as the concrete example.
4. **"What's wrong with success rate as a metric?"** No confidence intervals, no failure taxonomy, criterion sensitivity, no OOD decomposition, evaluator variance.
5. **"Why should we hire a frontend engineer for this?"** Have this ready, and answer it *confidently* rather than defensively. The honest answer is strong: the time-join insight, the schema-to-UI contract, and the fact that their researchers are currently curating datasets with tools nobody designed.
6. **"What don't you know yet?"** Name it plainly: you have not worked with physical hardware, you have not shipped a model to production robots, your ML depth is months not years. Then say what you did about it and what you'd need. Candidates who can accurately state their own limits are trusted more, not less.

### Portfolio checklist before applying

- [ ] GitHub profile README linking the artefacts, most relevant first
- [ ] **A live, clickable demo** — the single highest-leverage item on this list
- [ ] `robot-data-studio` with a 15-minute quickstart from a clean clone
- [ ] Hugging Face dataset with a complete dataset card
- [ ] The blog post on what frontend engineering offers robotics data
- [ ] CV rewritten in robotics-data vocabulary (schemas, trajectories, embodiments, evaluation) — not generic frontend terms
- [ ] LinkedIn headline naming the target field explicitly
- [ ] A 3–5 minute demo video
- [ ] One meaningful component ported to React, or a ready answer about framework portability

### Where to apply

Cast wider than research roles at frontier labs. Realistic and genuinely good targets:

- **AI data labs** — micro1, Scale, Surge, Turing, Invisible: they all run *platforms* for human data contribution.
- **Annotation/data platforms** — Encord, Segments.ai, Labelbox, Voxel51: multimodal 3D data tooling is their product.
- **Robotics tooling companies** — Foxglove, Rerun: web-first robotics visualisation is literally the business.
- **Robotics companies' data/infra teams** — usually a separate and less-contested hiring pipeline than their research teams.
- **Autonomous vehicle data teams** — mature versions of the same problem, with more headcount.

---

## Appendix E — The Compressed 6-Month Variant

If eight months is not available, cut scope rather than depth. Removing whole modules is better than half-learning all of them.

| Month | Content |
|---|---|
| **1** | Month 0 bridge, compressed to 4 weeks — Python + NumPy only, drop the extra linear algebra and do it inline in Month 2 |
| **2** | Module 1 (spatial math) + `frame-viz` |
| **3** | Module 2 (data protocols) + `dataset-inspector` — **do not compress this one** |
| **4** | Module 3 (ML foundations), compressed — skip the transformer implementation, use a pretrained encoder as a black box |
| **5** | Module 4 (policy learning) — train ACT only, skip diffusion; run the ablation |
| **6** | Capstone, scoped down: pipeline + studio + one ablation result. Drop simulation and 3D perception entirely; use existing LeRobot datasets instead of generating your own |

**What you lose:** simulation and synthetic-data generation (Module 5) and 3D perception (Module 6). Both are real gaps, and you should say so in interviews rather than paper over them.

**What you keep:** the differentiated artefact, one real experimental result, and enough domain fluency to hold a technical conversation. That's the load-bearing 80%.

---

## A closing note on realism

The backend-track version of this document ends by saying six months won't make you a robotics researcher, but will make you a credible data-infrastructure specialist. The frontend version needs a sharper version of the same honesty:

Eight months will not make you competitive for a Member of Technical Staff research role against candidates with robotics PhDs and published papers. What it will make you is **the strongest candidate in the room for robotics data tooling** — a role that exists at nearly every company on the target list, is chronically under-hired for, and sits three desks from the research team.

Take that seat. Spend eighteen months with real datasets, real researchers, and real failure modes in front of you. Then the research role is an internal transfer with a track record behind it, rather than a cold application with a portfolio.

The single highest-leverage thing in this document is the Module 7 capstone with a live demo and one real experimental result. If time compresses, cut Modules 5 and 6 and protect that.
