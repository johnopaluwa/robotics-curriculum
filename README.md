# robotics-curriculum

# Embodied AI & Robotics Data Engineering — 6-Month Transition Curriculum

**Target role:** Member of Technical Staff (MTS), Robotics Research & Data Infrastructure
**Reference posting:** micro1 — Robotics Research (remote, base $200k–$320k + equity)
**Assumed background:** Senior software engineer — Python, system architecture, production data pipelines
**Assumed effort:** 12–15 focused hours/week (≈2h weekday evenings + one long weekend block)

---

## Table of Contents

1. [What the role actually is](#0-what-the-role-actually-is)
2. [How this curriculum is built](#0b-how-this-curriculum-is-built)
3. [Module 1 — Spatial Mathematics, Kinematics & Trajectory Schemas](#module-1--spatial-mathematics-kinematics--trajectory-schemas)
4. [Module 2 — Open Robotics Data Protocols & Ecosystems](#module-2--open-robotics-data-protocols--ecosystems-lerobot-oxe-mcap)
5. [Module 3 — Policy Learning & VLA Models](#module-3--embodied-ai-policy-learning--vla-models)
6. [Module 4 — Physics Simulation & Synthetic Data](#module-4--physics-simulation--synthetic-data-pipelines)
7. [Module 5 — 3D Vision, Point Clouds & Perception](#module-5--3d-vision-point-clouds--perception-engineering)
8. [Module 6 — Production Capstone](#module-6--production-capstone--end-to-end-robotics-data--evaluation-engine)
9. [Timeline & milestone checklist](#timeline--milestone-checklist)
10. [Appendix A — Hardware & environment notes](#appendix-a--hardware--environment-notes)
11. [Appendix B — Reading list](#appendix-b--core-reading-list)
12. [Appendix C — Interview preparation map](#appendix-c--interview-preparation-map)

---

## 0. What the role actually is

Read the job description carefully and it is **not** a control-theory or hardware-controls role. The verbs are *design datasets*, *define schemas*, *define ground truth*, *define evaluations*, *run experiments on what data improves model performance*, *turn research needs into scalable pipelines*.

That is a **data infrastructure role wearing a robotics hat**. The scarce skill is not "can you tune a PID loop" — it is "can you look at a frontier robot-learning problem, decide what data would actually solve it, and build the pipeline and the benchmark that proves it."

This matters for how you study. Your existing 8+ years of pipeline/architecture experience is the majority of the job. The delta you need to close is:

| Gap | Why it matters to the role |
|---|---|
| Spatial math (SE(3), kinematics) | You cannot design an action schema if you don't know what an action *is* |
| Multimodal time-series synchronisation | Every real robot dataset is asynchronous sensors with clock drift |
| Robotics dataset standards (LeRobot, OXE, RLDS, MCAP) | This is the literal file format of the job |
| Policy learning intuition (BC, ACT, Diffusion Policy, VLAs) | You must know what the consumer of your data needs |
| Simulation & domain randomisation | The scalable half of "data collection methods" |
| 3D perception (RGB-D, point clouds, extrinsics) | Ground truth and annotation design for spatial tasks |
| Evaluation design | The differentiator — most candidates cannot design a benchmark |

**Positioning note:** don't try to out-publish a robotics PhD. Compete on the axis where you're already stronger — production-grade data systems, schema design, validation, reproducibility, and evaluation rigour. The capstone in Module 6 is engineered to display exactly that.

---

## 0b. How this curriculum is built

Each module contains:

- **Objective** — the one sentence that defines "done".
- **Week-by-week plan** — so you never open a session wondering what to do.
- **Key concepts & theory** — the vocabulary you'll be interviewed on.
- **Applied code lab** — runnable starting points, not pseudocode.
- **Deliverables** — artefacts that go in the portfolio repo.
- **Self-check** — questions you must be able to answer without notes before moving on.

**Rule:** every module ships a public artefact. Six modules, six repos (or six directories in one monorepo that becomes the capstone). Learning that produces no artefact does not count toward the role.

**Suggested workspace layout — start this in Week 1:**

```
robotics-portfolio/
├── m1-spatial-math/
├── m2-data-protocols/
├── m3-policy-learning/
├── m4-sim-synthetic/
├── m5-perception-3d/
└── m6-capstone-eds/        # absorbs the best of 1–5
```

---

## Module 1 — Spatial Mathematics, Kinematics & Trajectory Schemas

> **Month 1 · Objective:** Build fluency in 3D rigid-body transformations, and be able to take raw asynchronous sensor logs and emit a single correctly-aligned, schema-validated trajectory file.

### Week-by-week

| Week | Focus | Output |
|---|---|---|
| 1 | Rotations: matrices, Euler, quaternions, axis-angle. Conversions & pitfalls. | Rotation conversion utility + property-based tests |
| 2 | SE(3): homogeneous transforms, frame composition, inverses, the `T_a_b` naming discipline | Frame-graph transform resolver |
| 3 | Forward/inverse kinematics on a 6–7 DoF arm; joint vs Cartesian space | FK/IK notebook against a real URDF |
| 4 | Time-series alignment: clock drift, resampling, SLERP, schema definition | Sync tool emitting HDF5/MCAP |

### Key concepts & theory

**3D geometry and the SE(3) group**

- Rigid transformations: translation `t ∈ R³` and rotation `R ∈ SO(3)`; a pose is the pair `(R, t)`.
- Orientation representations and their trade-offs:
  - **Rotation matrix** (9 numbers, 3 DoF) — composable, no singularities, redundant; must be re-orthonormalised after numerical drift.
  - **Euler angles** (3 numbers) — human-readable, order-dependent (`xyz` ≠ `zyx`), suffers **gimbal lock**. Never store these in a dataset.
  - **Quaternion** `q = w + xi + yj + zk` (4 numbers) — no gimbal lock, cheap composition, **double cover** (`q` and `−q` are the same rotation → canonicalise the sign or your regression loss will be discontinuous).
  - **Axis-angle / Rodrigues vector** (3 numbers) — compact, the usual choice for *network output* of rotations; ambiguous at ±π.
  - **6D rotation representation** (Zhou et al.) — the continuous representation most modern policies regress, because quaternions and Euler angles are discontinuous as regression targets. Know why this matters: **the representation you choose in the schema directly changes trainability.**
- Homogeneous 4×4 matrices in SE(3); composition `T_a_c = T_a_b @ T_b_c`; inverse `T_b_a = T_a_b⁻¹`.
- **Naming discipline:** always name transforms `T_target_source` (read right-to-left, adjacent indices cancel). This single convention eliminates most frame bugs in a data pipeline.

**Forward & inverse kinematics**

- Joint space (`q ∈ Rⁿ`, n = number of joints) vs Cartesian workspace (6-DoF end-effector pose).
- **Forward kinematics:** joint angles → end-effector pose. Deterministic, a chain of SE(3) products from the URDF.
- **Inverse kinematics:** target pose → joint angles. Non-unique (elbow-up/elbow-down), may have no solution (outside workspace), solved numerically (Jacobian pseudo-inverse, damped least squares) or in closed form for special geometries.
- **Jacobian** `J(q)`: maps joint velocities to end-effector twist. Singularities = loss of rank = the arm cannot move in some direction. Relevant to data quality: trajectories near singularities produce huge joint velocities and look like "spikes" in your logs.
- Why this matters for datasets: **the action space choice** (joint position / joint velocity / end-effector delta pose / absolute pose) determines whether a policy trained on your data can transfer across robot embodiments.

**Multimodal time-series synchronisation**

- Real rigs stream at different rates: RGB 30 Hz, depth 30 Hz (often phase-offset), joint encoders 100–1000 Hz, IMU 200 Hz, force/torque 500–1000 Hz.
- Sources of misalignment: independent hardware clocks, driver buffering, USB latency jitter, dropped frames, PTP/NTP drift over long sessions.
- Fixes: hardware trigger/sync lines where possible; otherwise timestamp on arrival **and** record device timestamps, then estimate offset by cross-correlation of a shared signal (e.g. a deliberate motion spike visible to both camera and encoder).
- Resampling: zero-order hold for discrete signals (gripper open/close), linear interpolation for positions, **SLERP** for quaternions (never lerp raw quaternion components — the result isn't unit-norm and the angular velocity is wrong).
- Decide and document: **what is the canonical clock?** Usually the camera, because you cannot interpolate pixels.

### Applied engineering & code lab

```python
"""m1-spatial-math/spatial.py — SE(3) utilities and trajectory alignment."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


class SpatialTransform:
    """SE(3) frame transformations and rotation interpolation."""

    @staticmethod
    def se3(translation, rotation: Rotation) -> np.ndarray:
        """Build a 4x4 homogeneous transform from a translation and a rotation."""
        matrix = np.eye(4)
        matrix[:3, :3] = rotation.as_matrix()
        matrix[:3, 3] = np.asarray(translation, dtype=float)
        return matrix

    @staticmethod
    def se3_from_euler(translation, euler_deg, order: str = "xyz") -> np.ndarray:
        return SpatialTransform.se3(
            translation, Rotation.from_euler(order, euler_deg, degrees=True)
        )

    @staticmethod
    def invert(T: np.ndarray) -> np.ndarray:
        """Inverse of a homogeneous transform (cheaper and stabler than np.linalg.inv)."""
        R, t = T[:3, :3], T[:3, 3]
        out = np.eye(4)
        out[:3, :3] = R.T
        out[:3, 3] = -R.T @ t
        return out

    @staticmethod
    def slerp(q_start: np.ndarray, q_end: np.ndarray, alpha: float) -> np.ndarray:
        """Spherical linear interpolation between two quaternions (scipy [x,y,z,w] order).

        Note: scipy exposes Slerp as a separate class -- Rotation has no .slerp method.
        Many code samples on the internet get this wrong.
        """
        key_rots = Rotation.from_quat(np.stack([q_start, q_end]))
        interpolator = Slerp([0.0, 1.0], key_rots)
        return interpolator(float(alpha)).as_quat()

    @staticmethod
    def canonicalize_quat(q: np.ndarray) -> np.ndarray:
        """Resolve the q/-q double cover by forcing a non-negative scalar part."""
        q = np.asarray(q, dtype=float)
        return -q if q[..., 3] < 0 else q


def resample_to_clock(
    src_times: np.ndarray,
    src_values: np.ndarray,
    target_times: np.ndarray,
    kind: str = "linear",
) -> np.ndarray:
    """Resample a signal onto a canonical clock.

    kind: 'linear' for positions/velocities, 'zoh' for discrete signals,
          'slerp' for quaternion columns (expects shape (N, 4)).
    """
    if kind == "slerp":
        rots = Rotation.from_quat(src_values)
        return Slerp(src_times, rots)(np.clip(target_times, src_times[0], src_times[-1])).as_quat()
    if kind == "zoh":
        idx = np.searchsorted(src_times, target_times, side="right") - 1
        return src_values[np.clip(idx, 0, len(src_values) - 1)]
    return np.stack(
        [np.interp(target_times, src_times, src_values[:, c]) for c in range(src_values.shape[1])],
        axis=-1,
    )
```

**Sanity tests to write (this is the part interviewers respect):**

```python
# Property tests -- run these with hypothesis or plain random sampling.
# 1. Round-trip: euler -> matrix -> quat -> matrix reproduces the original within 1e-9.
# 2. Composition: T_a_c == T_a_b @ T_b_c for randomly generated frames.
# 3. Inverse: invert(T) @ T == I.
# 4. SLERP endpoints: slerp(q0, q1, 0) == q0 and slerp(q0, q1, 1) == q1 (up to sign).
# 5. SLERP constant angular velocity: the angle between slerp(t) and slerp(t+dt)
#    is constant in t -- this is the property naive lerp violates.
```

### Deliverables

1. **`spatialmath-lite`** — a small, tested Python package for SE(3) composition, rotation conversions, and frame-graph resolution (given `T_base_cam` and `T_cam_ee`, resolve `T_base_ee` automatically by graph search). Include the property tests above.
2. **`traj-sync`** — a CLI that ingests unsynchronised CSV logs (joint states at 100 Hz, camera frame timestamps at 30 Hz, gripper events at irregular intervals) and emits a single aligned HDF5 (and/or MCAP) file on the camera clock, with an explicit written schema and a report of how many samples were interpolated vs extrapolated.
3. **A one-page schema document** (`SCHEMA.md`) that specifies field names, dtypes, units, frame conventions, and rotation representation. This document is a work sample for the exact job.

### Self-check

- Why is a quaternion a bad regression target for a neural policy, and what do modern policies use instead?
- Given `T_world_base`, `T_base_cam`, `T_cam_object`, write the expression for the object pose in the world frame.
- Your gripper signal is binary and your camera is 30 Hz; why is linear interpolation wrong here?
- Two sensors disagree by a constant 40 ms. How do you detect that offset automatically from data alone?

---

## Module 2 — Open Robotics Data Protocols & Ecosystems (LeRobot, OXE, MCAP)

> **Month 2 · Objective:** Know the real formats the field uses, and be able to convert, validate and audit large heterogeneous robot datasets.

### Week-by-week

| Week | Focus | Output |
|---|---|---|
| 1 | LeRobot dataset format: structure, metadata, video encoding, episode indexing | Load, inspect, and visualise an existing LeRobot dataset |
| 2 | Open X-Embodiment / RLDS: heterogeneity across 20+ embodiments, action space normalisation | OXE → LeRobot converter for one source dataset |
| 3 | Storage engineering: HDF5, Zarr, Parquet, MCAP, WebDataset; chunking and I/O throughput | Benchmark write/read throughput across formats |
| 4 | Dataset QA: automated auditing, statistics, outlier detection | `traj-audit` validation suite + HTML report |

### Key concepts & theory

**Robotics dataset architectures**

- **Open X-Embodiment (OXE):** the field's largest pooled dataset — 1M+ trajectories aggregated from 20+ robot platforms and 60+ constituent datasets. The interesting engineering problem it exposes is **heterogeneity**: every source lab used different action spaces, control rates, camera counts, coordinate conventions, and gripper semantics. Read how they normalised it — that normalisation decision *is* the job you're applying for.
- **RLDS** (Reinforcement Learning Datasets, TFDS-based) — the original OXE serialisation. Episode = sequence of steps; step = `{observation, action, reward, is_first, is_last, is_terminal}`.
- **Hugging Face LeRobot** — the current de-facto community standard. Flat tabular frames + separately encoded videos:
  - `observation.images.<camera_name>` — video-encoded (MP4/AV1), decoded lazily per frame
  - `observation.state` — proprioception vector
  - `action` — the commanded action vector
  - `episode_index`, `frame_index`, `index`, `timestamp`, `task_index`
  - `meta/info.json` (fps, feature shapes, video specs), `meta/episodes.jsonl`, `meta/tasks.jsonl`, `meta/stats.json` (per-feature mean/std/min/max — used for normalisation at train time)
  - Understand *why* they store video rather than raw frames: a 50-episode dataset is ~100 GB raw and ~2 GB encoded. Understand the cost: random access requires keyframe-aware seeking, which dominates dataloader time.

**Storage technologies — and when each wins**

| Format | Strength | Use when |
|---|---|---|
| **HDF5** | Mature, hierarchical, chunked, one file per episode | Local episode archives; the ALOHA/ACT ecosystem uses it |
| **Zarr** | Chunked like HDF5 but cloud-native, parallel-write-safe | Large synthetic generation runs, S3/GCS-backed |
| **Parquet** | Columnar, excellent for tabular state/action + metadata | LeRobot's tabular half; analytics over episodes |
| **MCAP** | ROS 2-native container, self-describing schemas, append-only, seekable | Real-robot logging; anything touching ROS or Foxglove |
| **WebDataset / TFRecord** | Sequential sharded tar/records, saturates network I/O | Distributed multi-node training |

Know the failure modes: HDF5 has no safe concurrent write; Zarr chunk size mismatched to access pattern destroys throughput; MCAP without registered schemas is unreadable later; WebDataset gives you no random access, so shuffling must be buffer-based.

**Action space formulations**

- **Absolute joint positions** — embodiment-specific, easy to learn, doesn't transfer.
- **Joint velocities** — needs a well-defined control rate; sensitive to latency.
- **End-effector absolute pose** (6-DoF + gripper) — transfers better across arms with similar workspaces.
- **End-effector delta pose** `(Δx, Δy, Δz, Δrot)` — the most common VLA action space; scale-sensitive, so normalisation statistics matter enormously.
- **Action chunking** — predicting `H` future actions at once (H ≈ 8–100) instead of one. Reduces compounding error and produces smoother motion. Your schema must therefore support efficient windowed reads.
- **Normalisation:** min-max vs mean/std vs quantile. Percentile clipping (e.g. 1st/99th) is standard because a single teleop glitch otherwise dominates the scale.

**Dataset quality — what actually goes wrong in real robot data**

- Dropped/duplicated frames; timestamps that go backwards.
- Action spikes from teleop hardware disconnects or leader-arm collisions.
- Frozen sensor channels (a camera that returns the same buffer for 40 frames).
- Episodes labelled success that are actually failures (label noise from unsupervised collection).
- Mismatched episode lengths between modalities.
- Silent unit changes mid-collection (metres vs millimetres, radians vs degrees) — the classic dataset-killer.

### Applied engineering & code lab

```python
"""m2-data-protocols/episode_writer.py — standardized HDF5 episode archive."""
import h5py
import numpy as np


def create_episode(path: str, num_frames: int, fps: int = 30) -> None:
    """Write a schema-conformant single-episode archive.

    Layout mirrors the ALOHA/ACT convention so downstream tooling can read it.
    """
    with h5py.File(path, "w") as f:
        # --- Metadata: everything a future reader needs to interpret the numbers.
        f.attrs["robot_type"] = "franka_panda"
        f.attrs["fps"] = fps
        f.attrs["schema_version"] = "1.0.0"
        f.attrs["action_space"] = "joint_velocity+gripper"
        f.attrs["rotation_repr"] = "quat_xyzw"
        f.attrs["length_units"] = "meters"
        f.attrs["angle_units"] = "radians"

        obs = f.create_group("observations")
        # Chunk along time so a windowed read touches few chunks.
        obs.create_dataset(
            "images/front", shape=(num_frames, 480, 640, 3), dtype="uint8",
            chunks=(1, 480, 640, 3), compression="gzip", compression_opts=4,
        )
        obs.create_dataset(
            "images/wrist", shape=(num_frames, 480, 640, 3), dtype="uint8",
            chunks=(1, 480, 640, 3), compression="gzip", compression_opts=4,
        )
        obs.create_dataset("qpos", shape=(num_frames, 7), dtype="float32")
        obs.create_dataset("qvel", shape=(num_frames, 7), dtype="float32")
        obs.create_dataset("ee_pose", shape=(num_frames, 7), dtype="float32")  # xyz + quat
        obs.create_dataset("timestamp", shape=(num_frames,), dtype="float64")

        # 7 joint velocity targets + 1 gripper command
        f.create_dataset("actions", shape=(num_frames, 8), dtype="float32")
```

```python
"""m2-data-protocols/audit.py — automated trajectory auditing."""
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
    joint_limits: np.ndarray,   # shape (n_joints, 2)
    fps: int,
    name: str = "episode",
) -> AuditReport:
    """Flag the failure modes that actually occur in teleop data."""
    r = AuditReport(episode=name)

    # 1. Monotonic, gap-free clock.
    dt = np.diff(timestamps)
    if np.any(dt <= 0):
        r.errors.append(f"non-monotonic timestamps at {np.flatnonzero(dt <= 0)[:5].tolist()}")
    expected = 1.0 / fps
    dropped = np.flatnonzero(dt > 1.5 * expected)
    if dropped.size:
        r.errors.append(f"{dropped.size} frame gaps > 1.5x nominal period")

    # 2. Action spikes -- robust z-score on the first difference.
    da = np.abs(np.diff(actions, axis=0))
    mad = np.median(np.abs(da - np.median(da, axis=0)), axis=0) + 1e-8
    spikes = np.flatnonzero((da / (1.4826 * mad) > 8).any(axis=1))
    if spikes.size:
        r.warnings.append(f"{spikes.size} action spikes (>8 robust sigma)")

    # 3. Kinematic bounds.
    below = (qpos < joint_limits[:, 0]).any(axis=1)
    above = (qpos > joint_limits[:, 1]).any(axis=1)
    if below.any() or above.any():
        r.errors.append(f"{int((below | above).sum())} frames outside joint limits")

    # 4. Frozen channels -- a sensor returning a constant buffer.
    for i in range(qpos.shape[1]):
        run = np.max(np.diff(np.flatnonzero(np.diff(qpos[:, i]) != 0), prepend=0), initial=0)
        if run > fps:  # unchanged for more than a second
            r.warnings.append(f"joint {i} frozen for {run} frames")

    # 5. Jerk -- third derivative of position; smoothness proxy.
    jerk = np.diff(qpos, n=3, axis=0) * (fps ** 3)
    if np.abs(jerk).max() > 5e3:
        r.warnings.append(f"peak jerk {np.abs(jerk).max():.0f} rad/s^3 exceeds threshold")

    return r
```

### Deliverables

1. **`oxe2lerobot`** — an ETL that pulls one Open X-Embodiment source dataset and converts it to LeRobot format, with an explicit, *documented* mapping of action space, coordinate convention and gripper semantics. Write down every assumption you had to make; that document is the interview material.
2. **`traj-audit`** — the validation suite above, extended to scan a whole dataset directory and emit an HTML/Markdown report: per-episode pass/fail, dataset-level statistics, histograms of action magnitudes, a "suspicious episodes" ranked list.
3. **A storage benchmark write-up** — same 10 GB of synthetic trajectories written to HDF5, Zarr, Parquet+MP4 and WebDataset; measure write time, size on disk, sequential read throughput, and random-window read latency. Publish the table.

### Self-check

- Why does LeRobot store images as encoded video, and what does that cost you at training time?
- You are merging two datasets: one records absolute end-effector poses in millimetres, the other delta poses in metres. List every step needed to unify them.
- What is action chunking, and how does it change your storage layout?
- Name three dataset defects that a schema validator catches and three it cannot.

---

## Module 3 — Embodied AI Policy Learning & VLA Models

> **Month 3 · Objective:** Understand — by training them yourself — how modern policies consume observations and emit actions, so you can reason about what data would improve them.

### Week-by-week

| Week | Focus | Output |
|---|---|---|
| 1 | Behavioural cloning fundamentals; covariate shift; DAgger | BC baseline trained on PushT |
| 2 | Action Chunking Transformers (ACT); temporal ensembling | ACT trained + loss curves logged |
| 3 | Diffusion Policy: DDPM/flow-matching over action sequences | Diffusion Policy trained; compare vs BC |
| 4 | VLAs: RT-2, OpenVLA, π₀; what changes at scale; ablation study | Data ablation write-up |

### Key concepts & theory

**Imitation learning**

- **Behavioural cloning:** supervised regression `a_t = π(o_t)`. Trivial to implement; fails in a specific, learnable way.
- **Covariate shift / compounding error:** the policy is trained on the *expert's* state distribution. Any small error moves it slightly off-distribution, where its predictions are worse, which moves it further off — error compounds as `O(εT²)` over a horizon `T`. This is *the* central problem of imitation learning and the reason data collection strategy matters.
- **Mitigations** — and note that all of them are *data* decisions, i.e. your job:
  - **DAgger** — iteratively collect expert corrections on states the policy actually visits.
  - **Recovery data** — deliberately perturb the robot and record the human recovering. Small quantities of this data have outsized effect.
  - **Action chunking** — predict `H` steps at once; reduces the number of decision points and therefore compounding.
  - **Temporal ensembling** — average overlapping predicted chunks for smoothness.
  - Augmentation: random crop, colour jitter, camera-pose jitter.
- **Multimodality of human demonstrations:** if half the demonstrators go left around an obstacle and half go right, an L2-regression policy learns the *mean* — straight into the obstacle. This is the single best motivation for diffusion policies, and a data-curation issue you'll be expected to discuss.

**Architectures worth knowing by name**

| Model | Core idea | Why it matters to a data role |
|---|---|---|
| **BC-MLP / BC-RNN** | Regression baseline | Your control condition in every experiment |
| **ACT** | Transformer + CVAE, predicts action chunks | The standard ALOHA-family baseline; cheap to train |
| **Diffusion Policy** | Denoising diffusion over action sequences | Handles multimodal demonstrations; strong default |
| **RT-1 / RT-2** | Discretised action tokens; RT-2 co-trains on web VQA | Showed web data improves *physical* generalisation |
| **OpenVLA** | 7B open VLA on OXE; LoRA-finetunable | Your realistic hands-on VLA |
| **π₀ / π₀.₅** | Flow-matching action expert on a VLM backbone | Current frontier reference point |
| **World models** (Dreamer, Genie-style, V-JEPA-style) | Learn dynamics/prediction rather than actions | "What data teaches physics?" — a real interview question |

**Evaluation — the part most candidates get wrong**

- Validation loss is a weak proxy for task success. Report both, and say so.
- **Success rate** over N trials with a *stated* success criterion, plus a binomial confidence interval. `9/10` and `90/100` are not the same claim.
- Held-out axes: unseen object instances, unseen positions, unseen lighting, unseen distractors, unseen backgrounds. Each is a separate generalisation axis and should be a separate column in your results table.
- Smoothness metrics: jerk, path length ratio, action-rate saturation.
- Failure taxonomy: never-grasped / grasped-then-dropped / wrong-object / collided / timed-out. A benchmark that only reports a success percentage tells a researcher nothing about what data to collect next; a failure taxonomy tells them exactly.

### Applied engineering & code lab

```python
"""m3-policy-learning/bc_policy.py — chunked behavioural cloning baseline."""
import torch
import torch.nn as nn


class ChunkedBCPolicy(nn.Module):
    """Maps a visual embedding + proprioception to a chunk of future actions.

    Predicting a chunk (rather than one step) is the cheapest single
    improvement over vanilla BC: fewer decision points => less compounding error.
    """

    def __init__(self, visual_dim: int, state_dim: int, action_dim: int, horizon: int):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(visual_dim + state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim * horizon),
        )

    def forward(self, visual_features: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        x = torch.cat([visual_features, state], dim=-1)
        return self.net(x).view(x.shape[0], self.horizon, self.action_dim)


@torch.no_grad()
def temporal_ensemble(chunk_buffer: list[torch.Tensor], k: float = 0.01) -> torch.Tensor:
    """Exponentially-weighted average of overlapping predicted chunks (ACT-style).

    chunk_buffer[i] is the prediction for the current timestep made i steps ago.
    Older predictions get lower weight.
    """
    weights = torch.tensor([torch.exp(torch.tensor(-k * i)) for i in range(len(chunk_buffer))])
    weights = weights / weights.sum()
    stacked = torch.stack(chunk_buffer)  # (n, action_dim)
    return (stacked * weights[:, None]).sum(0)
```

**The experiment that matters (run this in Week 4):**

Take the LeRobot PushT dataset and train the *same* policy on systematically varied data:

| Condition | Episodes | Question answered |
|---|---|---|
| Full dataset | 200 | Baseline |
| Half | 100 | Where is the data-scaling knee? |
| Quarter | 50 | |
| Full minus 10% worst-quality (per your Module 2 auditor) | 180 | **Does curation beat volume?** |
| Full + augmentation | 200 | Does augmentation substitute for data? |

Publish the curve. "I ran a controlled ablation showing that removing the worst 10% of episodes matched the performance of 30% more raw data" is precisely the sentence that gets you this job.

### Deliverables

1. **Trained ACT or Diffusion Policy** on a LeRobot benchmark dataset, with W&B (or plain-CSV) logged loss curves and evaluation success rates.
2. **`traj-viz`** — a visualiser overlaying predicted action chunks on ground-truth trajectories, rendered over the camera feed. Being able to *see* where the policy diverges is a data-debugging tool, which is the framing to use when you present it.
3. **Data ablation report** — the table above, with confidence intervals and a written conclusion.

### Self-check

- Explain covariate shift to a non-ML engineer in three sentences.
- Why does an L2 regression policy fail on multimodal demonstrations, and how does a diffusion policy fix it?
- You have budget for 100 more teleop episodes. How do you decide *which* 100 to collect?
- What is temporal ensembling and what artefact does it remove?

---

## Module 4 — Physics Simulation & Synthetic Data Pipelines

> **Month 4 · Objective:** Generate domain-randomised synthetic datasets at scale, and be able to argue precisely about where sim data helps and where it lies.

### Week-by-week

| Week | Focus | Output |
|---|---|---|
| 1 | MuJoCo/MJCF basics; loading a robot; scripted control | Tabletop scene with a 7-DoF arm |
| 2 | Scripted pick-and-place with an IK-based controller; offscreen rendering | Working pick-place episode generator |
| 3 | Domain randomisation: visual + dynamics; randomisation config schema | `randomization_rules.yaml` + sampler |
| 4 | Scale-out: parallel episode generation, Zarr/HDF5 export, generation QA | 5k-episode synthetic dataset |

### Key concepts & theory

**Simulators**

- **MuJoCo** (now DeepMind, Apache-2.0) — fast, accurate soft-contact solver, CPU-first, trivial to install, excellent Python bindings. **MuJoCo is the right choice for this curriculum** (see Appendix A).
- **MuJoCo MJX** — JAX/GPU-parallel MuJoCo for massively parallel rollouts.
- **NVIDIA Isaac Sim / Isaac Lab** — photorealistic RTX rendering, GPU-parallel physics, USD scene format. Powerful but demands an RTX GPU and a large install; know the concepts and the vocabulary even if you don't run it.
- **Others worth naming:** PyBullet (simple, legacy), Genesis, SAPIEN, Drake (contact-rich, verification-oriented).

**Description formats**

- **URDF** — the ROS lingua franca: links, joints, inertials, visual/collision geometry. Limitations: no closed kinematic chains, weak contact parameters.
- **MJCF** — MuJoCo's XML: richer contact/solver parameters, actuators, sensors, tendons.
- **USD** — Pixar's format, Isaac Sim's native scene graph, composable layers.
- Know that converting URDF → MJCF is routine but lossy in exactly the parameters that matter for contact-rich manipulation (friction, solref/solimp, damping).

**Domain randomisation**

- **Visual DR:** lighting position/colour/intensity, textures and materials, background clutter and distractor objects, camera intrinsics (focal length, distortion) and extrinsics (pose jitter), sensor noise, motion blur, exposure.
- **Dynamics DR:** mass, inertia, friction coefficients (sliding/torsional/rolling), joint damping and armature, actuator gains, control latency, observation noise.
- **The key insight:** DR works by making the real world look like *one more sample* from the training distribution. Too little → no transfer. Too much → the policy learns an over-conservative average behaviour and underperforms. The width of the randomisation distribution is a hyperparameter you must tune and report.
- **Automatic DR / curriculum DR** — widen ranges as the policy succeeds.
- **Structured/real-to-sim DR** — calibrate the ranges from measured real-world variation instead of guessing.

**The sim-to-real gap — be specific, not hand-wavy**

| Gap source | Manifestation |
|---|---|
| Contact & friction modelling | Objects slip in real life but not in sim, or vice versa |
| Rendering realism | Policy overfits to clean sim textures |
| Actuator dynamics & latency | Sim assumes instantaneous torque; real motors don't |
| Deformables, cloth, liquids | Poorly modelled in most rigid-body engines |
| Sensor noise characteristics | Real depth cameras have holes, flying pixels, reflective failures |

**Reproducibility of generated data** — this is your software-engineering edge:

- Every episode must record the seed and full randomisation parameter vector as metadata.
- The generation config must be versioned and hashed into the dataset metadata.
- Regeneration from a seed must be bit-identical, or you cannot debug the pipeline.

### Applied engineering & code lab

```python
"""m4-sim-synthetic/generator.py — reproducible domain-randomised data generation."""
from dataclasses import asdict, dataclass

import mujoco
import numpy as np


@dataclass
class RandomizationSpec:
    """Serialised into every episode's metadata so runs are reproducible."""
    friction_range: tuple = (0.5, 1.5)
    mass_scale_range: tuple = (0.8, 1.2)
    light_pos_jitter: float = 0.3
    camera_pos_jitter: float = 0.02
    camera_euler_jitter_deg: float = 2.0


class MujocoDataGenerator:
    """Headless synthetic trajectory collector."""

    def __init__(self, model_xml_path: str, spec: RandomizationSpec, width=640, height=480):
        self.model = mujoco.MjModel.from_xml_path(model_xml_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.spec = spec

    def randomize(self, seed: int) -> dict:
        """Apply DR from an explicit seed; return the sampled parameters as metadata."""
        rng = np.random.default_rng(seed)
        params = {"seed": int(seed), **asdict(self.spec)}

        # Dynamics: per-geom sliding friction.
        friction = rng.uniform(*self.spec.friction_range, size=self.model.ngeom)
        self.model.geom_friction[:, 0] = friction
        params["friction"] = friction.tolist()

        # Dynamics: body masses.
        scale = rng.uniform(*self.spec.mass_scale_range, size=self.model.nbody)
        self.model.body_mass[:] = self.model.body_mass * scale
        params["mass_scale"] = scale.tolist()

        # Visual: object colours.
        self.model.geom_rgba[:, :3] = rng.uniform(0.1, 0.9, size=(self.model.ngeom, 3))

        # Visual: light positions.
        if self.model.nlight:
            self.model.light_pos[:] += rng.normal(0, self.spec.light_pos_jitter,
                                                  size=self.model.light_pos.shape)
        return params

    def step(self, camera: str = "hand_eye_cam"):
        mujoco.mj_step(self.model, self.data)
        self.renderer.update_scene(self.data, camera=camera)
        return (
            self.renderer.render(),
            self.data.qpos.copy(),
            self.data.qvel.copy(),
            float(self.data.time),
        )

    def rollout(self, controller, max_steps: int, seed: int) -> dict:
        """Generate one episode. `controller(data) -> ctrl vector`."""
        mujoco.mj_resetData(self.model, self.data)
        params = self.randomize(seed)
        frames, qpos, qvel, actions, times = [], [], [], [], []
        for _ in range(max_steps):
            ctrl = controller(self.data)
            self.data.ctrl[:] = ctrl
            rgb, q, dq, t = self.step()
            frames.append(rgb); qpos.append(q); qvel.append(dq)
            actions.append(np.asarray(ctrl).copy()); times.append(t)
        return {
            "images": np.stack(frames),
            "qpos": np.stack(qpos),
            "qvel": np.stack(qvel),
            "actions": np.stack(actions),
            "timestamp": np.asarray(times),
            "randomization": params,
        }
```

### Deliverables

1. **Headless MuJoCo pick-and-place environment** with a scripted IK controller that reliably completes the task.
2. **`synth-gen`** — a parallel generator producing 5,000 episodes with per-episode DR, exporting to Zarr or HDF5 in your Module 1 schema, with full randomisation metadata per episode and a deterministic seed → dataset guarantee.
3. **A DR ablation** — train the Module 3 policy on (a) narrow DR, (b) wide DR, (c) no DR, and evaluate all three on a *held-out* simulated "test world" with parameters outside the training ranges. This is your stand-in for sim-to-real and it is a legitimate, publishable experiment without owning a robot.

### Self-check

- Why can too much domain randomisation hurt?
- What is lost when converting URDF → MJCF for a contact-rich task?
- Your generated dataset trains a policy that reaches 95% in sim and 20% on a real arm. List five candidate causes in order of likelihood.
- How do you make a 5,000-episode generation run bit-reproducible?

---

## Module 5 — 3D Vision, Point Clouds & Perception Engineering

> **Month 5 · Objective:** Turn raw RGB-D telemetry into calibrated 3D representations, and build the automated quality filters that decide whether a frame is usable as training data.

### Week-by-week

| Week | Focus | Output |
|---|---|---|
| 1 | Pinhole model, intrinsics, distortion, calibration | Calibration + undistortion tool |
| 2 | Depth → point cloud; extrinsics; multi-camera registration with ICP | Two-view merged point cloud |
| 3 | Point cloud processing: downsampling, normals, segmentation, ICP refinement | Object extraction from a tabletop scene |
| 4 | Perception data QA: occlusion, sparsity, depth dropout detection | `depth-qa` filter |

### Key concepts & theory

**Camera geometry**

- **Intrinsic matrix** `K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]` — maps camera-frame 3D points to pixels.
- **Extrinsics** `[R | t]` — the rigid transform from world (or robot base) to camera frame. In a robot rig, `T_base_cam` for a fixed camera, or `T_ee_cam` for a wrist camera (found by **hand-eye calibration**, solving `AX = XB`).
- **Distortion:** radial `(k1, k2, k3)` and tangential `(p1, p2)`; undistort before any geometric reasoning.
- **Back-projection:** `X = (u − cx)·d / fx`, `Y = (v − cy)·d / fy`, `Z = d`.
- **Depth units and conventions** — the perennial bug: 16-bit depth in millimetres vs float metres; depth-to-colour registration; `depth_scale` from the driver. Assert on it, don't assume.
- **Depth sensing modalities:** structured light (Kinect-style, fails in sunlight), active stereo (RealSense D4xx, needs texture), ToF (Orbbec/Kinect Azure, multipath artefacts on corners), LiDAR (sparse, long range). Each has a *distinct* failure signature you can detect programmatically.

**3D representations**

- **Point clouds** — unordered `(N, 3)` + optional colour/normals. Networks: PointNet → PointNet++ → PointNeXt; and sparse-voxel approaches (MinkowskiNet).
- **Voxel grids / TSDFs** — regular structure, fusion-friendly (KinectFusion-style), memory-hungry.
- **Meshes** — from Poisson or marching cubes over a TSDF.
- **NeRF** — implicit radiance field, high-fidelity novel views, slow.
- **3D Gaussian Splatting (3DGS)** — explicit, real-time rendering; now used to build *simulation assets* and augmented views from egocentric video. This is a live area for robotics data: scan the real scene, render novel viewpoints, augment the training set.
- **Registration:** ICP (point-to-point and point-to-plane), global registration with FPFH features + RANSAC before ICP refinement. Know that ICP is a *local* method and needs a decent initial guess.

**Perception data quality checks — the deliverable that ties to the job**

- Depth validity ratio per frame (fraction of non-zero depth pixels).
- Flying pixels at depth discontinuities.
- Specular/reflective dropout (a metal object returning zero depth).
- **Occlusion of the target object by the robot's own arm** — the most common silent quality killer in wrist-camera manipulation data.
- Motion blur (variance-of-Laplacian threshold).
- Exposure clipping (histogram saturation at 0/255).
- Point-cloud sparsity in the region of interest.

### Applied engineering & code lab

```python
"""m5-perception-3d/pointcloud.py — RGB-D to point cloud and quality filters."""
import numpy as np
import open3d as o3d


def rgbd_to_point_cloud(
    color_img: np.ndarray,          # (H, W, 3) uint8
    depth_img: np.ndarray,          # (H, W) float32 in METERS
    K: np.ndarray,                  # (3, 3) intrinsics
    depth_trunc: float = 3.0,
) -> o3d.geometry.PointCloud:
    """Convert a synchronised RGB-D frame pair into a coloured point cloud."""
    h, w = depth_img.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(color_img, dtype=np.uint8)),
        o3d.geometry.Image(np.ascontiguousarray(depth_img, dtype=np.float32)),
        depth_scale=1.0,            # depth already in meters
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False,
    )
    intrinsics = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)


def register_views(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    voxel: float = 0.005,
    init: np.ndarray | None = None,
) -> np.ndarray:
    """Point-to-plane ICP refinement returning T_target_source."""
    src = source.voxel_down_sample(voxel)
    tgt = target.voxel_down_sample(voxel)
    for pc in (src, tgt):
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 4, max_nn=30))

    result = o3d.pipelines.registration.registration_icp(
        src, tgt, max_correspondence_distance=voxel * 3,
        init=np.eye(4) if init is None else init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    return result.transformation


def depth_quality(depth_img: np.ndarray, roi: tuple[int, int, int, int] | None = None) -> dict:
    """Per-frame depth quality metrics. roi = (y0, y1, x0, x1)."""
    d = depth_img if roi is None else depth_img[roi[0]:roi[1], roi[2]:roi[3]]
    valid = np.isfinite(d) & (d > 0)

    # Flying pixels: large local depth gradients.
    gy, gx = np.gradient(np.where(valid, d, np.nan))
    grad = np.hypot(np.nan_to_num(gy), np.nan_to_num(gx))

    return {
        "valid_ratio": float(valid.mean()),
        "median_depth_m": float(np.median(d[valid])) if valid.any() else float("nan"),
        "flying_pixel_ratio": float((grad > 0.05).mean()),
        "dropout_ratio": float((~valid).mean()),
    }


def is_target_occluded(
    depth_img: np.ndarray,
    target_mask: np.ndarray,        # bool (H, W): where the target *should* be
    expected_depth_m: float,
    tol: float = 0.03,
) -> bool:
    """True when something (usually the arm) sits in front of the target region."""
    observed = depth_img[target_mask]
    observed = observed[np.isfinite(observed) & (observed > 0)]
    if observed.size == 0:
        return True  # no depth at all counts as unusable
    return bool(np.median(observed) < expected_depth_m - tol)
```

### Deliverables

1. **`rgbd-fuse`** — a tool ingesting two RGB-D streams (egocentric/wrist + fixed environment camera), estimating relative extrinsics via global registration + ICP, and merging into a single point cloud in the robot base frame. Include a visual before/after.
2. **`depth-qa`** — an automated per-frame quality filter producing a per-episode usability score, with a specific detector for **robot-arm occlusion of the target object**. Wire it into the Module 2 auditor.
3. **A calibration write-up** — how you obtained intrinsics and extrinsics, what the residual error was, and how error propagates into 3D ground-truth annotations. Quantifying annotation error is a research-grade habit and very few candidates do it.

### Self-check

- Given `K` and a depth pixel, write the back-projection by hand.
- Why does ICP need a good initial guess, and what do you do when you don't have one?
- Your merged cloud shows the object duplicated in two nearby positions. What's wrong?
- Name four distinct physical causes of missing depth pixels and how you'd detect each from the data.

---

## Module 6 — Production Capstone — End-to-End Robotics Data & Evaluation Engine

> **Month 6 · Objective:** Ship one public, reproducible system that demonstrates every capability the job description asks for, plus a written experimental result.

### The artefact

**Repository name:** `embodied-data-system` (EDS) — *Automated Robotics Dataset Pipeline & Benchmark*

```
embodied-data-system/
│
├── configs/
│   ├── dataset_schema.yaml        # Canonical spec: actions, images, state, units, frames
│   ├── randomization_rules.yaml   # DR parameter boundaries, versioned
│   └── benchmark_suite.yaml       # Evaluation axes and thresholds
│
├── eds/
│   ├── schema.py                  # Pydantic models; single source of truth
│   ├── generator.py               # MuJoCo synthetic batch engine (Module 4)
│   ├── ingest.py                  # Real/recorded log ingestion + clock alignment (Module 1)
│   ├── validator.py               # Schema, spike, continuity, kinematic, depth QA (Modules 2 & 5)
│   ├── converter.py               # Export: LeRobot, MCAP, Zarr, WebDataset (Module 2)
│   └── stats.py                   # Normalisation statistics, dataset cards
│
├── models/
│   ├── policy_wrapper.py          # Uniform interface over BC / ACT / Diffusion Policy
│   └── train.py                   # Training entrypoint, config-driven, seeded
│
├── evaluation/
│   ├── metrics.py                 # Success rate + CI, spatial deviation, jerk, failure taxonomy
│   ├── benchmark.py               # OOD evaluation runner
│   └── report.py                  # Markdown/HTML report generation
│
├── tests/                         # Property + regression tests on the schema and validators
├── notebooks/                     # Exploratory analysis, figures for the write-up
├── DATASET_CARD.md                # Provenance, collection method, known defects, intended use
├── RESULTS.md                     # The experiment and its conclusion
└── README.md                      # Architecture, quickstart, benchmarks
```

### Concrete requirements

**1. Pipeline execution**
Generate (or ingest) multi-episode manipulation data containing: 2× RGB-D streams, joint encoder telemetry, 6-DoF end-effector actions, gripper state, and per-episode task labels — all synchronised onto a canonical clock and conforming to your published schema.

**2. Quality infrastructure — programmatic, tested, CI-enforced**

- **Frame continuity:** zero dropped frames, monotonic timestamps, no gaps > 1.5× nominal period.
- **Kinematic bounds:** every commanded action inside joint position/velocity limits.
- **Trajectory smoothness:** jerk (3rd derivative of position) below a stated threshold.
- **Cross-modal alignment:** all modalities have equal frame counts and matching timestamps within tolerance.
- **Perception usability:** depth validity ratio and occlusion checks from Module 5.
- **Schema conformance:** dtypes, shapes, units, and required metadata fields.
- Run all of it in GitHub Actions on a small fixture dataset committed to the repo.

**3. Evaluation benchmark**
Train a baseline policy on your data and produce an automated report covering *at minimum* these generalisation axes, each with a success rate and confidence interval:

| Axis | Variation |
|---|---|
| In-distribution | Training conditions |
| Novel object pose | Initial position/orientation outside training range |
| Novel lighting | Intensity and direction shifts |
| Novel appearance | Unseen object colours/textures |
| Distractors | Additional clutter objects present |
| Long horizon | Extended episode length |

Plus a **failure taxonomy breakdown** per axis.

**4. The experiment — this is what makes it research, not a tutorial**

Pick one question and answer it with a controlled study. Good candidates:

- *Does dataset curation beat dataset volume?* Train on N episodes vs 0.8N curated-by-validator episodes.
- *Which domain randomisation axis contributes most to OOD robustness?* Leave-one-out DR ablation.
- *How much does camera-pose jitter augmentation substitute for real camera diversity?*
- *What is the success-rate cost of 50 ms of unmodelled sensor latency in the training data?* (Deliberately inject misalignment — this directly quantifies why your Module 1 sync tool matters.)

Write it up in `RESULTS.md` with a plot, a table, confidence intervals, and an honest limitations section.

**5. Publication**

- GitHub repo, permissive licence, working quickstart that a stranger can run in under 15 minutes.
- Dataset uploaded to Hugging Face in LeRobot format with a complete **dataset card** (provenance, collection method, schema, known defects, intended use, limitations).
- A short technical blog post or README write-up framing the work as *data-centric robotics research*.
- A 3–5 minute screen-recorded demo: the schema, the validator catching a deliberately corrupted episode, the benchmark report.

### Why this specific artefact wins the role

Map it back to the posting line by line:

| Job requirement | Where EDS demonstrates it |
|---|---|
| "Design new datasets and collection methods" | `generator.py` + `randomization_rules.yaml` |
| "Define data schemas, annotations, ground truth" | `dataset_schema.yaml`, `schema.py`, `DATASET_CARD.md` |
| "…and evaluations" | `evaluation/` + the generalisation axes table |
| "Run experiments to understand what data most improves model performance" | `RESULTS.md` |
| "Turn research needs into scalable data pipelines and products" | The whole repo, CI-tested and documented |
| "Strong technical communication" | Dataset card, results write-up, demo video |

---

## Timeline & Milestone Checklist

| Month | Milestone | Core output | Done when |
|---|---|---|---|
| **1** | Spatial math & trajectory foundations | `spatialmath-lite` + `traj-sync` + `SCHEMA.md` | Property tests pass; a misaligned log becomes an aligned HDF5/MCAP |
| **2** | Standardised data formats | `oxe2lerobot` converter + `traj-audit` + storage benchmark | An OXE episode round-trips to LeRobot and loads in `lerobot` |
| **3** | Policy learning | Trained ACT/Diffusion Policy + `traj-viz` + ablation table | Success-rate table with CIs published |
| **4** | Synthetic data at scale | `synth-gen` 5k-episode dataset + DR ablation | Seed → bit-identical dataset verified |
| **5** | 3D perception | `rgbd-fuse` + `depth-qa` + calibration write-up | Two views merge with sub-cm residual |
| **6** | Capstone delivery | `embodied-data-system` published | Stranger-runnable quickstart + `RESULTS.md` + HF dataset |

**Weekly cadence that makes this survivable**

- Mon/Tue/Thu evenings (2h): theory + reading, one paper per week, notes in the repo.
- Wed evening (2h): code lab.
- Sat morning (4–5h): the week's deliverable.
- Sun (1h): write. Commit a progress note. **Public writing every week is non-negotiable** — it compounds into the portfolio and forces clarity.

**Monthly review questions**

1. Did I ship the artefact, or do I have a folder of half-finished notebooks?
2. Can I explain this month's core concept to an engineer with no robotics background?
3. What did I learn that changes how I'd design a dataset?

---

## Appendix A — Hardware & Environment Notes

**Your machine: NVIDIA Quadro P3000 (Pascal, 6 GB).** This shapes several choices, so plan around it rather than discovering it in Month 4:

| Task | Feasible locally? | Plan |
|---|---|---|
| Modules 1, 2, 5 (math, data formats, Open3D) | ✅ Yes, CPU-bound | No changes needed |
| MuJoCo simulation + offscreen rendering | ✅ Yes | MuJoCo runs well on CPU; EGL offscreen rendering works on Pascal |
| Training BC / ACT / small Diffusion Policy | ⚠️ Tight | Use small image resolution (96–128 px), batch 8–16, gradient accumulation. Expect hours, not minutes |
| **NVIDIA Isaac Sim / Isaac Lab** | ❌ No | Requires an RTX GPU (ray tracing cores). **Learn the concepts, use MuJoCo for all hands-on work.** Substitute MJX on a rented GPU if you want parallel envs |
| Fine-tuning OpenVLA (7B) | ❌ No | Rent an A100/H100 for a day (Lambda, RunPod, Vast.ai — roughly $1–3/hr) or use LoRA on a rented 24 GB card. Budget ~$100–200 across the whole 6 months |
| 3D Gaussian Splatting | ⚠️ Small scenes only | 6 GB limits scene size; fine for a tabletop |

**Practical setup**

- Python 3.11, `uv` or `conda` for environments; pin everything, commit the lockfile.
- Install order that avoids pain: `numpy`, `scipy`, then `torch` (CUDA 12.x build), then `mujoco`, then `open3d`, then `lerobot`.
- MuJoCo headless rendering on Linux needs `MUJOCO_GL=egl`; on Windows use `MUJOCO_GL=wgl` or run generation under WSL2.
- **Use WSL2 for the robotics stack.** Much of the ecosystem (ROS 2, MCAP tooling, EGL rendering, LeRobot's video backends) assumes Linux, and fighting Windows-native builds will cost you a week you don't have.
- Track experiments from day one — Weights & Biases free tier, or a plain CSV + matplotlib discipline. Untracked experiments are unpublishable experiments.

**Rented-GPU discipline:** develop and debug at tiny scale locally, then run the full job on rented hardware from a committed config. Never debug on a paid GPU.

---

## Appendix B — Core Reading List

Roughly one paper per week; read the method and the *data* section, skim the rest.

**Foundations**
- Modern Robotics (Lynch & Park) — Chapters 3 (rigid-body motions) and 4–6 (kinematics). The free lectures are enough.
- Zhou et al., *On the Continuity of Rotation Representations in Neural Networks* — why 6D rotation representations exist.

**Datasets & infrastructure**
- *Open X-Embodiment: Robotic Learning Datasets and RT-X Models* — read the data normalisation appendix closely.
- LeRobot documentation & dataset format spec (Hugging Face).
- *DROID: A Large-Scale In-the-Wild Robot Manipulation Dataset* — collection methodology at scale.
- *Universal Manipulation Interface (UMI)* — handheld gripper data collection without robots.
- *Mobile ALOHA* / *ALOHA* — low-cost bimanual teleoperation and its data pipeline.
- *Ego4D* / *Ego-Exo4D* — egocentric video at scale; annotation and benchmark design.

**Policies**
- Ross et al., *DAgger* — the covariate-shift paper.
- Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (ACT).
- Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*.
- Brohan et al., *RT-1* and *RT-2*.
- Kim et al., *OpenVLA: An Open-Source Vision-Language-Action Model*.
- Black et al., *π₀: A Vision-Language-Action Flow Model for General Robot Control*.

**Simulation & transfer**
- Tobin et al., *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World*.
- Peng et al., *Sim-to-Real Transfer of Robotic Control with Dynamics Randomization*.
- MuJoCo documentation — the modelling and computation chapters.

**Perception**
- Qi et al., *PointNet* / *PointNet++*; Qian et al., *PointNeXt*.
- Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*.
- Open3D tutorials on registration and reconstruction.

**Evaluation & data-centric ML**
- *RoboArena* / *SIMPLER* — on the difficulty of evaluating robot policies fairly.
- Anything from the data-centric AI literature on curation-vs-volume; the argument transfers directly.

---

## Appendix C — Interview Preparation Map

Assume the loop is: AI screen → technical deep-dive → research/design discussion → take-home or portfolio review.

**Have crisp answers ready for:**

1. *"Design a dataset for [task X]."* — the core question. Structure your answer: task definition → embodiment & sensors → action space & rate → schema & units → collection protocol → annotation & ground truth → quality gates → evaluation splits → known limitations. Practise this out loud until it's a 5-minute answer.
2. *"How would you evaluate whether more data would help?"* — scaling curve on subsets, ablations by data slice, curation-vs-volume comparison, held-out generalisation axes.
3. *"What data would you collect to improve a policy that keeps dropping objects?"* — failure taxonomy first, then targeted collection: recovery data, in-hand slip events, tactile/force channels, close-up wrist views.
4. *"Egocentric video vs teleop vs UMI vs simulation — when do you use which?"* — cost per hour, action-label fidelity, embodiment gap, scale ceiling. Have the trade-off table memorised.
5. *"How do you handle heterogeneous datasets from 20 labs?"* — this is the OXE problem; answer with your Module 2 converter as the concrete example.
6. *"What's wrong with success rate as a metric?"* — no confidence intervals, no failure taxonomy, criterion sensitivity, no OOD decomposition, evaluator variance.

**Your differentiated pitch, in one paragraph:**

> "I spent eight years building production data infrastructure. Robotics data has all the classic hard problems — asynchronous multimodal streams, clock drift, silent schema violations, heterogeneous sources — but the tooling maturity of data engineering in 2015. I built EDS to bring production discipline to embodied datasets: a versioned schema, CI-enforced validation, reproducible synthetic generation, and an evaluation harness that reports generalisation axes and failure modes rather than a single number. Then I used it to run a controlled study showing [your result]."

**Portfolio checklist before applying**

- [ ] GitHub profile README linking the six artefacts, ordered by relevance
- [ ] EDS repo with a quickstart that works from a clean clone
- [ ] Hugging Face dataset with a complete dataset card
- [ ] One written technical post on a real experimental result
- [ ] CV rewritten with robotics-data vocabulary (schemas, trajectories, embodiments, evaluation) rather than generic backend terms
- [ ] LinkedIn headline naming the target field explicitly
- [ ] A 3–5 minute demo video

---

## A note on realism

Six months of part-time study will not make you a robotics researcher. What it *will* do — if you ship every artefact — is make you a **credible data-infrastructure specialist for robotics**, which is what this specific class of role is actually hiring for. The posting asks for "2–5+ years in robotics, CV, multimodal learning, embodied AI, **or applied ML research**", and explicitly values the ability to "translate research problems into concrete datasets, experiments, and specifications." That translation ability is a systems-engineering skill you already have; this curriculum gives it the domain vocabulary and the evidence.

The single highest-leverage thing in this document is the Module 6 capstone with a real experimental result. If time compresses, cut depth from Modules 3–5 and protect Module 6.
