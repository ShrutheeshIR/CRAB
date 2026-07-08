# CRAB JIT backend: redesign plan

**Status: implemented.** This was the planning doc; see `JIT_ARCHITECTURE.md` for the
current-state description of the system this produced. Kept here as the record of why
it's shaped this way and exactly what got deleted and why.

Goal: get to "write one Python function, everything else falls into place," with
exactly two authoring paths (simple, generic-bound; custom/fused, hand-written but
still generic-bound) and one binding mechanism shared by both.

## Why: what's wrong with the current shape

The current system has **two parallel binding mechanisms** doing the same job
differently:
- `crab_backend` (nanobind, ahead-of-time, `src/`): a hand-written C++ struct
  (`RobotSpherized`) and hand-written functions, each accepting an optional raw
  `fk_fn_ptr`/`jac_fn_ptr`. Every new JIT'd function that wants to reach this path
  needs matching C++ added to `collision_checker.hpp/.cpp` + `bindings.cpp`, plus a
  CMake rebuild.
- `crab_jit` (ctypes, JIT'd): `SimpleKernel` for the generic `void**` case,
  `FusedCollisionKernel` with hand-typed `ctypes.CFUNCTYPE`s for the fused case. Two
  different Python-side binding styles for two different C ABIs.

That's three or four places to touch for one new function (as the `q_to_ee` work just
demonstrated), and two build systems (`build/` for `crab_backend`, `crusty_compilations/build/`
for the JIT). None of this duplication is buying anything anymore — collapse to one
binding mechanism, one ABI convention, one build.

## Part 1: Restructuring

### One ABI, one binder, for everything

Standardize on the `void** inputs, void** outputs` convention (already used by
`jit_wrapper.py`/`SimpleKernel`) as the *only* boundary between JIT'd C++ and Python —
for both authoring paths below. Concretely this means the fused kernel's exported
functions (`jit_is_collision_free` etc.) switch from their current bespoke
`double*`/`double`/`int` signatures to `void** in, void** out` too, so
`FusedCollisionKernel`'s hand-written `ctypes.CFUNCTYPE`s per function go away —
replaced by one generic binder class used everywhere:

- Accepts `q` as `list`/`tuple`/`np.ndarray` (Python-side convenience only — it
  always becomes a flat `ctypes` double buffer before crossing into C++).
- Takes a declared list of output sizes (one output today; the interface should allow
  more, e.g. `(cost, gradient)` as two outputs instead of one function returning cost
  and a second one returning gradient) and returns `np.ndarray` (or a tuple of them).
- Does the `void*` array boxing once, generically, regardless of which path produced
  the underlying function.

Eigen only ever appears inside generated/hand-written C++ (`Eigen::Map` over the raw
`double*` a `void*` points at) — Python never needs to know Eigen exists. "List,
tuple, numpy array to Eigen, that's it" is satisfied entirely at this one boundary.

### Path A — simple functions (generalize `crab_jit/simple.py`)

Already mostly built. Keep:
- `generate_math.build_and_generate_function(robot, symbolic_fn, name)` — codegen,
  read back SymForce's real emitted symbol name, infer output shape from the
  returned `sf.Matrix`.
- `jit_wrapper.generate_value_wrapper` — the one `void**` wrapper template.
- The generic binder (successor to `SimpleKernel`).

Usage stays: write `def my_fn(robot, q) -> sf.Matrix: ...`, call
`build_simple_jit_function(robot, my_fn, name="my_fn")`, call the result like a
function.

### Path B — custom/fused kernels (generalize `collision_kernel.py`)

For cases where fusing multiple generated functions plus hand-written control flow
into one `-O3` unit matters (the collision loop is the existing example). Keep the
mechanism — a Python function that assembles a C++ *source template*, pasting in one
or more generated headers plus hand-written logic — but:
- The exported entry points switch to the shared `void** in, void** out` convention
  (see above), so they bind through the *same* generic binder as Path A instead of a
  bespoke class.
- Everything specific to the collision loop specifically (sphere radii/pairs as
  `constexpr`, the early-exit loop, the cubic penalty cost) stays exactly as-is — only
  the function *signatures* at the boundary change, not the fusion technique itself.

### Compiler layer (`crusty_compilations`) — unchanged

`ClangCompiler` / `JitSession` / `DiskObjectCache` / `crab_jit_capi.cc` /
`crab_jit/engine.py` (`CrabJitEngine`) already do their one job (compile a C++ string
at `-O3`, cache by content hash, hand back symbol addresses) and aren't the source of
per-function toil. No changes planned here beyond what's already landed (the Eigen
include-path fix, the content-hash cache-key fix).

### End state

```python
def my_fn(robot, q) -> sf.Matrix:
    ...                                    # the only part that should take real effort

kernel = jit(robot, my_fn)                 # Path A
kernel([0.1, 0.2, ...])                    # list/tuple/np.ndarray in, np.ndarray out
```
```python
kernel = jit_custom(robot, MY_TEMPLATE, generated=[fk, jac])   # Path B
kernel(q)                                                       # same call shape
```

One build (`crusty_compilations/build/`), one ABI, one binder class, one mental model.

## Part 2: Delete / throw away

- **`jit_cpp_compiler/` (entire directory)** — `jit_engine.cc`, `jit_wrapper.cc`,
  `jit_wrapper.py`, `jit_python_example.py`. Never compiled, imports a `.so` that's
  never built (`libjit_engine.so`), references a `crusty_kinematics.collision_checker`
  module that doesn't exist. Fully superseded by `crusty_compilations` + `crab_jit`.
- **`crusty_compilations/compile_panda.py`** — same dead `crusty_kinematics.collision_checker`
  import; superseded by `crab_codegen/generate_math.py` + `jit_demo.py`.
- **`src/` (entire directory)** — `bindings.cpp`, `collision_checker.cpp/.hpp`,
  `collision_helpers.hpp`, `generated_placeholder.cpp`. This is the whole nanobind
  ahead-of-time `crab_backend` module; it disappears once bindings are generic/JIT-native.
  Two things here are real logic, not dead code, and need to be *ported*, not just
  deleted:
  - `collision_helpers.hpp`'s `sphere_environment_collision_checker` (ground-plane
    check) — port into the fused-kernel C++ template (it's conceptually already
    duplicated there as inline `z < radius` logic; consolidate on one copy).
  - `collision_helpers.hpp`'s `sphere_sphere_collision_checker` is *already* dead
    code today — `collision_checker.cpp`'s loop reimplements the same check inline
    instead of calling it. Confirmed via the exported header's own body never being
    referenced. Fine to just drop.
- **Root `CMakeLists.txt` + root `build/`** — existed only to build `crab_backend`.
  With nanobind gone, nothing is left for it to build; `crusty_compilations`'s CMake
  project becomes the only build in the repo. This also retires the "two build
  directories, why do I need both" confusion from earlier, and the latent
  `HAVE_GENERATED_KINEMATICS` bug (the macro `collision_checker.cpp` checks for but
  root `CMakeLists.txt` never actually `#define`s) becomes moot.
- **`crab_jit/build.py`** (`build_robot_jit_functions`, `build_robot_fk_jit`) — exists
  only to hand raw FK/Jacobian pointers to `crab_backend.forward_kinematics(...)`.
  No consumer once `crab_backend` is gone.
- **`crab_codegen/robot_spherized.py`** — builds a `crab_backend.RobotSpherized`;
  meaningless without `crab_backend`. Its actual payload (sphere radii, resolved
  collision-pair indices) is already independently recomputed inside
  `collision_kernel.py`'s `build_fused_kernel_source`, so nothing of value is lost.
- **`crab_codegen/jit_wrapper.py`'s FK/Jacobian-bundle-specific pieces** —
  `build_jit_source`, `build_fk_only_source`, `DEFAULT_FK_WRAPPER_NAME`,
  `DEFAULT_JAC_WRAPPER_NAME`, and `generate_jacobian_wrapper`'s sphere-major reshape
  (`jac.template block<3, nq>(s*3, 0)` copied into a separate flat buffer). This
  reshape exists solely to match `collision_checker.cpp`'s expected raw-buffer layout;
  once nothing consumes a standalone Jacobian in that specific shape (the fused kernel
  computes and uses the Jacobian internally, never exposing it raw), it has no
  purpose. `generate_value_wrapper` is the one piece of this file that survives.
- **`crab_jit/fused.py`'s `FusedCollisionKernel` class** — the bespoke
  `double*`/`double`/`int`-signature `ctypes.CFUNCTYPE`s. Replaced by the one generic
  binder once the fused kernel's exports switch to the shared `void**` ABI.
  `build_fused_collision_kernel`'s *source-assembly* logic (in `collision_kernel.py`)
  stays; only its Python-side binding changes.
- **`generate_math.py`'s AOT path** (`generate_math()`, writing to `src/generated`)
  — only existed to feed the now-deleted `crab_backend` build.
- **Not touched**: `robots/panda/...` (URDF), `crusty_model`, `crusty_kinematics`,
  `crusty_io` — the actual robot-description and math layer, orthogonal to this
  redesign.

## Part 3: Benchmarking and testing

Nothing like this exists yet. Two separate concerns, two separate scripts:

### Correctness harness

For each kernel (simple or fused), evaluate the JIT'd function against a reference
computed by evaluating the *same* symbolic expression numerically in Python (SymForce
expressions support numeric substitution directly — no second implementation to keep
in sync, no risk of "the test encodes the same bug as the code"), over a batch of
random `q` samples within joint limits. Catches ABI/shape mistakes (the exact class of
bug hit repeatedly this session: wrong function name, wrong output size, wrong
Jacobian layout) automatically instead of only when someone happens to run a demo.

Proposed home: `tests/test_kernel_correctness.py` at the repo root (new `tests/`
directory — nothing currently plays this role).

### Speed benchmark

Goal: get real numbers behind the claims this doc and `JIT_ARCHITECTURE.md` make
about JIT and fusion being worthwhile, instead of asserting it. Measure, per kernel,
median/p95 wall-clock time per call across many repeated calls (steady-state, after
warming the disk cache) for:
1. A pure-Python/NumPy reference implementation of the same computation.
2. A Path-A simple JIT'd kernel.
3. For collision checking specifically: Path-A kernels called individually in sequence
   (FK, then a Python-side loop) vs. the Path-B fused kernel — this is the number that
   actually validates "fusing FK into the collision loop is worth the complexity,"
   which today is only argued from first principles, never measured.

Proposed home: `benchmarks/bench_kernels.py`, printing a small table (nothing
elaborate — this doesn't need a framework, just `time.perf_counter` and a loop).

Both are scripts *you* run (per `CLAUDE.md` — builds/runs/tests stay on your side);
this doc only proposes where they live and what they check.
