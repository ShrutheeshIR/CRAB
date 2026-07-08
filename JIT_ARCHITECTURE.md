# How CRAB's JIT backend fits together

Reflects the post-redesign architecture (see `REDESIGN.md` for the plan and rationale
behind this shape, and the delete list of everything this replaced). No prior
JIT/LLVM knowledge assumed.

## The problem we're solving

Checking a robot config `q` (forward kinematics, collision checking, end-effector
pose, ...) is pure math derived from the URDF -- the same formula every time for a
given robot, but re-deriving it symbolically on every call is slow. So we generate
custom, robot-specific machine code once per robot (via SymForce) and JIT-compile it
at `-O3` (via LLVM/Clang), instead of either a generic interpreted implementation or a
CMake rebuild every time you touch the math.

## Where the math lives vs. where the machinery lives

These are deliberately separate, so "add a new kinematic quantity" never means
touching codegen/JIT code:

- **`crusty_kinematics/derived.py`** — the actual math: plain
  `(robot, q, *extra_inputs) -> sf.Matrix` functions (`forward_kinematics_spheres`,
  `q_to_ee_pose`, `task_space_distance`). No codegen, no JIT, no `crab_*` imports.
  Anything here is just a symbolic function you could evaluate directly in Python.
- **`crab_codegen/codegen.py`** — the generic machinery that turns *any* such function
  into generated C++ (`build_and_generate_function`, `build_and_generate_function_with_jacobian`).
  Knows nothing about FK, poses, or collision checking specifically.
- **`crab_codegen/fixtures.py`** — `load_panda_robot()`, a demo/test convenience, not
  part of the machinery either.

One cross-cutting detail this split surfaces: SymForce's symbolic backend
(`symforce.set_symbolic_api(...)`) is process-global, one-time, order-sensitive
configuration -- it must run before *anything* does `import symforce.symbolic`,
regardless of which module happens to be imported first by a given entry point. It's
centralized in `crusty_model/__init__.py` (rather than wherever used to import first
"by accident"), since `crusty_model.robot.Robot` is a dependency of essentially every
module here that touches `symforce.symbolic`.

## The one ABI, the one binder

Every JIT'd C++ function in this codebase, regardless of how it was written, exports
`void fn(void** inputs, void** outputs)` and gets called from Python through exactly
one class: `crab_jit.binder.JitFunction`.

- **Inputs** are `list`/`tuple`/`np.ndarray`/scalars on the Python side; `JitFunction`
  flattens each into its own `ctypes` double buffer and boxes it into the `inputs`
  array positionally.
- **Outputs** always come back as `np.ndarray` -- a single array if there's one
  output, a tuple if there's more than one (e.g. `(cost, gradient)`).
- **Eigen only ever appears inside the generated/hand-written C++**, on the far side
  of `Eigen::Map`-ing the raw `double*` a `void*` points at. Python never needs to
  know Eigen exists.

This single convention is what lets both authoring paths below share one binder
instead of each needing bespoke marshalling code.

## Path A — simple functions (`crab_jit/simple.py`)

For a `(q, *extra_inputs) -> flat vector` function with no special ABI needs (the
common case for one-off kinematic quantities):

```python
def q_to_ee_pose(robot, q) -> sf.Matrix:
    ...                                     # the actual math
    return sf.Matrix(...)

kernel = build_simple_jit_function(robot, q_to_ee_pose, name="q_to_ee")
kernel(q)                                    # -> np.ndarray
```

`q` isn't special-cased beyond being first -- any number of additional runtime inputs
are supported via `extra_inputs=[(name, size), ...]`, e.g. a function that also takes
a goal pose:

```python
def task_space_distance(robot, q, goal_pose_flat) -> sf.Matrix:
    ...                                     # can reuse q_to_ee_pose's own math
    return sf.Matrix([...])

kernel = build_simple_jit_function(robot, task_space_distance, name="task_space_distance",
                                    extra_inputs=[("goal_pose", 7)])
kernel(q, goal_pose)                         # positional, in extra_inputs' order after q
```

What happens under the hood, all automatic:
1. `crab_codegen.codegen.build_and_generate_function(robot, symbolic_fn, name, extra_inputs)`
   — builds a symbolic `q` (and one symbolic placeholder per declared extra input),
   calls `symbolic_fn(robot, q, *extras)`, runs `Codegen.function`, generates into a
   scratch dir, and reads the *real* emitted C++ symbol name back out of the
   generated source (SymForce reformats whatever name you pass to its own naming
   convention -- e.g. `"forward_kinematics_generated"` becomes
   `ForwardKinematicsGenerated` -- so this can't be assumed, only read back). Output
   size (`n_out`) is read from `symbolic_fn`'s own returned `sf.Matrix` shape, not
   asked for separately. `_make_traced_func` builds the callable `Codegen.function`
   introspects for parameter names, since it has to expose one named parameter per
   input while still returning an expression graph already built directly from the
   real `q`/extra symbols (that graph can depend on arbitrary robot-specific Python
   logic -- FK's joint traversal -- that isn't itself symbolic tracing through a
   generic wrapper function).
2. `crab_codegen.jit_wrapper.generate_value_wrapper` — wraps the generated function in
   the `void**` ABI.
3. `crab_jit.CrabJitEngine.compile(source, module_id)` — JIT-compiles the whole thing
   at `-O3` via `crusty_compilations` (below).
4. `crab_jit.binder.bind_jit_function(...)` — looks up the exported symbol and returns
   a `JitFunction`.

Adding a new one-off function costs exactly: write `symbolic_fn(robot, q)`, call
`build_simple_jit_function`. No new C++ template, no new builder function, no new
export to remember.

## Path B — custom/fused kernels (`crab_codegen/collision_kernel.py`, `crab_jit/fused.py`)

For cases where fusing multiple generated functions plus hand-written control flow
into one `-O3` translation unit matters -- collision checking is the example: FK,
its Jacobian, the self/environment collision-pair loop, and the cost/gradient math
all live in one hand-written C++ template
(`crab_codegen.collision_kernel.FUSED_KERNEL_TEMPLATE`), so the compiler can inline
across what would otherwise be call boundaries. Sphere radii and which sphere-index
pairs to check are known once the robot is loaded (they don't depend on `q`), so
they're baked in as `constexpr` literals rather than looked up at runtime.

The only thing that changed from a plain "one hand-written kernel" design: its
exported functions (`jit_is_collision_free`, `jit_compute_collision_cost`,
`jit_compute_collision_cost_with_gradient`) use the *same* `void**` ABI as Path A, so
`crab_jit.fused.FusedCollisionKernel` is just three `JitFunction`s wearing a
convenience class (`is_collision_free(q)`, `compute_collision_cost(q, margin)`,
`compute_collision_cost_with_gradient(q, margin)` methods) -- no bespoke
`ctypes.CFUNCTYPE` per function.

`crab_codegen.codegen.build_and_generate_function_with_jacobian(robot,
symbolic_fn, name)` is Path B's codegen entry point: like Path A's
`build_and_generate_function`, but also runs `Codegen.function(...).with_jacobian(...)`
and returns both the value and Jacobian source/symbol names, for pasting into a
hand-written template. Path A never needs this -- nothing exposes a raw Jacobian
externally anymore; it's only ever consumed internally by a fused kernel's own loop.

## The compiler layer (`crusty_compilations/`) — unchanged by this redesign

The generic, robot-agnostic engine that takes a C++ source string and hands back
callable machine code:
- `crab::jit::ClangCompiler` (`include/jit/compiler.hh`, `src/jit/compiler.cc`) --
  invokes Clang's compiler internals directly on a source string at `-O3`.
- `crab::jit::JitSession` (`session.hh`/`.cc`) -- wraps LLVM's ORC JIT, loads compiled
  code into the process, resolves symbol lookups by name.
- `crab::jit::DiskObjectCache` (`object_cache.hh`/`.cc`) -- caches compiled objects on
  disk (`~/.cache/crab/`), keyed by a hash of source + compiler flags (so editing a
  kernel's source always busts the cache -- see `crab_jit_capi.cc`).
- `examples/crab_jit_capi.cc` -- the plain C API (`crab_jit_create/compile/lookup/destroy`)
  exposed as `libcrab_jit_capi.so`, called from Python via `ctypes`
  (`crab_jit/engine.py`'s `CrabJitEngine`). Also passes `-isystem <Eigen include dir>`
  (`CRAB_EIGEN_INCLUDE_DIR`, baked in at CMake time) since the Clang driver invocation
  here has no notion of this process's own build flags otherwise.

## One build, not two

Before this redesign there were two independent CMake projects: the root
`CMakeLists.txt` building a nanobind extension (`crab_backend`), and
`crusty_compilations/CMakeLists.txt` building the JIT. `crab_backend` and everything
that only existed to feed it (`src/`, `crab_jit/build.py`, `crab_codegen/robot_spherized.py`,
...) are gone -- see `REDESIGN.md`'s delete list. `crusty_compilations/build/` is now
the only build directory in the repo:
```
cmake -S crusty_compilations -B crusty_compilations/build
cmake --build crusty_compilations/build --target crab_jit_capi
```

## Walkthrough: `fused_demo()`

1. `crab_codegen.fixtures.load_panda_robot()` loads the URDF into a `Robot`.
2. `crab_codegen.collision_kernel.build_fused_kernel_source(robot)` generates FK +
   Jacobian, resolves this robot's sphere radii/collision pairs, and fills in the
   fused C++ template.
3. `crab_jit.fused.build_fused_collision_kernel(robot)` compiles that source at `-O3`
   via `CrabJitEngine` (cached to disk by content hash) and binds the three exported
   functions through `JitFunction`.
4. `kernel.is_collision_free(q)` / `.compute_collision_cost(q)` /
   `.compute_collision_cost_with_gradient(q)` call straight into machine code compiled
   specifically for this robot, moments ago, with FK and the collision loop fused
   into one optimized function.

## Testing and benchmarking

- **`tests/test_kernel_correctness.py`** -- checks each Path A kernel's output against
  a reference computed by numerically substituting `q` into the *same* symbolic
  expression (no second implementation to drift out of sync), plus a cross-check
  between the fused kernel's own `is_collision_free`/`compute_collision_cost` exports.
- **`benchmarks/bench_kernels.py`** -- median per-call wall-clock time for a JIT'd
  simple kernel vs. the plain Python/SymForce reference it was generated from.
- Both are scripts you run yourself (`python -m tests.test_kernel_correctness`,
  `python -m benchmarks.bench_kernels`) -- per `CLAUDE.md`, builds/runs/tests stay on
  your side.

## Adding a new function, end to end

```python
def my_fn(robot, q) -> sf.Matrix:
    ...                                      # the only part that should take effort

kernel = build_simple_jit_function(robot, my_fn, name="my_fn")
kernel(q)
```

That's it for Path A. Reach for Path B (a hand-written template like
`collision_kernel.py`'s) only when you specifically need multiple generated functions
fused into one call with hand-written control flow between them.
