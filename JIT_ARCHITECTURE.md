# How CRAB's JIT backend fits together

This explains what `src/`, `crab_codegen/`, `crab_jit/`, and `crusty_compilations/` each
do, and how they connect, using the Panda-arm collision-checking example
(`crab_codegen/jit_demo.py`) as the walkthrough. No prior JIT/LLVM knowledge assumed.

## The problem we're solving

Checking a robot config `q` for collisions means:
1. Forward kinematics (FK): turn joint angles `q` into world-space sphere centers.
2. Loop over sphere pairs and check if any two overlap.
3. (For gradient-based planners) also compute the Jacobian of each sphere position
   w.r.t. `q`, so you can get a gradient of a collision "cost" for optimization.

Step 1 is pure math derived from the URDF (link lengths, joint axes) — the same
formula every time for a given robot. Step 2/3 depend on which spheres exist and
which pairs are worth checking, which is also fixed once you know the robot.

None of this depends on the *user's Python code* — it only depends on *which robot*
you loaded. So instead of writing one generic "FK for any robot" function that
re-derives everything at every call (slow — lots of matrix multiplies for joints that
don't even move the sphere in question), we generate custom, robot-specific machine
code once per robot and then call that.

## What "JIT" means here

JIT = "just-in-time" compilation: instead of compiling C++ into machine code ahead of
time (like a normal C++ build), we compile a string of C++ source code *while the
Python program is running*, and get back a raw function pointer we can call
immediately. LLVM is the compiler backend doing this; Clang is the C++ frontend that
understands C++ syntax and hands LLVM the intermediate representation to optimize
and turn into machine code.

Why bother, instead of just building a `.so` with CMake beforehand? Because the C++
source itself is *generated per-robot* (see next section) — it has the robot's
sphere radii and collision pairs baked in as literal numbers. If you swap robots at
runtime (e.g. in an interactive session), JIT compiling on the spot avoids re-running
CMake by hand every time.

## The four pieces

### 1. `crab_codegen/` — turn robot math into C++ source text

This is pure Python, no compiling happens here. It uses SymForce (a symbolic math
library) to write out the forward-kinematics and Jacobian formulas for *this specific
robot* as C++ functions, as plain text.

- `generate_math.py` — loads the robot (`load_panda_robot()`), builds symbolic
  expressions for FK and its Jacobian using SymForce, and can either:
  - write them to `src/generated/*.h` on disk (`generate_math()`, used for an
    ahead-of-time, normal-compiled build), or
  - return the generated C++ as an in-memory string (`generate_jit_sources()`, used
    by the JIT path — nothing touches disk under `src/`).
- `sphere_order.py` — the FK code walks the robot's joints in a specific order and
  emits one `[x, y, z, radius]` per collision sphere. Anything downstream that
  refers to "sphere #3" needs to agree on what sphere #3 is. This file is the single
  source of truth for that ordering, used both by the JIT kernel generator and by
  `robot_spherized.py` (see below), so they can't drift apart.
- `collision_kernel.py` — the interesting part. It takes the generated FK/Jacobian
  C++ text and wraps it in one more C++ file that also contains the collision-pair
  loop and the cost/gradient math, with this robot's sphere radii and which
  sphere-pairs to check against each other baked in as compile-time constants
  (`constexpr`). The result is a single, self-contained C++ **translation unit**
  (one `.cc` file's worth of text) exporting three C functions:
  `jit_is_collision_free`, `jit_compute_collision_cost`,
  `jit_compute_collision_cost_with_gradient`. Everything for this robot lives in one
  file, so the optimizer can inline FK straight into the collision loop instead of
  going through a function call.
- `jit_wrapper.py` / `robot_spherized.py` — an older, more modular alternative
  (`pointer_demo()` in `jit_demo.py`) where only FK/Jacobian are JIT'd standalone,
  and the collision loop lives separately in `src/collision_checker.cpp`, calling the
  JIT'd FK/Jacobian through function pointers. Kept for cases that need the generic
  `crab_backend.RobotSpherized` API rather than one fused kernel. `collision_kernel.py`
  is the recommended path — see "why fuse everything" below.

**Output of this stage:** a Python string containing valid C++ source code, specific
to Panda, that nothing has compiled yet.

### 2. `crusty_compilations/` — the actual JIT compiler, robot-agnostic

This is the generic engine that takes a C++ source string and gives you back
callable machine code. It knows nothing about robots, FK, or collisions — it would
happily compile `int add(int a, int b) { return a + b; }`.

- `include/jit/compiler.hh`, `src/jit/compiler.cc` (`crab::jit::ClangCompiler`) —
  invokes Clang's compiler internals directly on a source string (`-O3`, full
  optimizations) and produces an in-memory LLVM module — think of this as "the
  optimizing compiler."
- `include/jit/session.hh`, `src/jit/session.cc` (`crab::jit::JitSession`) — wraps
  LLVM's ORC JIT engine, which takes a compiled module and actually loads it into
  the current process's memory so its functions become callable, and resolves symbol
  lookups by name (`lookup("jit_is_collision_free")` gives you a raw address).
- `include/jit/object_cache.hh`, `src/jit/object_cache.cc`
  (`crab::jit::DiskObjectCache`) — compiling C++ from scratch (especially with `-O3`)
  takes real time (parsing Eigen headers alone is slow). This caches the compiled
  object code on disk (`~/.cache/crab/`), keyed by a hash of the source + compiler
  flags, so re-running the same robot kernel a second time skips compilation
  entirely and just loads the cached bytes.
- `examples/crab_jit_capi.cc` — a plain C API (`crab_jit_create/compile/lookup/destroy`)
  wrapping all of the above, so it can be exposed as a shared library
  (`libcrab_jit_capi.so`) and called from Python without needing pybind/nanobind
  bindings — just `ctypes`.
- `examples/example_jit.py`, `examples/example_jit.cc` — the original toy example
  proving the C API works, unrelated to robots.

**Output of this stage:** a `.so` you build once with CMake. It's a general-purpose
"compile C++ string, get function pointer" service.

### 3. `crab_jit/` — Python glue between the two

- `engine.py` (`CrabJitEngine`) — `ctypes` wrapper around `libcrab_jit_capi.so`.
  `engine.compile(source, module_id)` hands your C++ string to the JIT; `engine.lookup(name)`
  gets back a raw integer address for an exported function.
- `fused.py` (`FusedCollisionKernel`, `build_fused_collision_kernel`) — the
  recommended path. Calls `crab_codegen.collision_kernel.build_fused_kernel_source()`
  to get the one-file-per-robot C++ text, JITs it via `CrabJitEngine`, looks up the
  three exported functions, and wraps each raw pointer in a `ctypes.CFUNCTYPE` so you
  can call `kernel.is_collision_free(q)` from Python like a normal function.
- `build.py` (`build_robot_jit_functions`) — same idea but for the older
  FK/Jacobian-only split (paired with `pointer_demo()`).

**Output of this stage:** a Python object (`FusedCollisionKernel`) whose methods are
backed by machine code compiled seconds ago, specific to your robot, at `-O3`.

### 4. `src/` — the hand-written, normally-compiled C++/Python extension

This is a regular pybind/nanobind extension module (`crab_backend`), built once via
CMake like any C++ project — not JIT'd.

- `collision_checker.hpp/.cpp` — `RobotSpherized`, `Sphere`, and the collision
  functions (`is_collision_free`, `compute_collision_cost`,
  `compute_collision_cost_with_gradient`). Each optionally accepts a raw
  `fk_fn_ptr`/`jac_fn_ptr` (an integer address) — if given, it calls through that
  pointer instead of a fallback statically-compiled FK. This is what the older
  `pointer_demo()` path uses: JIT compiles FK/Jacobian, and this file's loop calls
  them via function pointer.
- `bindings.cpp` — exposes all of the above to Python as the `crab_backend` module.

The fused path (`crab_jit/fused.py`) bypasses this file entirely for the collision
loop — the loop itself got moved into the JIT'd kernel (see below).

## Why fuse FK + the collision loop together (`collision_kernel.py`)

The original design was: JIT-compile FK and the Jacobian only, and call them through
a function pointer from the fixed, separately-compiled loop in
`collision_checker.cpp`. That works, but a function-pointer call is an "optimization
barrier" — the compiler can't see across it, so it can't inline FK's math directly
into the loop that uses it, even at `-O3`.

`collision_kernel.py` instead generates *one* C++ file containing FK, the Jacobian,
and the collision-pair loop and cost/gradient math together, with the robot's sphere
radii and which pairs to check baked in as literal constants (they don't depend on
`q`, only on which robot you loaded). Compiling that whole thing at `-O3` in one shot
lets the compiler inline and fuse across what used to be a call boundary.

## Putting it together: what happens when you run `fused_demo()`

1. `crab_codegen.generate_math.load_panda_robot()` loads the URDF into a `Robot` object.
2. `generate_jit_sources(robot)` runs SymForce, returns FK/Jacobian C++ as strings.
3. `crab_codegen.collision_kernel.build_fused_kernel_source(...)` wraps those strings
   plus the collision loop into one C++ file, with Panda's sphere radii and collision
   pairs as `constexpr` literals.
4. `crab_jit.fused.build_fused_collision_kernel(robot)` sends that file's text to
   `CrabJitEngine`, which hands it to `crusty_compilations`' Clang+LLVM pipeline:
   compiled at `-O3`, cached to disk by content hash, loaded into the process.
5. You get back a `FusedCollisionKernel` — call `kernel.is_collision_free(q)` and
   you're calling machine code that was compiled specifically for Panda, moments ago,
   with FK and the collision loop fused into one optimized function.

## Simplifying the "new function" workflow

Right now, adding one new JIT'd function means hand-editing three separate places:

1. **`crab_codegen`** — write (or reuse) a wrapper template that knows the exact C++
   calling convention of the SymForce-generated function (return-by-value? jacobian
   as an out-pointer? how many outputs, what shape?) and emit `extern "C"` glue for it.
2. **`crab_jit` (`build.py` / `fused.py`)** — concatenate the right sources, invent a
   `module_id` string, call `engine.compile()`, `engine.lookup()` each exported symbol
   by its hardcoded name, and hand-write a matching `ctypes.CFUNCTYPE`/argtypes that
   must exactly match the C++ signature or you get a silent ABI mismatch (wrong
   argument count/type reads garbage, doesn't error).
3. **`crab_jit/__init__.py`** — remember to export the new builder function.

This is manual and duplicated: the shape information (how many doubles in, how many
out) effectively has to be written down twice — once in the C++ wrapper generator,
once in the matching Python `ctypes` signature — with nothing checking the two agree.

### Why JAX's `jit` doesn't feel like this

`jax.jit` looks automatic because JAX never hand-writes a calling convention per
function. A few things make that possible, and they map fairly directly onto pieces
we already have:

- **One universal calling convention.** JAX traces any Python function into a
  `jaxpr` over flattened arrays; the compiled artifact always takes "some flat arrays
  in, some flat arrays out." It never generates a bespoke C ABI per function. We
  already have this too — it's the `void** inputs, void** outputs` convention in
  `crab_codegen/jit_wrapper.py`. The fused kernel (`collision_kernel.py`) *doesn't*
  use it (it uses bespoke `double*`/`double`/`int` signatures instead), which is why
  `fused.py` needs hand-written `ctypes.CFUNCTYPE`s per exported symbol while the
  pointer-based path in principle wouldn't.
- **Shape/dtype metadata drives both sides.** JAX derives the compiled signature and
  the Python-side dispatch from the same `ShapeDtypeStruct`/pytree metadata — there's
  one source of truth, not two hand-maintained copies. Here, the analogous metadata
  already exists (SymForce's `Codegen` object knows its inputs/outputs; our own dicts
  like `generated["nq"]`/`generated["n_spheres"]` carry the same information) — it's
  just not being used to *generate* the wrapper and the `ctypes` binding together.
- **Caching keyed by identity + signature, not a hand-picked string.** `jax.jit`
  caches compiled versions of a function keyed off the function itself plus the
  argument shapes/dtypes — you never invent a cache key by hand. We already moved
  most of the way there for the *disk* cache (content hash of source + flags, see the
  `crab_jit_capi.cc` fix above); the `module_id` prefix and the Python-level "which
  builder function do I call for this kernel" step are still manual.

### Concrete directions

- **Generate the wrapper from the `Codegen` object's own metadata** instead of a
  separate hand-written template per calling-convention shape
  (`FK_WRAPPER_TEMPLATE` vs `JACOBIAN_WRAPPER_TEMPLATE`). SymForce's `Codegen`
  already knows its inputs and outputs; one generic wrapper generator that walks
  that metadata and emits the `void**` marshalling loop would cover any future
  function (arbitrary in/out count and shape) without a new template file.
- **Derive the `ctypes` signature from the same metadata used to generate the C++
  wrapper**, instead of writing a matching `CFUNCTYPE` by hand in `build.py`/`fused.py`.
  If both come from one shape description, they can't drift apart — today's failure
  mode (C++ signature and Python `CFUNCTYPE` silently disagreeing) becomes structurally
  impossible instead of something to remember to keep in sync.
- **A small kernel registry instead of manual plumbing per function.** A declarative
  list of "kernel specs" (name, the symbolic-function builder, robot) that one generic
  loader walks — build source, compile, look up symbols, bind — turns "add a new JIT'd
  function" into "add one entry to a table" instead of edits in three files, and
  removes the manual `crab_jit/__init__.py` export step (the registry itself is the
  thing other code queries).
- **A `crab_jit.jit`-style decorator/memoizing entry point**, analogous to `jax.jit`:
  wrap the symbolic-function builder once; the first call for a given robot triggers
  codegen + compile + bind automatically and caches the result (keyed by robot identity
  + function identity, mirroring how the disk cache is now keyed by content hash);
  later calls just dispatch. This would remove the need for demo code to explicitly
  call a `build_*_jit` function at all.
- **Worth keeping in mind:** the universal `void**` ABI is what makes all of the above
  easy to automate, but it costs an extra layer of pointer indirection/boxing per call
  compared to the fused kernel's bespoke native signatures. That's the same tradeoff
  JAX makes (a uniform dispatch convention) versus what `collision_kernel.py` does today
  (bespoke signatures, hand-fused for a specific hot loop). Automating the wrapper/`ctypes`
  generation doesn't have to mean giving up the fused, bespoke-ABI kernel for the
  performance-critical path — it mainly helps the simpler one-function-in/one-function-out
  cases like FK-only, where the boilerplate is pure overhead today.

## Building/running

See `CLAUDE.md` — Claude won't run build/lint/test commands in this repo; you run
those yourself. Roughly:

```
cmake -S crusty_compilations -B crusty_compilations/build
cmake --build crusty_compilations/build --target crab_jit_capi
python -m crab_codegen.jit_demo
```
