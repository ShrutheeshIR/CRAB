"""Generic path for one-off `q -> flat vector` symbolic functions: codegen, extern "C"
wrapping, JIT compiling, and ctypes binding in a single call, instead of hand-writing a
new template/builder/export for every new function (see the "Simplifying the 'new
function' workflow" section of JIT_ARCHITECTURE.md).

Not used for FK, the Jacobian, or the fused collision kernel -- those have bespoke
shapes/ABIs for genuinely load-bearing reasons (fixed argument layouts, fusion across
call boundaries). This is for the common case that isn't any of that, like q_to_ee.

Usage:

    def q_to_ee_pose(robot, q) -> sf.Matrix:
        ...  # the actual math -- the only part that should take real effort
        return sf.Matrix(...)

    kernel = build_simple_jit_function(robot, q_to_ee_pose, name="q_to_ee")
    kernel(q)  # -> np.ndarray, shape inferred from q_to_ee_pose's own return shape
"""

import ctypes
from typing import Callable, Optional

import numpy as np

from crab_codegen.generate_math import build_and_generate_function, load_panda_robot
from crab_codegen.jit_wrapper import generate_value_wrapper

from .engine import CrabJitEngine


class SimpleKernel:
    """Callable wrapper around one JIT'd `void fn(void** inputs, void** outputs)`
    function that takes a single flat `q` vector and returns a single flat vector of
    known size. Handles the void**-ABI box/call/unbox marshalling generically, once,
    instead of once per function."""

    def __init__(self, engine: CrabJitEngine, wrapper_name: str, nq: int, n_out: int):
        self._engine = engine
        self._nq = nq
        self._n_out = n_out
        fn_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p))
        self._fn = fn_t(engine.lookup(wrapper_name))

    def __call__(self, q) -> np.ndarray:
        if len(q) != self._nq:
            raise ValueError(f"expected q of length {self._nq}, got {len(q)}")
        q_buf = (ctypes.c_double * self._nq)(*(float(v) for v in q))
        out_buf = (ctypes.c_double * self._n_out)()
        inputs = (ctypes.c_void_p * 1)(ctypes.cast(q_buf, ctypes.c_void_p))
        outputs = (ctypes.c_void_p * 1)(ctypes.cast(out_buf, ctypes.c_void_p))
        self._fn(inputs, outputs)
        return np.array(out_buf, dtype=np.float64)


def build_simple_jit_function(robot=None,
                               symbolic_fn: Callable = None,
                               name: str = None,
                               engine: Optional[CrabJitEngine] = None,
                               module_id: Optional[str] = None) -> SimpleKernel:
    """symbolic_fn(robot, q) -> sf.Matrix (a flat column vector). Everything else --
    Codegen.function, reading back SymForce's real emitted symbol name, the extern "C"
    wrapper, compiling at -O3, and the ctypes binding -- is automatic."""
    robot = robot or load_panda_robot()
    generated = build_and_generate_function(robot, symbolic_fn, name)

    wrapper_name = f"jit_{name}"
    source = "\n".join([
        "#include <Eigen/Dense>",
        "",
        generated["source"],
        "",
        generate_value_wrapper(generated["func_name"], generated["nq"], generated["n_out"], wrapper_name),
    ])

    engine = engine or CrabJitEngine()
    module_id = module_id or f"crab_jit_simple_{name}_{robot.name}_{generated['nq']}dof"
    engine.compile(source, module_id)

    return SimpleKernel(engine, wrapper_name, generated["nq"], generated["n_out"])
