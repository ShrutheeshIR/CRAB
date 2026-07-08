"""Builds and calls the fully-fused per-robot collision kernel (FK + collision loop +
cost/gradient, all inlined into one -O3 compile -- see crab_codegen.collision_kernel).
This bypasses crab_backend's collision_checker.cpp entirely: there's no longer a
generic loop calling through a function pointer, so there's no indirect-call boundary
left for the compiler to fuse across.
"""

import ctypes
from typing import Optional, Tuple

import numpy as np

from crab_codegen.collision_kernel import build_fused_kernel_source, FK_IS_FREE, FK_COST, FK_COST_GRAD
from crab_codegen.generate_math import generate_jit_sources, load_panda_robot

from .engine import CrabJitEngine


class FusedCollisionKernel:
    def __init__(self, engine: CrabJitEngine, nq: int):
        self._engine = engine
        self._nq = nq

        is_free_ptr = engine.lookup(FK_IS_FREE)
        cost_ptr = engine.lookup(FK_COST)
        grad_ptr = engine.lookup(FK_COST_GRAD)

        is_free_fn_t = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_double))
        cost_fn_t = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.c_double)
        cost_grad_fn_t = ctypes.CFUNCTYPE(
            ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.c_double, ctypes.POINTER(ctypes.c_double)
        )

        self._is_free = is_free_fn_t(is_free_ptr)
        self._cost = cost_fn_t(cost_ptr)
        self._cost_grad = cost_grad_fn_t(grad_ptr)

    def _q_array(self, q):
        if len(q) != self._nq:
            raise ValueError(f"expected q of length {self._nq}, got {len(q)}")
        return (ctypes.c_double * self._nq)(*(float(v) for v in q))

    def is_collision_free(self, q) -> bool:
        return bool(self._is_free(self._q_array(q)))

    def compute_collision_cost(self, q, margin: float = 0.05) -> float:
        return self._cost(self._q_array(q), margin)

    def compute_collision_cost_with_gradient(self, q, margin: float = 0.05) -> Tuple[float, np.ndarray]:
        grad_buf = (ctypes.c_double * self._nq)()
        cost = self._cost_grad(self._q_array(q), margin, grad_buf)
        return cost, np.array(grad_buf, dtype=np.float64)


def build_fused_collision_kernel(robot=None,
                                  engine: Optional[CrabJitEngine] = None,
                                  module_id: Optional[str] = None) -> FusedCollisionKernel:
    robot = robot or load_panda_robot()
    generated = generate_jit_sources(robot)
    source = build_fused_kernel_source(generated, robot)

    engine = engine or CrabJitEngine()
    module_id = module_id or f"crab_fused_{robot.name}_{generated['nq']}dof"
    engine.compile(source, module_id)

    return FusedCollisionKernel(engine, generated["nq"])
