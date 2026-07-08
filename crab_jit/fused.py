"""Builds and calls Path B's fully-fused per-robot collision kernel (FK + collision
loop + cost/gradient, all inlined into one -O3 compile -- see
crab_codegen.collision_kernel). FusedCollisionKernel is a thin, purely-for-ergonomics
wrapper over three crab_jit.binder.JitFunctions -- there's no bespoke ctypes signature
code here; binding is identical to Path A's (see REDESIGN.md).
"""

from typing import Optional, Tuple

import numpy as np

from crab_codegen.collision_kernel import build_fused_kernel_source, FK_IS_FREE, FK_COST, FK_COST_GRAD
from crab_codegen.generate_math import load_panda_robot

from .binder import bind_jit_function
from .engine import CrabJitEngine


class FusedCollisionKernel:
    def __init__(self, engine: CrabJitEngine, nq: int):
        self._is_free = bind_jit_function(engine, FK_IS_FREE, input_sizes=[nq], output_sizes=[1])
        self._cost = bind_jit_function(engine, FK_COST, input_sizes=[nq, 1], output_sizes=[1])
        self._cost_grad = bind_jit_function(engine, FK_COST_GRAD, input_sizes=[nq, 1], output_sizes=[1, nq])

    def is_collision_free(self, q) -> bool:
        return bool(self._is_free(q)[0])

    def compute_collision_cost(self, q, margin: float = 0.05) -> float:
        return float(self._cost(q, margin)[0])

    def compute_collision_cost_with_gradient(self, q, margin: float = 0.05) -> Tuple[float, np.ndarray]:
        cost, grad = self._cost_grad(q, margin)
        return float(cost[0]), grad


def build_fused_collision_kernel(robot=None,
                                  engine: Optional[CrabJitEngine] = None,
                                  module_id: Optional[str] = None) -> FusedCollisionKernel:
    robot = robot or load_panda_robot()
    generated = build_fused_kernel_source(robot)

    engine = engine or CrabJitEngine()
    module_id = module_id or f"crab_fused_{robot.name}_{generated['nq']}dof"
    engine.compile(generated["source"], module_id)

    return FusedCollisionKernel(engine, generated["nq"])
