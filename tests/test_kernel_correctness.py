"""Correctness harness: compares each JIT'd kernel's output against a reference
computed by numerically substituting q into the *same* symbolic expression -- no
second implementation to keep in sync, so this catches ABI/shape mistakes (wrong
function name picked up, wrong output size, wrong Jacobian layout) instead of
re-deriving the same bug twice in two places.

You run this yourself (see CLAUDE.md): `python -m tests.test_kernel_correctness`
"""

import numpy as np
import symforce.symbolic as sf

from crab_codegen.generate_math import load_panda_robot, forward_kinematics_spheres, q_to_ee_pose
from crab_jit import build_simple_jit_function, build_fused_collision_kernel


def symbolic_reference(robot, symbolic_fn, q_vals) -> np.ndarray:
    """Evaluates symbolic_fn(robot, q) numerically at q_vals by direct substitution
    into the symbolic expression, instead of re-implementing the math in plain
    Python/NumPy (which would risk encoding the same mistake twice)."""
    nq = robot.nq
    q_sym = sf.Matrix(nq, 1).symbolic("q")
    expr = symbolic_fn(robot, q_sym)
    subs = dict(zip(q_sym, q_vals))
    expr_at_q = expr.subs(subs)
    return np.array(expr_at_q.to_numpy(), dtype=np.float64).flatten()


def check_simple_kernel(robot, symbolic_fn, name: str, n_samples: int = 20, atol: float = 1e-8) -> None:
    kernel = build_simple_jit_function(robot, symbolic_fn, name=name)
    rng = np.random.default_rng(0)
    for _ in range(n_samples):
        q_vals = rng.uniform(-1.0, 1.0, size=robot.nq)
        expected = symbolic_reference(robot, symbolic_fn, q_vals)
        actual = kernel(q_vals)
        if not np.allclose(actual, expected, atol=atol):
            raise AssertionError(f"{name}: JIT output {actual} != reference {expected} for q={q_vals}")
    print(f"{name}: OK ({n_samples} samples)")


def check_fused_kernel(robot, n_samples: int = 20) -> None:
    """The fused kernel's is_collision_free/cost don't reduce to one symbolic
    expression to substitute against (they're a hand-written loop over FK's output),
    so this cross-checks the fused kernel's own exports against each other instead:
    at margin=0, is_collision_free==True should imply cost==0, and vice versa. FK
    itself (which the fused kernel embeds) is already independently checked by
    check_simple_kernel above."""
    kernel = build_fused_collision_kernel(robot)
    rng = np.random.default_rng(1)
    for _ in range(n_samples):
        q_vals = rng.uniform(-1.0, 1.0, size=robot.nq)
        free = kernel.is_collision_free(q_vals)
        cost = kernel.compute_collision_cost(q_vals, margin=0.0)
        if free and cost > 0.0:
            raise AssertionError(f"is_collision_free=True but zero-margin cost={cost} > 0 for q={q_vals}")
        if not free and cost == 0.0:
            raise AssertionError(f"is_collision_free=False but zero-margin cost=0 for q={q_vals}")
    print(f"fused is_collision_free/cost cross-check: OK ({n_samples} samples)")


if __name__ == "__main__":
    robot = load_panda_robot()
    check_simple_kernel(robot, forward_kinematics_spheres, name="forward_kinematics")
    check_simple_kernel(robot, q_to_ee_pose, name="q_to_ee")
    check_fused_kernel(robot)
    print("All correctness checks passed.")
