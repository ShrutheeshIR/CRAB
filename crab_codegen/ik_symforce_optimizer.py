"""IK via SymForce's own built-in nonlinear least-squares optimizer (Values/Factor/
Optimizer). The Levenberg-Marquardt loop runs inside SymForce's compiled C++ runtime
(the `sym` library), not a Python loop, even though you drive it from Python by
calling .optimize() once. Companion to ik_fused.py, which solves the same problem by
hand-fusing everything into this repo's own JIT pipeline instead -- benchmark both
(see benchmarks/bench_kernels.py for the harness pattern) rather than assume which is
faster.

NOTE: written from best recollection of SymForce's opt API (Factor's constructor
args, Optimizer.Params' field names, how the solved Values comes back on the result
object). Treat exact names here as "probably right, verify against your installed
symforce version" rather than certain -- the residual formulation and overall
approach are the part that matters.
"""

import numpy as np
import symforce.symbolic as sf
from symforce.values import Values
from symforce.opt.factor import Factor
from symforce.opt.optimizer import Optimizer

from crusty_model.robot import Robot, _joint_limits_by_index
from crusty_kinematics.fk import q_to_eeposes
from crab_codegen.fixtures import load_panda_robot

_EPSILON = 1e-9




def build_ik_residual(robot: Robot, joint_limits: list[tuple[float, float]],joint_limit_weight: float):
    """Returns a plain Python (q, goal_pose, epsilon) -> sf.Matrix function. Factor
    traces and differentiates this itself -- no manual Codegen/Jacobian call needed,
    unlike the Path A/B machinery elsewhere in this repo.

    Residual layout: [translation error (3), rotation error (3, tangent space),
    joint-limit soft penalties (2 * nq)] -- same idea as
    crusty_kinematics.derived.ik_pose_residual, with joint limits folded in as extra
    residual rows so the same Gauss-Newton machinery handles both.
    """
    nq = robot.nq
    ee_link_name = robot.end_effectors[0]

    # Factor needs a concrete-size type annotation to know q's shape -- a bare sf.M
    # carries no size info (same reason Codegen.function elsewhere in this repo needs
    # input_types=[type(q)] passed explicitly instead of relying on annotations).
    # sf.Matrix(nq, 1) returns an *instance* of the shaped class (e.g. a zero-filled
    # Matrix71), not the class itself -- type() on that instance gives the actual
    # class, same pattern codegen.py already uses (type(q) after q = sf.Matrix(nq,
    # 1).symbolic("q")).
    QType = type(sf.Matrix(nq, 1))

    def residual(q: QType, goal_pose: sf.Pose3, epsilon: sf.Scalar) -> sf.M:
        ee_pose = q_to_eeposes(robot, q, [ee_link_name])[0]
        pose_error = (goal_pose.inverse() * ee_pose).to_tangent(epsilon=epsilon)

        limit_residuals = []
        for i in range(nq):
            lower, upper = joint_limits[i]
            if lower < upper:  # only apply penalty if joint has limits
                limit_residuals.append(sf.Max(0, q[i] - upper) * joint_limit_weight)
                limit_residuals.append(sf.Max(0, lower - q[i]) * joint_limit_weight)

        return sf.M(list(pose_error) + limit_residuals)

    return residual


class IKSolver:
    """Wraps an already-built Optimizer -- construction (tracing the residual,
    generating/compiling its evaluator) happened once in build_ik_solver(); .solve()
    only builds a fresh numeric Values and calls .optimize() on the existing
    Optimizer, which is the part actually worth timing."""

    def __init__(self, robot: Robot, optimizer: Optimizer):
        self._robot = robot
        self._optimizer = optimizer

    def solve(self, goal_pose_flat, q_init=None) -> np.ndarray:
        nq = self._robot.nq
        q_init = np.zeros(nq) if q_init is None else np.asarray(q_init, dtype=np.float64)

        values = Values(
            q=sf.Matrix(q_init.tolist()),
            goal_pose=sf.Pose3.from_storage(list(goal_pose_flat)),
            epsilon=_EPSILON,
        )
        result = self._optimizer.optimize(values)
        solved_q = result.optimized_values["q"]
        return np.array(solved_q, dtype=np.float64).flatten()


def build_ik_solver(robot: Robot, joint_limit_weight: float = 10.0) -> IKSolver:
    """Builds the Factor + Optimizer once. This is the expensive part -- SymForce
    symbolically traces build_ik_residual's function and generates (and likely
    compiles) the C++ evaluator for the residual and its Jacobian here -- so build
    once and reuse the returned IKSolver for many .solve() calls, the same "build
    once, call many" shape as crab_jit.simple.build_simple_jit_function elsewhere in
    this repo. Calling solve_ik() (below) in a loop/benchmark instead times this
    setup cost on every iteration, not just the actual solve -- that's why a
    wall-clock benchmark around solve_ik() can report ~100x more time per call than
    SymForce's own internal TicToc profiling of just the Optimize() step: TicToc only
    instruments code already running inside the compiled optimizer, not the
    Python-side Factor/Optimizer construction that happens before it exists.
    """
    joint_limits = _joint_limits_by_index(robot)
    residual_fn = build_ik_residual(robot, joint_limits, joint_limit_weight)
    factor = Factor(residual=residual_fn, keys=["q", "goal_pose", "epsilon"])

    optimizer = Optimizer(
        factors=[factor],
        optimized_keys=["q"],
        params=Optimizer.Params(verbose=False, debug_stats=False),
    )
    return IKSolver(robot, optimizer)


def solve_ik(robot: Robot, goal_pose_flat, q_init=None, joint_limit_weight: float = 10.0) -> np.ndarray:
    """Convenience one-shot wrapper: builds a solver and solves once. Don't call this
    in a loop/benchmark -- call build_ik_solver() once and reuse .solve() instead, or
    you're timing Factor/Optimizer construction on every iteration (see
    build_ik_solver's docstring)."""
    return build_ik_solver(robot, joint_limit_weight).solve(goal_pose_flat, q_init=q_init)

if __name__ == "__main__":
    from crab_jit import build_simple_jit_function
    from crusty_kinematics.derived import q_to_ee_pose

    robot = load_panda_robot()

    # A reachable-ish goal: FK at some other configuration, via the Path A kernel
    # already built for exactly this in jit_demo.py's task_space_distance_demo.
    ee_kernel = build_simple_jit_function(robot, q_to_ee_pose, name="q_to_ee")
    goal_pose = ee_kernel(np.full(robot.nq, 0.3))

    solved_q = solve_ik(robot, goal_pose)
    print("solved q:", solved_q)
