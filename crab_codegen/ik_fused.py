"""IK by hand-fusing everything into one JIT'd kernel through this repo's own
pipeline (crab_codegen.codegen + crab_jit) instead of SymForce's own Optimizer --
same idea as crab_codegen.collision_kernel: generate the smooth part (the pose
residual and its Jacobian) via SymForce, hand-write the branchy/iterative part (the
Levenberg-Marquardt loop and the joint-limit penalty rows, which are trivial 0/1
derivatives, not worth pushing through symbolic differentiation) directly in C++, and
compile the whole thing -- residual, Jacobian, and the solve loop -- as one
translation unit with fixed-size Eigen types the compiler can fully inline.

Companion to ik_symforce_optimizer.py; benchmark both (see benchmarks/bench_kernels.py
for the harness pattern) rather than assume which is faster -- see the conversation
this was written from for a discussion of why they might differ.
"""

from typing import Optional, Tuple

import numpy as np

from crusty_model.robot import Robot
from crusty_kinematics.derived import ik_pose_residual
from crab_codegen.codegen import build_and_generate_function_with_jacobian
from crab_codegen.fixtures import load_panda_robot
from crab_jit.binder import bind_jit_function
from crab_jit.engine import CrabJitEngine

FN_SOLVE = "jit_solve_ik"

IK_FUSED_KERNEL_TEMPLATE = """\
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>

{value_source}
{jac_source}

namespace crab_ik {{

constexpr int kNQ = {nq};
constexpr int kPoseDim = 6;
constexpr int kResidualDim = kPoseDim + 2 * kNQ;
constexpr double kLower[kNQ] = {{{lower}}};
constexpr double kUpper[kNQ] = {{{upper}}};
constexpr int kMaxIters = 30;
constexpr double kLimitWeight = 10.0;

}}  // namespace crab_ik

namespace {{

// Full augmented residual (pose error + joint-limit penalties) and its Jacobian
// w.r.t. q, at the current q. goal_pose is fixed across the whole solve.
void compute_residual_and_jacobian(
    const Eigen::Matrix<double, crab_ik::kNQ, 1>& q,
    const Eigen::Matrix<double, 7, 1>& goal_pose,
    Eigen::Matrix<double, crab_ik::kResidualDim, 1>* residual,
    Eigen::Matrix<double, crab_ik::kResidualDim, crab_ik::kNQ>* jacobian) {{

    Eigen::Matrix<double, crab_ik::kPoseDim, 1> pose_err = sym::{value_func_name}<double>(q, goal_pose);
    Eigen::Matrix<double, crab_ik::kPoseDim, crab_ik::kNQ> pose_jac;
    sym::{jac_func_name}<double>(q, goal_pose, &pose_jac);

    residual->template segment<crab_ik::kPoseDim>(0) = pose_err;
    jacobian->template block<crab_ik::kPoseDim, crab_ik::kNQ>(0, 0) = pose_jac;

    const double sqrt_weight = std::sqrt(crab_ik::kLimitWeight);

    for (int i = 0; i < crab_ik::kNQ; ++i) {{
        const int row_upper = crab_ik::kPoseDim + 2 * i;
        const int row_lower = row_upper + 1;

        jacobian->row(row_upper).setZero();
        jacobian->row(row_lower).setZero();

        const double upper_violation = q(i) - crab_ik::kUpper[i];
        if (upper_violation > 0.0) {{
            (*residual)(row_upper) = sqrt_weight * upper_violation;
            (*jacobian)(row_upper, i) = sqrt_weight;
        }} else {{
            (*residual)(row_upper) = 0.0;
        }}

        const double lower_violation = crab_ik::kLower[i] - q(i);
        if (lower_violation > 0.0) {{
            (*residual)(row_lower) = sqrt_weight * lower_violation;
            (*jacobian)(row_lower, i) = -sqrt_weight;
        }} else {{
            (*residual)(row_lower) = 0.0;
        }}
    }}
}}

}}  // namespace

// inputs[0] = q_init (kNQ doubles), inputs[1] = goal_pose (7 doubles: xyz + quat).
// outputs[0] = solved q (kNQ doubles), outputs[1][0] = final squared residual norm.
extern "C" void {fn_solve}(void** inputs, void** outputs) {{
    Eigen::Matrix<double, crab_ik::kNQ, 1> q(
        Eigen::Map<const Eigen::Matrix<double, crab_ik::kNQ, 1>>(reinterpret_cast<const double*>(inputs[0])));
    Eigen::Matrix<double, 7, 1> goal_pose(
        Eigen::Map<const Eigen::Matrix<double, 7, 1>>(reinterpret_cast<const double*>(inputs[1])));

    double lambda = 1e-3;

    Eigen::Matrix<double, crab_ik::kResidualDim, 1> residual;
    Eigen::Matrix<double, crab_ik::kResidualDim, crab_ik::kNQ> jacobian;
    compute_residual_and_jacobian(q, goal_pose, &residual, &jacobian);
    double cost = residual.squaredNorm();

    for (int iter = 0; iter < crab_ik::kMaxIters; ++iter) {{
        Eigen::Matrix<double, crab_ik::kNQ, crab_ik::kNQ> JtJ = jacobian.transpose() * jacobian;
        Eigen::Matrix<double, crab_ik::kNQ, 1> Jtr = jacobian.transpose() * residual;

        Eigen::Matrix<double, crab_ik::kNQ, crab_ik::kNQ> damping(JtJ.diagonal().asDiagonal());
        Eigen::Matrix<double, crab_ik::kNQ, crab_ik::kNQ> A = JtJ + lambda * damping;
        Eigen::Matrix<double, crab_ik::kNQ, 1> delta = A.ldlt().solve(-Jtr);

        Eigen::Matrix<double, crab_ik::kNQ, 1> q_trial = q + delta;

        Eigen::Matrix<double, crab_ik::kResidualDim, 1> trial_residual;
        Eigen::Matrix<double, crab_ik::kResidualDim, crab_ik::kNQ> trial_jacobian;
        compute_residual_and_jacobian(q_trial, goal_pose, &trial_residual, &trial_jacobian);
        double trial_cost = trial_residual.squaredNorm();

        if (trial_cost < cost) {{
            q = q_trial;
            residual = trial_residual;
            jacobian = trial_jacobian;
            cost = trial_cost;
            lambda = std::max(lambda * 0.5, 1e-9);
            if (delta.norm() < 1e-10) {{
                break;
            }}
        }} else {{
            lambda = std::min(lambda * 2.0, 1e9);
        }}
    }}

    Eigen::Map<Eigen::Matrix<double, crab_ik::kNQ, 1>>(reinterpret_cast<double*>(outputs[0])) = q;
    reinterpret_cast<double*>(outputs[1])[0] = cost;
}}
"""


def _joint_limits_by_index(robot: Robot) -> list[tuple[float, float]]:
    """(lower, upper) per q-index. Unlimited (e.g. continuous) or unspecified joints
    get a wide-open range so the penalty never activates for them."""
    limits = [(-1e3, 1e3)] * robot.nq
    for joint in robot.joints.values():
        if joint.is_actuated and joint.limits is not None:
            limits[joint.id] = joint.limits
    return limits


def build_ik_kernel_source(robot: Robot, fn_solve: str = FN_SOLVE) -> dict:
    generated = build_and_generate_function_with_jacobian(
        robot, ik_pose_residual, "ik_pose_residual", extra_inputs=[("goal_pose", 7)]
    )
    nq = generated["nq"]
    limits = _joint_limits_by_index(robot)

    source = IK_FUSED_KERNEL_TEMPLATE.format(
        value_source=generated["value_source"],
        jac_source=generated["jac_source"],
        value_func_name=generated["value_func_name"],
        jac_func_name=generated["jac_func_name"],
        nq=nq,
        lower=", ".join(repr(lo) for lo, _ in limits),
        upper=", ".join(repr(hi) for _, hi in limits),
        fn_solve=fn_solve,
    )
    return {"source": source, "nq": nq}


class FusedIKSolver:
    def __init__(self, engine: CrabJitEngine, nq: int):
        self._solve = bind_jit_function(engine, FN_SOLVE, input_sizes=[nq, 7], output_sizes=[nq, 1])

    def solve(self, q_init, goal_pose_flat) -> Tuple[np.ndarray, float]:
        q_solution, cost = self._solve(q_init, goal_pose_flat)
        return q_solution, float(cost[0])


def build_fused_ik_solver(robot=None,
                           engine: Optional[CrabJitEngine] = None,
                           module_id: Optional[str] = None) -> FusedIKSolver:
    robot = robot or load_panda_robot()
    generated = build_ik_kernel_source(robot)

    engine = engine or CrabJitEngine()
    module_id = module_id or f"crab_ik_fused_{robot.name}_{generated['nq']}dof"
    engine.compile(generated["source"], module_id)

    return FusedIKSolver(engine, generated["nq"])


if __name__ == "__main__":
    from crab_jit import build_simple_jit_function
    from crusty_kinematics.derived import q_to_ee_pose

    robot = load_panda_robot()

    ee_kernel = build_simple_jit_function(robot, q_to_ee_pose, name="q_to_ee")
    goal_pose = ee_kernel(np.full(robot.nq, 0.3))

    solver = build_fused_ik_solver(robot)
    solved_q, final_cost = solver.solve(np.zeros(robot.nq), goal_pose)
    print("solved q:", solved_q, "final cost:", final_cost)
