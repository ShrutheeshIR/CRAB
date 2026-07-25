"""Speed harness: wall-clock time per call for JIT'd kernels vs. a plain Python
reference. Exists to put real numbers behind claims JIT_ARCHITECTURE.md/REDESIGN.md
make about JIT and fusion being worthwhile, instead of only arguing it from first
principles.

You run this yourself (see CLAUDE.md): `python -m benchmarks.bench_kernels`
"""

import time

import numpy as np

from crab_codegen.fixtures import load_panda_robot
from crusty_kinematics.derived import forward_kinematics_spheres, task_space_distance, q_to_ee_pose
from crab_jit import build_simple_jit_function, build_fused_collision_kernel

import symforce.symbolic as sf
from .python_timer import time_call


def bench_simple_vs_python(robot) -> None:
    q = np.zeros(robot.nq)

    fk_kernel = build_simple_jit_function(robot, forward_kinematics_spheres, name="forward_kinematics")
    jit_us = time_call(fk_kernel, q, n_calls=10_000)  # this path is fast; more samples

    from crusty_kinematics.fk import forward_kinematics
    def python_reference(q):
        # Naive baseline: crusty_kinematics.fk.forward_kinematics doing its normal
        # SymForce-object-based computation, with none of the JIT'd kernel's fusion
        # or native-code speed.
        primitive_xyzr, _, _ = forward_kinematics(robot, q)
        return np.array(primitive_xyzr, dtype=np.float64).flatten()

    python_us = time_call(python_reference, q, n_calls=200)  # this path is slow; fewer samples

    print(f"forward_kinematics: JIT simple = {jit_us:.2f} us/call, "
          f"plain Python/SymForce eval = {python_us:.2f} us/call, "
          f"speedup = {python_us / jit_us:.1f}x")


def bench_fused(robot) -> None:
    q = np.zeros(robot.nq)
    fused = build_fused_collision_kernel(robot)
    fused_us = time_call(fused.compute_collision_cost_with_gradient, q)

    print(f"fused compute_collision_cost_with_gradient: {fused_us:.2f} us/call")
    print("(no independently-JIT'd, non-fused cost+gradient loop exists to compare "
          "against today -- Path A only produces single value functions, not a "
          "Jacobian-consuming loop, by design; add one here if that comparison "
          "becomes load-bearing.)")

def bench_task_space_distance(robot) -> None:
    q = np.zeros(robot.nq)
    ee_kernel = build_simple_jit_function(robot, q_to_ee_pose, name="q_to_ee")
    goal_pose = ee_kernel(np.full(robot.nq, 0.3))
    dist_kernel = build_simple_jit_function(robot, task_space_distance, name="task_space_distance",
                                            extra_inputs=[("goal_pose", 7)])

    jit_us = time_call(dist_kernel, q, goal_pose, n_calls=10_000)
    from crusty_kinematics.fk import forward_kinematics
    def python_reference(q, goal_pose):
        dist = task_space_distance(robot, q, goal_pose)
        return np.array(dist, dtype=np.float64).flatten()
        # primitive_xyzr, _, link_poses = forward_kinematics(robot, q)
        # ee_pose = link_poses[robot.link_name_to_index[robot.end_effectors[0]]]
        # goal_pose_obj = sf.Pose3.from_storage(list(goal_pose))
        # rel = goal_pose_obj.inverse() * ee_pose
        # return np.array([rel.t.norm()], dtype=np.float64)

    python_us = time_call(python_reference, q, goal_pose, n_calls=200)

    print(f"task_space_distance: JIT simple = {jit_us:.2f} us/call, "
          f"plain Python/SymForce eval = {python_us:.2f} us/call, "
          f"speedup = {python_us / jit_us:.1f}x")


def bench_q_to_ee_pose(robot) -> None:
    q = np.zeros(robot.nq)
    ee_kernel = build_simple_jit_function(robot, q_to_ee_pose, name="q_to_ee")
    jit_us = time_call(ee_kernel, q, n_calls=10_000)

    from crusty_kinematics.fk import forward_kinematics
    def python_reference(q):
        eefk_pose = q_to_ee_pose(robot, q)
        return np.array(eefk_pose, dtype=np.float64).flatten()

    python_us = time_call(python_reference, q, n_calls=200)

    print(f"q_to_ee_pose: JIT simple = {jit_us:.2f} us/call, "
          f"plain Python/SymForce eval = {python_us:.2f} us/call, "
          f"speedup = {python_us / jit_us:.1f}x")

if __name__ == "__main__":
    robot = load_panda_robot()
    # bench_simple_vs_python(robot)
    # bench_task_space_distance(robot)
    # bench_fused(robot)
    bench_q_to_ee_pose(robot)
