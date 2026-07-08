"""End-to-end examples for CRAB's JIT backend (see REDESIGN.md for the two-path design).

fused_demo() is Path B: FK + the collision loop + cost/gradient are all JIT-compiled
into one -O3 translation unit per robot (crab_codegen.collision_kernel), so there's no
call boundary between FK and the collision math for the compiler to fuse across.

fk_demo() and q_to_ee_demo() are Path A (crab_jit.simple): a single q -> flat vector
symbolic function, JIT'd and bound generically -- no template/builder/export needed
per function, just the math plus one call to build_simple_jit_function.

task_space_distance_demo() is Path A with an extra runtime input beyond q (a goal
pose), showing task_space_distance built on top of q_to_ee_pose's own convention.
"""

import numpy as np

from crab_codegen.generate_math import load_panda_robot, q_to_ee_pose, forward_kinematics_spheres, task_space_distance
from crab_jit import build_fused_collision_kernel, build_simple_jit_function


def fused_demo():
    robot = load_panda_robot()
    kernel = build_fused_collision_kernel(robot)

    q = np.zeros(robot.nq)
    print("collision free:", kernel.is_collision_free(q))
    print("cost:", kernel.compute_collision_cost(q))
    cost, grad = kernel.compute_collision_cost_with_gradient(q)
    print("cost:", cost, "grad:", grad)


def fk_demo():
    """Smallest possible example of Path A: JITs just forward kinematics (sphere
    positions), nothing else."""
    robot = load_panda_robot()
    kernel = build_simple_jit_function(robot, forward_kinematics_spheres, name="forward_kinematics")

    q = np.zeros(robot.nq)
    print("spheres_world (flat xyzr per sphere):", kernel(q))


def q_to_ee_demo():
    robot = load_panda_robot()
    kernel = build_simple_jit_function(robot, q_to_ee_pose, name="q_to_ee")

    q = np.zeros(robot.nq)
    print("ee pose (xyz + quat):", kernel(q))


def task_space_distance_demo():
    robot = load_panda_robot()

    # Get a real goal pose from q_to_ee_pose itself, at some other configuration --
    # the two kernels agree on the flat-pose convention (xyz + quat) by construction,
    # since task_space_distance's goal_pose_flat input uses the same layout
    # q_to_ee_pose's output does.
    ee_kernel = build_simple_jit_function(robot, q_to_ee_pose, name="q_to_ee")
    goal_pose = ee_kernel(np.full(robot.nq, 0.3))

    dist_kernel = build_simple_jit_function(robot, task_space_distance, name="task_space_distance",
                                             extra_inputs=[("goal_pose", 7)])

    q = np.zeros(robot.nq)
    print("task space distance to goal:", dist_kernel(q, goal_pose))


if __name__ == "__main__":
    # fused_demo()
    q_to_ee_demo()
    fk_demo()
    task_space_distance_demo()
