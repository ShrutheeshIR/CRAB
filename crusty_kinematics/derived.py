"""Symbolic quantities derived from forward kinematics -- plain
`(robot, q, *extra_inputs) -> sf.Matrix` math, no codegen/JIT-specific code. Meant to
be handed as symbolic_fn to crab_jit.simple.build_simple_jit_function (Path A) or
pasted into a hand-fused kernel (Path B, crab_codegen.collision_kernel).
"""

import symforce.symbolic as sf

from crusty_model.robot import Robot
from crusty_kinematics.fk import forward_kinematics, q_to_eeposes


def forward_kinematics_spheres(robot: Robot, q) -> sf.Matrix:
    """q -> flat [x, y, z, radius] per collision sphere, in
    crab_codegen.sphere_order's canonical ordering (same ordering
    forward_kinematics already produces)."""
    primitive_xyzr, _bounding_xyzr, _link_poses = forward_kinematics(robot, q)
    flat_xyzr = []
    for x, y, z, r in primitive_xyzr:
        flat_xyzr.extend([x, y, z, r])
    return sf.Matrix(flat_xyzr)


def q_to_ee_pose(robot: Robot, q) -> sf.Matrix:
    """q -> flat 7-vector Pose3.to_storage() of robot.end_effectors[0]."""
    ee_pose = q_to_eeposes(robot, q, [robot.end_effectors[0]])[0]
    return sf.Matrix(ee_pose.to_storage())


def task_space_distance(robot: Robot, q, goal_pose_flat) -> sf.Matrix:
    """(q, goal_pose) -> [translation distance] between the end effector and a goal
    pose. goal_pose_flat is a flat 7-vector, Pose3.to_storage()'s layout -- same
    convention q_to_ee_pose returns, so a goal can come directly from a previous
    q_to_ee_pose call."""
    ee_pose = q_to_eeposes(robot, q, [robot.end_effectors[0]])[0]
    goal_pose = sf.Pose3.from_storage(list(goal_pose_flat))
    rel = goal_pose.inverse() * ee_pose
    return sf.Matrix([rel.t.norm()])


def ik_pose_residual(robot: Robot, q, goal_pose_flat) -> sf.Matrix:
    """(q, goal_pose) -> 6D tangent-space pose error (3 translation + 3 rotation)
    between the end effector and a goal pose -- the residual for least-squares IK
    (see crab_codegen.ik_fused, crab_codegen.ik_symforce_optimizer).

    Unlike task_space_distance, this returns the residual *vector*, not its norm: a
    least-squares solver needs the actual error vector to form proper normal
    equations (J^T J, J^T r) -- a norm is non-differentiable at zero and throws away
    direction entirely.
    """
    ee_pose = q_to_eeposes(robot, q, [robot.end_effectors[0]])[0]
    goal_pose = sf.Pose3.from_storage(list(goal_pose_flat))
    tangent_error = (goal_pose.inverse() * ee_pose).to_tangent(epsilon=1e-9)
    return sf.Matrix(list(tangent_error))
