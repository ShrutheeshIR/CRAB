"""Canonical per-robot sphere ordering, shared by codegen and the Python-side model
builder so both agree on which flat-array index corresponds to which sphere.

crusty_kinematics.fk.forward_kinematics (used by crab_codegen.generate_math to build
the SymForce FK function) iterates robot.traversal_order and appends each visited
joint's child link's primitives, in that order -- it never visits the root link's own
primitives, since the root link is never anyone's "child" in that loop. Any other
ordering (e.g. iterating robot.links.values() directly) will silently misalign with
the compiled FK output's flat vector. This module is the one place that ordering is
defined, so it can't drift.
"""

from crusty_model.robot import Robot


def robot_sphere_list(robot: Robot) -> list[dict]:
    spheres = []
    for joint_name in robot.traversal_order:
        joint = robot.joints[joint_name]
        link = robot.links[joint.child]
        for idx, primitive in enumerate(link.primitives):
            spheres.append({
                "link_id": link.id,
                "link_name": link.name,
                "index": idx,
                "local_x": float(primitive.pose.t[0]),
                "local_y": float(primitive.pose.t[1]),
                "local_z": float(primitive.pose.t[2]),
                "radius": float(primitive.radius),
            })
    return spheres
