"""Builds a crab_backend.RobotSpherized (the nanobind-bound struct consumed by
is_collision_free / compute_collision_cost* / forward_kinematics) from a
crusty_model.robot.Robot. This is the missing link between the Python robot model
and the hand-written C++ collision checker in src/collision_checker.cpp."""

import crab_backend

from crusty_model.robot import Robot
from crab_codegen.sphere_order import robot_sphere_list


def build_robot_spherized(robot: Robot) -> "crab_backend.RobotSpherized":
    if not robot.allowed_collision_pairs:
        robot.compute_allowed_collision_pairs()

    spherized = crab_backend.RobotSpherized()
    spherized.name = robot.name
    spherized.nq = robot.nq

    spheres = []
    # Order must match crusty_kinematics.fk.forward_kinematics exactly (see
    # crab_codegen.sphere_order), since this is the order the compiled/JIT'd FK
    # kernel's flat output vector is indexed by.
    for s in robot_sphere_list(robot):
        sphere = crab_backend.Sphere()
        sphere.link_id = s["link_id"]
        sphere.index = s["index"]
        sphere.local_x = s["local_x"]
        sphere.local_y = s["local_y"]
        sphere.local_z = s["local_z"]
        sphere.radius = s["radius"]
        spheres.append(sphere)
    spherized.spheres = spheres

    spherized.link_names = [(link.name, str(link.id)) for link in robot.links.values()]

    name_to_id = {link.name: link.id for link in robot.links.values()}
    spherized.active_collision_pairs = [
        (name_to_id[a], name_to_id[b]) for a, b in robot.allowed_collision_pairs
    ]

    return spherized
