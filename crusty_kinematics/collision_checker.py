from crusty_model.robot import Robot
import symforce.symbolic as sf
from typing import List, Tuple
import sympy as sp

def collision_checker(robot: Robot, q: sf.Vector7):
    """Compute collision checks for the robot given the joint angles q.
    For each link, compute the pose of each primitives in the link's frame and return a list of lists of poses.
    In addition, each link also has a bounding primtive, which is stored in link.bounding_primitive. 
    The pose of the bounding primitive is also computed and returned as a list of poses.
    """

    num_links = len(robot.links)
    link_poses: list[sf.Pose3] = [sf.Pose3() for _ in range(num_links)]
    primitive_xyzrs: list[list[list[float]]] = [[] for _ in range(num_links)]
    bounding_xyzrs: list[list[float]] = [[] for _ in range(num_links)]

    link_env_collision: list[float] = [0.0] * num_links
    link_self_collision: list[float] = [0.0] * num_links


    CCC = sp.Function("sphere_environment_collision_checker")
    SSC = sp.Function("sphere_sphere_collision_checker")

    for joint_name in robot.traversal_order:
        joint = robot.joints[joint_name]
        parent_link = robot.links[joint.parent]
        child_link = robot.links[joint.child]

        parent_link_pose = link_poses[parent_link.id]

        # compute the joint transform
        if joint.type == "revolute":
            joint_transform = sf.Pose3(R=sf.Rot3.from_angle_axis(q[joint.id], joint.axis), t=sf.V3())
        elif joint.type == "prismatic":
            joint_transform = sf.Pose3(R=sf.Rot3(), t=joint.axis * q[joint.id])
        elif joint.type == "fixed":
            # get the joint transform and apply it
            joint_transform = joint.origin
        else:
            raise ValueError(f"Unsupported joint type: {joint.type}")

        # compute the child link pose
        child_link_pose = parent_link_pose * joint.origin * joint_transform
        link_poses[child_link.id] = child_link_pose


        bounding_primitive = child_link.bounding_primitive
        if bounding_primitive is not None:
            bounding_pose = child_link_pose * bounding_primitive.pose
            bounding_xyzrs[child_link.id] = [bounding_pose.t.x, bounding_pose.t.y, bounding_pose.t.z, bounding_primitive.radius]

        link_env_collision[child_link.id] = 0
        link_self_collision[child_link.id] = 0


        # compute the primitive poses in the child link frame
        for primitive in child_link.primitives:
            primitive_pose = child_link_pose * primitive.pose
            primitive_xyzrs[child_link.id].append([primitive_pose.t.x, primitive_pose.t.y, primitive_pose.t.z, primitive.radius])

            link_env_collision[child_link.id] += CCC(primitive_pose.t.x, primitive_pose.t.y, primitive_pose.t.z, primitive.radius)

            # go through all the previously computed primitives from the allowed collision pairs and check for collisions
            for other_link in robot.links.values():
                other_primitives = primitive_xyzrs[other_link.id]
                if (child_link.name, other_link.name) in robot.allowed_collision_pairs:
                    for other_primitive in other_primitives:
                        collision = SSC(primitive_pose.t.x, primitive_pose.t.y, primitive_pose.t.z, primitive.radius, other_primitive[0], other_primitive[1], other_primitive[2], other_primitive[3])
                        # collision is a bool, for now just add it up
                        link_self_collision[child_link.id] += collision

    return link_env_collision, link_self_collision
