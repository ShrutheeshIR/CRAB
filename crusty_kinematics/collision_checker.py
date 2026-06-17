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

    link_poses: dict[str, sf.Pose3] = {robot.root.name: sf.Pose3()}
    primitive_xyzrs: dict[str, list[list[float]]] = {link.name: [] for link in robot.links.values()}
    bounding_xyzrs: dict[str, list[float]] = {link.name: [] for link in robot.links.values()}

    link_env_collision: dict[str, list[float]] = {}
    link_self_collision: dict[str, list[float]] = {}


    CCC = sp.Function("sphere_environment_collision_checker")
    SSC = sp.Function("sphere_sphere_collision_checker")

    for joint_name in robot.traversal_order:
        joint = robot.joints[joint_name]
        parent_link_pose = link_poses[joint.parent]

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
        link_poses[joint.child] = child_link_pose


        bounding_pose = link_poses[joint.child] * robot.links[joint.child].bounding_primitive.pose
        bounding_xyzrs[joint.child] = [bounding_pose.t.x, bounding_pose.t.y, bounding_pose.t.z, robot.links[joint.child].bounding_primitive.radius]

        link_env_collision[joint.child] = 0
        link_self_collision[joint.child] = 0


        # compute the primitive poses in the child link frame
        for primitive in robot.links[joint.child].primitives:
            primitive_pose = child_link_pose * primitive.pose
            primitive_xyzrs[joint.child].append([primitive_pose.t.x, primitive_pose.t.y, primitive_pose.t.z, primitive.radius])

            link_env_collision[joint.child] += CCC(primitive_pose.t.x, primitive_pose.t.y, primitive_pose.t.z, primitive.radius)

            # go through all the previously computed primitives from the allowed collision pairs and check for collisions
            for other_link_name, other_primitives in primitive_xyzrs.items():
                if (joint.child, other_link_name) in robot.allowed_collision_pairs:
                    for other_primitive in other_primitives:
                        collision = SSC(primitive_pose.t.x, primitive_pose.t.y, primitive_pose.t.z, primitive.radius, other_primitive[0], other_primitive[1], other_primitive[2], other_primitive[3])
                        # collision is a bool, for now just add it up
                        link_self_collision[joint.child] += collision

    return link_env_collision, link_self_collision
