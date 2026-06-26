from crusty_model.robot import Robot
import symforce.symbolic as sf
from typing import List, Tuple
import sympy as sp

def forward_kinematics(robot: Robot, q: sf.Vector7):
    """Compute the forward kinematics of the robot given the joint angles q.
    For each link, compute the pose of each primitives in the link's frame and return a list of lists of poses.
    In addition, each link also has a bounding primtive, which is stored in link.bounding_primitive. 
    The pose of the bounding primitive is also computed and returned as a list of poses.
    """

    link_poses: dict[str, sf.Pose3] = {robot.root.name: sf.Pose3()}
    primitive_xyzr: list[list[float]] = []
    # Compute bounding sphere poses
    bounding_xyzr: list[list[float]] = []

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

        # compute the primitive poses in the child link frame
        for primitive in robot.links[joint.child].primitives:
            primitive_pose = child_link_pose * primitive.pose
            primitive_xyzr.append([primitive_pose.t.x, primitive_pose.t.y, primitive_pose.t.z, primitive.radius])

    for link in robot.links.values():
        if link.bounding_primitive:
            bounding_pose = link_poses[link.name] * link.bounding_primitive.pose
            bounding_xyzr.append([bounding_pose.t.x, bounding_pose.t.y, bounding_pose.t.z, link.bounding_primitive.radius])

    return primitive_xyzr, bounding_xyzr #, link_poses