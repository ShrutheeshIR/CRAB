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

    # Computation above stays dict-keyed (readable, matches link/joint names). Only the
    # return value needs to be dict-free: SymForce's symbolic tree walking (StorageOps)
    # can't recurse into a dict (or a raw string, so name-tagged tuples don't work
    # either), so re-index into a plain list ordered by robot.link_name_to_index right
    # before returning.
    link_poses_by_index: list[sf.Pose3] = [sf.Pose3()] * len(robot.links)
    for link_name, pose in link_poses.items():
        link_poses_by_index[robot.link_name_to_index[link_name]] = pose

    return primitive_xyzr, bounding_xyzr, link_poses_by_index


def q_to_eeposes(robot: Robot, q: sf.Vector7, eef_link_names: List[str]):
    """
    Compute fk of EEFs
    If eef_link_names is empty, compute fk of all eefs
    """
    # if empty, then do it for all robot.end_effectors
    if not eef_link_names:
        eef_link_names = robot.end_effectors
    eef_poses = []
    _, _, link_poses = forward_kinematics(robot, q)
    for eef_link_name in eef_link_names:
        eef_poses.append(link_poses[robot.link_name_to_index[eef_link_name]])
    return eef_poses
