from .fk import forward_kinematics
from crusty_io.urdf import load_urdf, urdf_to_robot
import symforce.symbolic as sf

def test_forward_kinematics():
    urdf = load_urdf("robots/panda/panda_spherized.urdf")
    robot = urdf_to_robot(urdf)
    robot = robot.finalize()

    q = sf.Vector7([0, 0, 0, 0, 0, 0, 0])
    print("Testing forward kinematics with q =", q)
    primitive_xyzrs, bounding_xyzrs, link_poses = forward_kinematics(robot, q)
    print("Primitive poses and radii (x, y, z, r):")
    for primitive_xyzr in primitive_xyzrs:
        print(f"  {primitive_xyzr}")
    print("Bounding sphere poses and radii (x, y, z, r):")
    for bounding_xyzr in bounding_xyzrs:
        print(f"  {bounding_xyzr}")
    for i, link in enumerate(robot.links.values()):
        print(f"Link {link.name} pose: {link_poses[i]}")

        for j, primitive_pose in enumerate(link.primitives):
            print(f"  Primitive {j} pose: {primitive_pose.pose}")

    # fetch end effector 
    eef_poses = q_to_eeposes(robot, q, [])

    print("End effector poses:")
    for eef_pose in eef_poses:
        print(f"  {eef_pose}")


    # for i, link in enumerate(robot.links.values()):
    #     print(f"Link {link.name} pose: {link_poses[i]}")

    #     for j, primitive_pose in enumerate(primitive_poses[i]):
    #         print(f"  Primitive {j} pose: {primitive_pose}")


if __name__ == "__main__":
    test_forward_kinematics()