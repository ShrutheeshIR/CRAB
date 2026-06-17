from .collision_checker import collision_checker
from crusty_io.urdf import load_urdf, urdf_to_robot
import symforce.symbolic as sf

def test_collision_checker():
    urdf = load_urdf("robots/panda/panda_spherized.urdf")
    robot = urdf_to_robot(urdf)
    robot = robot.finalize()

    q = sf.Vector7([0, 0, 0, 0, 0, 0, 0])
    print("Testing collision checker with q =", q)
    cccs = collision_checker(robot, q)
    print("Primitive poses and radii (x, y, z, r):")
    for primitive_xyzr in primitive_xyzrs:
        print(f"  {primitive_xyzr}")
    print("Bounding sphere poses and radii (x, y, z, r):")
    for bounding_xyzr in bounding_xyzrs:
        print(f"  {bounding_xyzr}")
    for link_name, link_pose in link_poses.items():
        print(f"Link {link_name} pose: {link_pose}")

    # for i, link in enumerate(robot.links.values()):
    #     print(f"Link {link.name} pose: {link_poses[i]}")

    #     for j, primitive_pose in enumerate(primitive_poses[i]):
    #         print(f"  Primitive {j} pose: {primitive_pose}")


if __name__ == "__main__":
    test_collision_checker()