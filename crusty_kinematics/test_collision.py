from .collision_checker import collision_checker
from crusty_io.urdf import load_urdf, urdf_to_robot
import symforce.symbolic as sf

def test_collision_checker():
    urdf = load_urdf("robots/panda/panda_spherized.urdf")
    robot = urdf_to_robot(urdf)
    robot = robot.finalize()

    q = sf.Vector7([0, 0, 0, 0, 0, 0, 0])
    print("Testing collision checker with q =", q)
    link_env_collision, link_self_collision = collision_checker(robot, q)
    print("Link environment collisions:")
    for i, val in enumerate(link_env_collision):
        print(f"  Link {i} ({list(robot.links.values())[i].name}): {val}")
    print("Link self collisions:")
    for i, val in enumerate(link_self_collision):
        print(f"  Link {i} ({list(robot.links.values())[i].name}): {val}")


if __name__ == "__main__":
    test_collision_checker()