from .urdf import load_urdf, urdf_to_robot

def test_load_urdf():
    urdf = load_urdf("robots/panda/panda_spherized.urdf")
    robot = urdf_to_robot(urdf)
    assert robot.nq == 7

    robot = robot.finalize()
    assert robot._frozen

if __name__ == "__main__":
    test_load_urdf()