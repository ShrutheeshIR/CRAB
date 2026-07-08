"""Robot fixtures for demos/tests/benchmarks -- not part of the codegen/JIT machinery
itself, just a convenient specific robot to run examples against."""

from crusty_model.robot import Robot
from crusty_io.urdf import load_urdf, urdf_to_robot


def load_panda_robot() -> Robot:
    urdf = load_urdf("robots/panda/panda_spherized.urdf")
    return urdf_to_robot(urdf, ["panda_grasptarget"]).finalize()
