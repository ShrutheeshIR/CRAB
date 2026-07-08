import os
import tempfile

import symforce
symforce.set_symbolic_api("symengine")  # Use symengine for speed, since we don't need SymPy's features

import symforce.symbolic as sf
from symforce.codegen import Codegen, CppConfig

from crusty_model.robot import Robot
from crusty_io.urdf import load_urdf, urdf_to_robot
from crusty_kinematics.fk import forward_kinematics

# Names of the SymForce-generated functions. Codegen.function(name=...) below pins
# these exactly, so the JIT wrapper (crab_codegen/jit_wrapper.py) never has to guess
# what SymForce decided to call them.
FK_FUNC_NAME = "forward_kinematics_generated"
JAC_FUNC_NAME = "forward_kinematics_jacobian_generated"


def load_panda_robot() -> Robot:
    urdf = load_urdf("robots/panda/panda_spherized.urdf")
    return urdf_to_robot(urdf).finalize()


def _build_symbolic_fk(robot: Robot, q):
    """Returns (forward_kinematics_generated, sphere_positions_generated, n_spheres).

    forward_kinematics_generated(q) -> flat [x, y, z, r] per sphere.
    sphere_positions_generated(q)   -> flat [x, y, z] per sphere (used for the Jacobian,
                                        since radii are constant and have a zero Jacobian).
    """
    primitive_xyzr, _ = forward_kinematics(robot, q)

    def forward_kinematics_generated(q):
        flat_xyzr = []
        for x, y, z, r in primitive_xyzr:
            flat_xyzr.extend([x, y, z, r])
        return sf.Matrix(flat_xyzr)

    def sphere_positions_generated(q):
        flat_xyz = []
        for x, y, z, r in primitive_xyzr:
            flat_xyz.extend([x, y, z])
        return sf.Matrix(flat_xyz)

    return forward_kinematics_generated, sphere_positions_generated, len(primitive_xyzr)


def _build_codegen_objects(robot: Robot):
    nq = robot.nq
    q = sf.Matrix(nq, 1).symbolic("q")

    fk_fn, pos_fn, n_spheres = _build_symbolic_fk(robot, q)

    cpp_config = CppConfig()
    # nq is only known at runtime, so q's type can't come from a static annotation on
    # forward_kinematics_generated/sphere_positions_generated (_build_symbolic_fk) --
    # it has to be passed explicitly here.
    input_types = [type(q)]
    codegen_fk = Codegen.function(func=fk_fn, name=FK_FUNC_NAME, input_types=input_types, config=cpp_config)
    codegen_pos = Codegen.function(func=pos_fn, name="sphere_positions_generated", input_types=input_types, config=cpp_config)
    # codegen_jac = codegen_pos.with_linearization(which_args=["q"], name=JAC_FUNC_NAME)

    return codegen_fk, nq, n_spheres


def generate_math(robot: Robot | None = None, output_dir: str | None = None) -> dict:
    """Ahead-of-time codegen: writes FK + Jacobian C++ sources to disk.

    Picked up by the root CMakeLists.txt (HAVE_GENERATED_KINEMATICS path) for a
    statically-compiled crab_backend. See generate_jit_sources() for the runtime-JIT
    equivalent that doesn't require a rebuild of the extension module.
    """
    robot = robot or load_panda_robot()
    codegen_fk, nq, n_spheres = _build_codegen_objects(robot)

    output_dir = output_dir or os.path.abspath("src/generated")
    print(f"Generating C++ files to {output_dir}...")
    codegen_fk.generate_function(output_dir=output_dir)
    # codegen_jac.generate(output_dir=output_dir)
    print("SymForce CodeGen successfully completed!")

    return {"output_dir": output_dir, "nq": nq, "n_spheres": n_spheres}


def _find_generated_header(codegen_result, output_dir: str, func_name: str) -> str:
    """Locates the generated `{func_name}.h` from a Codegen.generate() call.

    SymForce's return value shape has shifted across versions, so this first tries
    the documented `generated_files` attribute and falls back to walking the output
    directory directly.
    """
    generated_files = getattr(codegen_result, "generated_files", None) or []
    for path in generated_files:
        if str(path).endswith(f"{func_name}.h"):
            with open(path, "r") as f:
                return f.read()

    for root, _dirs, files in os.walk(output_dir):
        for fname in files:
            if fname == f"{func_name}.h":
                with open(os.path.join(root, fname), "r") as f:
                    return f.read()

    raise FileNotFoundError(f"Could not locate generated header '{func_name}.h' under {output_dir}")


def generate_jit_sources(robot: Robot | None = None) -> dict:
    """Runs the same SymForce codegen into a scratch directory and returns the raw
    generated C++ source text, for embedding directly into a JIT compilation unit
    via crab_codegen.jit_wrapper + crab_jit. This is the runtime-JIT counterpart to
    generate_math()'s ahead-of-time file output -- nothing is written under src/.
    """
    robot = robot or load_panda_robot()
    codegen_fk, nq, n_spheres = _build_codegen_objects(robot)

    with tempfile.TemporaryDirectory(prefix="crab_jit_codegen_") as tmp_dir:
        fk_dir = os.path.join(tmp_dir, "fk")
        jac_dir = os.path.join(tmp_dir, "jac")

        fk_result = codegen_fk.generate_function(output_dir=fk_dir)
        # jac_result = codegen_jac.generate(output_dir=jac_dir)

        fk_source = _find_generated_header(fk_result, fk_dir, FK_FUNC_NAME)
        # jac_source = _find_generated_header(jac_result, jac_dir, JAC_FUNC_NAME)

    return {
        "nq": nq,
        "n_spheres": n_spheres,
        "fk_source": fk_source,
        "fk_func_name": FK_FUNC_NAME,
        # "jac_source": jac_source,
        # "jac_func_name": JAC_FUNC_NAME,
    }


if __name__ == "__main__":
    generate_math()
