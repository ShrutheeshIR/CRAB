import inspect
import os
import re
import tempfile

import symforce
symforce.set_symbolic_api("symengine")  # Use symengine for speed, since we don't need SymPy's features

import symforce.symbolic as sf
from symforce.codegen import Codegen, CppConfig

from crusty_model.robot import Robot
from crusty_io.urdf import load_urdf, urdf_to_robot
from crusty_kinematics.fk import forward_kinematics


def load_panda_robot() -> Robot:
    urdf = load_urdf("robots/panda/panda_spherized.urdf")
    return urdf_to_robot(urdf).finalize()


def forward_kinematics_spheres(robot: Robot, q) -> sf.Matrix:
    """q -> flat [x, y, z, radius] per collision sphere, in
    crab_codegen.sphere_order's canonical ordering (same ordering
    crusty_kinematics.fk.forward_kinematics already produces). Path-A-compatible
    (see crab_jit.simple.build_simple_jit_function) and also embedded directly into
    the fused collision kernel (Path B, crab_codegen.collision_kernel)."""
    primitive_xyzr, _bounding_xyzr, _link_poses = forward_kinematics(robot, q)
    flat_xyzr = []
    for x, y, z, r in primitive_xyzr:
        flat_xyzr.extend([x, y, z, r])
    return sf.Matrix(flat_xyzr)


def q_to_ee_pose(robot: Robot, q) -> sf.Matrix:
    """q -> flat 7-vector Pose3.to_storage() of robot.end_effectors[0]. Example
    symbolic_fn for crab_jit.simple.build_simple_jit_function."""
    _primitive_xyzr, _bounding_xyzr, link_poses = forward_kinematics(robot, q)
    ee_pose = link_poses[robot.link_name_to_index[robot.end_effectors[0]]]
    return sf.Matrix(ee_pose.to_storage())


def task_space_distance(robot: Robot, q, goal_pose_flat) -> sf.Matrix:
    """(q, goal_pose) -> [translation distance] between the end effector and a goal
    pose. goal_pose_flat is a flat 7-vector, Pose3.to_storage()'s layout -- same
    convention q_to_ee_pose returns, so goal_pose can come from a previous
    q_to_ee_pose call. Example symbolic_fn with an extra runtime input beyond q; pass
    extra_inputs=[("goal_pose", 7)] to build_simple_jit_function."""
    _primitive_xyzr, _bounding_xyzr, link_poses = forward_kinematics(robot, q)
    ee_pose = link_poses[robot.link_name_to_index[robot.end_effectors[0]]]
    goal_pose = sf.Pose3.from_storage(list(goal_pose_flat))
    rel = goal_pose.inverse() * ee_pose
    return sf.Matrix([rel.t.norm()])


def _make_traced_func(param_names: list[str], expr):
    """Builds a callable whose introspectable parameter names match param_names but
    which always returns the already-built `expr`. Codegen.function reads parameter
    names off the function signature to name generated C++ arguments; the expression
    graph itself has to be built ahead of time in plain Python (closing over the real
    q/extra symbols), since it can involve arbitrary robot-specific Python logic
    (e.g. forward_kinematics's joint traversal) that isn't itself symbolic tracing
    through a generic wrapper.
    """
    def _traced(*_args):
        return expr

    _traced.__signature__ = inspect.Signature(
        [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in param_names]
    )
    return _traced


def _find_generated_header(codegen_result, output_dir: str, func_name: str) -> str:
    """Locates the generated `{func_name}.h` from a Codegen.generate_function() call.

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


# Matches the generated function's own signature (e.g.
# "Eigen::Matrix<Scalar, 232, 1> ForwardKinematicsGenerated(const Eigen::Matrix<Scalar, 7, 1>& q)").
_GENERATED_FUNC_NAME_RE = re.compile(r"(\w+)\(\s*const Eigen::Matrix<Scalar,\s*\d+,\s*1>\s*&\s*q\b")


def _extract_generated_function_name(source: str, requested_name: str) -> str:
    """SymForce's CppConfig reformats the name passed to Codegen.function(name=...)
    into its own C++ naming convention (e.g. "forward_kinematics_generated" becomes
    "ForwardKinematicsGenerated") instead of emitting it verbatim, so the real symbol
    has to be read back out of the generated source rather than assumed to match what
    we asked for.
    """
    match = _GENERATED_FUNC_NAME_RE.search(source)
    if not match:
        raise ValueError(
            f"Could not find generated function signature for '{requested_name}' in generated source"
        )
    return match.group(1)


def build_and_generate_function(robot: Robot, symbolic_fn, name: str,
                                 extra_inputs: list[tuple[str, int]] | None = None) -> dict:
    """Generic one-shot codegen for a Path-A function, for use with
    crab_jit.simple.build_simple_jit_function.

    symbolic_fn(robot, q, *extra_values) -> sf.Matrix, an (n, 1) column vector.
    extra_inputs declares any inputs beyond q as (name, size) pairs -- e.g.
    [("goal_pose", 7)] for a flat Pose3.to_storage() goal (see task_space_distance).
    Each becomes its own sf.Matrix(size, 1).symbolic(name) argument and its own
    void** input slot, in declaration order after q. n_out is read back from
    symbolic_fn's own returned Matrix shape rather than asked for separately.
    """
    nq = robot.nq
    extra_inputs = extra_inputs or []

    q = sf.Matrix(nq, 1).symbolic("q")
    extra_syms = [sf.Matrix(size, 1).symbolic(ename) for ename, size in extra_inputs]

    expr = symbolic_fn(robot, q, *extra_syms)
    n_out = expr.shape[0]

    param_names = ["q"] + [ename for ename, _ in extra_inputs]
    traced_func = _make_traced_func(param_names, expr)
    input_types = [type(q)] + [type(s) for s in extra_syms]

    codegen = Codegen.function(func=traced_func, name=name, input_types=input_types, config=CppConfig())

    with tempfile.TemporaryDirectory(prefix="crab_jit_codegen_") as tmp_dir:
        result = codegen.generate_function(output_dir=tmp_dir)
        source = _find_generated_header(result, tmp_dir, name)

    func_name = _extract_generated_function_name(source, name)
    input_sizes = [nq] + [size for _, size in extra_inputs]

    return {"nq": nq, "n_out": n_out, "input_sizes": input_sizes, "source": source, "func_name": func_name}


def build_and_generate_function_with_jacobian(robot: Robot, symbolic_fn, name: str) -> dict:
    """Path-B counterpart to build_and_generate_function: also generates the Jacobian
    of symbolic_fn's output w.r.t. q, for embedding both the value and its Jacobian
    into one hand-written fused kernel translation unit (see
    crab_codegen.collision_kernel). Path A never needs a Jacobian -- nothing exposes a
    raw Jacobian externally anymore, it's only ever consumed internally by a fused
    kernel's own control flow.
    """
    nq = robot.nq
    q = sf.Matrix(nq, 1).symbolic("q")
    expr = symbolic_fn(robot, q)
    n_out = expr.shape[0]

    codegen_value = Codegen.function(func=lambda q: expr, name=name, input_types=[type(q)], config=CppConfig())
    jac_name = f"{name}_jacobian"
    codegen_jac = codegen_value.with_jacobian(which_args=["q"], name=jac_name)

    with tempfile.TemporaryDirectory(prefix="crab_jit_codegen_") as tmp_dir:
        value_dir = os.path.join(tmp_dir, "value")
        jac_dir = os.path.join(tmp_dir, "jac")

        value_result = codegen_value.generate_function(output_dir=value_dir)
        jac_result = codegen_jac.generate_function(output_dir=jac_dir)

        value_source = _find_generated_header(value_result, value_dir, name)
        jac_source = _find_generated_header(jac_result, jac_dir, jac_name)

    value_func_name = _extract_generated_function_name(value_source, name)
    jac_func_name = _extract_generated_function_name(jac_source, jac_name)

    return {
        "nq": nq,
        "n_out": n_out,
        "value_source": value_source,
        "value_func_name": value_func_name,
        "jac_source": jac_source,
        "jac_func_name": jac_func_name,
    }
