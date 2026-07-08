"""Generates the extern "C" glue that adapts a raw SymForce-generated C++ function to
the `void** inputs, void** outputs` calling convention used throughout CRAB's JIT path
(see src/collision_checker.cpp's fk_fn_ptr/jac_fn_ptr and
crusty_compilations/examples/example_jit.py). This replaces the broken sketch in
jit_cpp_compiler/jit_wrapper.cc, which never compiled and was never wired to a real
SymForce function name.

SymForce's default CppConfig emits, for a Codegen.function() with a single Matrix
return value, a function that returns its result by value:

    template <typename Scalar>
    Eigen::Matrix<Scalar, N, 1> FuncName(const Eigen::Matrix<Scalar, NQ, 1>& q) { ... }

and for with_jacobian(), the jacobian is appended as an extra optional output pointer
while the primary value is still returned by value:

    template <typename Scalar>
    Eigen::Matrix<Scalar, M, 1> FuncName(const Eigen::Matrix<Scalar, NQ, 1>& q,
                                         Eigen::Matrix<Scalar, M, NQ>* const jacobian = nullptr) { ... }

The wrappers below assume that shape. crab_codegen.generate_math pins the emitted
function names via Codegen.function(name=...), so there's no name-guessing here.
"""

VALUE_WRAPPER_TEMPLATE = """\
extern "C" void {wrapper_name}(void** inputs, void** outputs) {{
    Eigen::Matrix<double, {nq}, 1> q(
        Eigen::Map<const Eigen::Matrix<double, {nq}, 1>>(reinterpret_cast<const double*>(inputs[0])));
    Eigen::Matrix<double, {n_out}, 1> result = sym::{func_name}<double>(q);
    Eigen::Map<Eigen::Matrix<double, {n_out}, 1>>(reinterpret_cast<double*>(outputs[0])) = result;
}}
"""


# collision_checker.cpp expects the flat Jacobian buffer laid out sphere-major: for
# each sphere, a column-major-flattened 3 x nq block. SymForce instead emits one
# (3*n_spheres) x nq matrix (joint-major columns), so the wrapper reshapes per sphere.
JACOBIAN_WRAPPER_TEMPLATE = """\
extern "C" void {wrapper_name}(void** inputs, void** outputs) {{
    Eigen::Matrix<double, {nq}, 1> q(
        Eigen::Map<const Eigen::Matrix<double, {nq}, 1>>(reinterpret_cast<const double*>(inputs[0])));
    Eigen::Matrix<double, {n_pos}, {nq}> jac;
    sym::{func_name}<double>(q, &jac);
    double* out = reinterpret_cast<double*>(outputs[0]);
    for (int s = 0; s < {n_spheres}; ++s) {{
        Eigen::Map<Eigen::Matrix<double, 3, {nq}>>(out + s * 3 * {nq}) =
            jac.template block<3, {nq}>(s * 3, 0);
    }}
}}
"""

DEFAULT_FK_WRAPPER_NAME = "jit_forward_kinematics"
DEFAULT_JAC_WRAPPER_NAME = "jit_forward_kinematics_jacobian"


def generate_value_wrapper(func_name: str, nq: int, n_out: int, wrapper_name: str) -> str:
    """Wraps any SymForce-generated `q -> flat vector of n_out doubles` function
    (return-by-value, no Jacobian) in the void**-ABI extern "C" glue. This is the
    generic case nearly every one-off kinematic quantity falls into -- see
    crab_jit.simple.build_simple_jit_function, which uses this directly instead of
    needing a bespoke template per function."""
    return VALUE_WRAPPER_TEMPLATE.format(wrapper_name=wrapper_name, nq=nq, func_name=func_name, n_out=n_out)


def generate_fk_wrapper(func_name: str, nq: int, n_spheres: int, wrapper_name: str = DEFAULT_FK_WRAPPER_NAME) -> str:
    return generate_value_wrapper(func_name, nq, n_spheres * 4, wrapper_name)


def generate_jacobian_wrapper(func_name: str, nq: int, n_spheres: int,
                               wrapper_name: str = DEFAULT_JAC_WRAPPER_NAME) -> str:
    return JACOBIAN_WRAPPER_TEMPLATE.format(
        wrapper_name=wrapper_name, nq=nq, func_name=func_name,
        n_pos=n_spheres * 3, n_spheres=n_spheres,
    )


def build_jit_source(generated: dict,
                      fk_wrapper_name: str = DEFAULT_FK_WRAPPER_NAME,
                      jac_wrapper_name: str = DEFAULT_JAC_WRAPPER_NAME) -> str:
    """Concatenates the raw SymForce-generated headers (from
    crab_codegen.generate_math.generate_jit_sources) with their extern "C" wrappers
    into a single translation unit ready for crab::jit::ClangCompiler::compile()."""
    nq = generated["nq"]
    n_spheres = generated["n_spheres"]

    parts = [
        "#include <Eigen/Dense>",
        "",
        generated["fk_source"],
        generated["jac_source"],
        "",
        generate_fk_wrapper(generated["fk_func_name"], nq, n_spheres, fk_wrapper_name),
        "",
        generate_jacobian_wrapper(generated["jac_func_name"], nq, n_spheres, jac_wrapper_name),
    ]
    return "\n".join(parts)


def build_fk_only_source(generated: dict, fk_wrapper_name: str = DEFAULT_FK_WRAPPER_NAME) -> str:
    """Same as build_jit_source but skips the Jacobian entirely -- only FK's generated
    header + its extern "C" wrapper, for callers that just want sphere positions."""
    nq = generated["nq"]
    n_spheres = generated["n_spheres"]

    parts = [
        "#include <Eigen/Dense>",
        "",
        generated["fk_source"],
        "",
        generate_fk_wrapper(generated["fk_func_name"], nq, n_spheres, fk_wrapper_name),
    ]
    return "\n".join(parts)
