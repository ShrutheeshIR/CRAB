"""Generates the extern "C" glue that adapts a raw SymForce-generated C++ function to
the `void** inputs, void** outputs` calling convention -- the one and only ABI boundary
between JIT'd C++ and Python in this codebase (see REDESIGN.md). Both authoring paths
use it: crab_jit.simple (Path A, one auto-generated function, any number of inputs)
and hand-written fused kernels (Path B, e.g. crab_codegen/collision_kernel.py) alike.

SymForce's default CppConfig emits, for a Codegen.function() with a single Matrix
return value, a function that returns its result by value, one parameter per declared
input:

    template <typename Scalar>
    Eigen::Matrix<Scalar, N, 1> FuncName(const Eigen::Matrix<Scalar, NQ, 1>& q,
                                         const Eigen::Matrix<Scalar, M, 1>& extra) { ... }

crab_codegen.codegen reads the real emitted function name back out of the
generated source (SymForce reformats whatever name you pass to its own C++ naming
convention), so there's no name-guessing here -- only the ABI wrapper.
"""

VALUE_WRAPPER_TEMPLATE = """\
extern "C" void {wrapper_name}(void** inputs, void** outputs) {{
{input_decls}
    Eigen::Matrix<double, {n_out}, 1> result = sym::{func_name}<double>({call_args});
    Eigen::Map<Eigen::Matrix<double, {n_out}, 1>>(reinterpret_cast<double*>(outputs[0])) = result;
}}
"""

_INPUT_DECL_TEMPLATE = (
    "    Eigen::Matrix<double, {size}, 1> {arg_name}("
    "Eigen::Map<const Eigen::Matrix<double, {size}, 1>>(reinterpret_cast<const double*>(inputs[{index}])));"
)


def generate_value_wrapper(func_name: str, input_sizes: list[int], n_out: int, wrapper_name: str) -> str:
    """Wraps any SymForce-generated `(arg0, arg1, ...) -> flat vector of n_out
    doubles` function (return-by-value, no Jacobian) in the void**-ABI extern "C"
    glue. input_sizes gives the flat size of each positional input, in order (e.g.
    [nq] for a plain q -> ... function, [nq, 7] for q plus a flat goal pose). This is
    the one wrapper shape nearly every one-off kinematic quantity needs -- see
    crab_jit.simple.build_simple_jit_function, which uses this directly."""
    arg_names = [f"arg{i}" for i in range(len(input_sizes))]
    input_decls = "\n".join(
        _INPUT_DECL_TEMPLATE.format(size=size, arg_name=arg_name, index=i)
        for i, (arg_name, size) in enumerate(zip(arg_names, input_sizes))
    )
    return VALUE_WRAPPER_TEMPLATE.format(
        wrapper_name=wrapper_name,
        input_decls=input_decls,
        n_out=n_out,
        func_name=func_name,
        call_args=", ".join(arg_names),
    )
