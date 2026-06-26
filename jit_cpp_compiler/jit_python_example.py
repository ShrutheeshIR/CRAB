import ctypes
import numpy as np

# 1. Load the host C++ JIT library
lib = ctypes.CDLL("./libjit_engine.so")

# Setup arg types and return types for the engine hooks
lib.init_jit_engine.restype = ctypes.c_bool

lib.compile_and_load_string.argtypes = [
    ctypes.c_char_p,  # cpp_source
    ctypes.c_char_p,  # func_name
    ctypes.POINTER(ctypes.c_char_p),  # include_paths
    ctypes.c_int  # num_includes
]
lib.compile_and_load_string.restype = ctypes.c_uint64

# Fire up the engine
if not lib.init_jit_engine():
    raise RuntimeError("Failed to initialize LLVM JIT engine.")

# 2. Define your SymForce function code dynamically
# (In a real app, use your automated string generator here)
cpp_wrapper_code = b"""
#include <Eigen/Core>
#include <array>

// Imagine this is the function inside your SymForce .h file
namespace sym {
    template <typename Scalar>
    void fk(const Eigen::Matrix<Scalar, 7, 1>& q, std::array<double, 3>* res) {
        (*res)[0] = q(0) * 2.0;
        (*res)[1] = q(1) * 3.0;
        (*res)[2] = q(2) * 4.0;
    }
}

// Our Universal Wrapper Interface
extern "C" void jit_fk(void** inputs, void** outputs) {
    // Cast the raw void pointers to exactly what SymForce needs
    auto* q_ptr = reinterpret_cast<const Eigen::Matrix<double, 7, 1>*>(inputs[0]);
    auto* res_ptr = reinterpret_cast<std::array<double, 3>*>(outputs[0]);
    
    sym::fk<double>(*q_ptr, res_ptr);
}
"""

# 3. Compile the code using the engine
# includes = [b"/usr/include/eigen3", b"."]  # Add your symforce output directory here

includes = [
    b"/usr/include/eigen3",
    b"/usr/include/c++/13",
    b"/usr/include/x86_64-linux-gnu/c++/13",
    b"/usr/lib/llvm-18/lib/clang/18/include",  # Crucial for internal compiler headers like stddef.h
    b"/usr/include",
    b"/usr/include/x86_64-linux-gnu",
    b"."  # Your local SymForce wrapper folder
]
include_type = ctypes.c_char_p * len(includes)
include_paths_ptr = include_type(*includes)

func_address = lib.compile_and_load_string(
    cpp_wrapper_code, 
    b"jit_fk", 
    include_paths_ptr, 
    len(includes)
)

if func_address == 0:
    raise RuntimeError("JIT Compilation Failed!")

print(f"🚀 Success! Function compiled in-memory at address: {hex(func_address)}")

# 4. Bind and execute the dynamic function address via ctypes
# Our universal signature: void(void** inputs, void** outputs)
JITFunc = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p))
compiled_fk = JITFunc(func_address)

# Create input and output buffers matching your exact robot states
q_input = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
result_output = np.zeros(3, dtype=np.float64)

# Build void* arrays tracking the memory locations of inputs/outputs
input_pointers = (ctypes.c_void_p * 1)(q_input.ctypes.data)
output_pointers = (ctypes.c_void_p * 1)(result_output.ctypes.data)

# Run it! Zero copy, pure native assembly execution speed.
compiled_fk(input_pointers, output_pointers)

print("Result from JIT Execution:", result_output)