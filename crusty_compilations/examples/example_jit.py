# Example Python wrapper for Crab JIT compiler (dummy)

import ctypes
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(SCRIPT_DIR, "../build")

lib_path = os.path.join(BUILD_DIR, "libcrab_jit_capi.so")

if not os.path.exists(lib_path):
    print("Compiled JIT library not found – this is a dummy example.")
    sys.exit(1)


lib = ctypes.CDLL(lib_path)
lib.crab_jit_create.restype = ctypes.c_void_p
lib.crab_jit_create.argtypes = []

lib.crab_jit_destroy.restype = None
lib.crab_jit_destroy.argtypes = [ctypes.c_void_p]

lib.crab_jit_compile.restype = ctypes.c_int
lib.crab_jit_compile.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

lib.crab_jit_lookup.restype = ctypes.c_void_p
lib.crab_jit_lookup.argtypes = [ctypes.c_void_p, ctypes.c_char_p]


SOURCE = b"""
extern "C" int add(int a, int b) {
    return a + b;
}
extern "C" int subtract(int a, int b) {
    return a - b;
}
extern "C" double dot3(double x1, double y1, double z1, double x2, double y2, double z2) {
    return x1 * x2  + y1 * y2 + z1 * z2;
}
"""

def main():
    ctx = lib.crab_jit_create()
    if not ctx:
        sys.exit("Failed to create JIT context")
    
    try:
        rc = lib.crab_jit_compile(ctx, SOURCE, b"math_jit_py")
        if rc < 0:
            sys.exit(f"Compilation failed with error code {rc}")
        elif rc == 1:
            print("[cache hit] Loaded from cache")
        else:
            print("[cache miss] Compiled and cached")

        add_ptr = lib.crab_jit_lookup(ctx, b"add")
        if not add_ptr:
            sys.exit("Failed to find 'add' function")
        subtract_ptr = lib.crab_jit_lookup(ctx, b"subtract")
        if not subtract_ptr:
            sys.exit("Failed to find 'subtract' function")
        dot3_ptr = lib.crab_jit_lookup(ctx, b"dot3")
        if not dot3_ptr:
            sys.exit("Failed to find 'dot3' function")

        add_func = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_int)(add_ptr)
        subtract_func = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_int)(subtract_ptr)
        dot3_func = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double)(dot3_ptr)

        print("add(3, 4) =", add_func(3, 4))
        print("subtract(10, 5) =", subtract_func(10, 5))
        print("dot3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0) =", dot3_func(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
    finally:
        lib.crab_jit_destroy(ctx)

if __name__ == "__main__":
    main()