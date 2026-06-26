# Example Python wrapper for Crab JIT compiler (dummy)

import ctypes
import os

# Assume the compiled shared library is produced by the C++ JIT example above
# For this dummy example we just load a placeholder library if it exists.

lib_path = os.path.abspath("./libadd_module.so")
if os.path.exists(lib_path):
    lib = ctypes.CDLL(lib_path)
    add = lib.add
    add.argtypes = [ctypes.c_int, ctypes.c_int]
    add.restype = ctypes.c_int
    result = add(2, 3)
    print(f"JIT add result: {result}")
else:
    print("Compiled JIT library not found – this is a dummy example.")
