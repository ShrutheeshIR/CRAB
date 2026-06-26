def generate_universal_stub(symforce_dir, func_name):
    return f"""
#include "{symforce_dir}/{func_name}.h"
#include "universal_wrapper.h" // The file we wrote above

extern "C" void jit_{func_name}(void** inputs, void** outputs) {{
    // We pass the function pointer. C++ template argument deduction 
    // automatically figures out the variable inputs/outputs and unpacks them!
    sym::{func_name}<double>(inputs, outputs); 
}}
"""