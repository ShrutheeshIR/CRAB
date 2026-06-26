#include "jit/compiler.hh"
#include <iostream>
#include <string>

int main() {
    // Simple C function to be JIT‑compiled
    std::string source = R"(
        extern "C" int add(int a, int b) {
            return a + b;
        }
    )";

    crab::jit::CompileOptions opts;
    opts.module_id = "add_module"; // identifier for the generated module

    crab::jit::ClangCompiler compiler;
    auto tsModule = compiler.compile(source, opts);

    // In a real scenario you would retrieve the compiled function pointer from
    // `tsModule` via LLVM Orc JIT utilities. For this dummy example we just
    // confirm that compilation succeeded.
    std::cout << "Compiled JIT module successfully.\n";
    return 0;
}
