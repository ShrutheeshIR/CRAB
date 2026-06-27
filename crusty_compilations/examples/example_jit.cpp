#include <jit/compiler.hh>
#include <jit/object_cache.hh>
#include <jit/session.hh>

#include <iostream>
#include <string>

int main() {
    // Simple C function to be JIT‑compiled
    std::string source = R"(
        extern "C" int add(int a, int b) {
            return a + b;
        }

        extern "C" int subtract(int a, int b) {
            return a - b;
        }
        
        extern "C" double dot3(double x1, double y1, double z1, double x2, double y2, double z2) {
            return x1 * x2 + y1 * y2 + z1 * z2;
        }
    )";

    auto cache = std::make_shared<crab::jit::DiskObjectCache>(crab::jit::default_cache_dir());

    crab::jit::CompileOptions compile_opts;
    compile_opts.module_id = "example_math"; // identifier for the generated module

    auto cached_obj = cache->load_object(compile_opts.module_id);

    crab::jit::JitSession jit_session(cache);

    if (cached_obj) {

        std::cout << "[cache hit] Found cached object file for module '" << compile_opts.module_id << "'.\n";
        // If the object file is cached, load it into the JIT session
        if (auto err = jit_session.add_object_file(std::move(cached_obj))) {
            llvm::consumeError(std::move(err));
            std::cerr << "Failed to add cached object file to JIT session "<< llvm::toString(std::move(err)) << "\n";
            return -1;
        }
        std::cout << "Loaded cached object file into JIT session.\n";
    } else {
        // Compile the source code and add the resulting module to the JIT session
        std::cout << "[cache miss] No cached object file found for module '" << compile_opts.module_id << "'. Compiling...\n";
        crab::jit::ClangCompiler compiler;
        auto tsModule = compiler.compile(source, compile_opts);

        if (auto err = jit_session.add_module(std::move(tsModule))) {
            llvm::consumeError(std::move(err));
            std::cerr << "Failed to add compiled module to JIT session "<< llvm::toString(std::move(err)) << "\n";
            return -1;
        }
        std::cout << "Compiled and added module to JIT session.\n";
    }

    auto add_func = jit_session.lookup_fn<int(int, int)>("add");
    if (!add_func) {
        std::cerr << "Failed to lookup 'add' function: " << llvm::toString(add_func.takeError()) << "\n";
        return -1;
    }

    auto subtract_func = jit_session.lookup_fn<int(int, int)>("subtract");
    if (!subtract_func) {
        std::cerr << "Failed to lookup 'subtract' function: " << llvm::toString(subtract_func.takeError()) << "\n";
        return -1;
    }
    auto dot3_func = jit_session.lookup_fn<double(double, double, double, double, double, double)>("dot3");
    if (!dot3_func) {
        std::cerr << "Failed to lookup 'dot3' function: " << llvm::toString(dot3_func.takeError()) << "\n";
        return -1;
    }

    std::cout << "add(3, 4) = " << (*add_func)(3, 4) << "\n";
    std::cout << "subtract(10, 5) = " << (*subtract_func)(10, 5) << "\n";
    std::cout << "dot3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0) = " << (*dot3_func)(1.0, 2.0, 3.0, 4.0, 5.0, 6.0) << "\n";


    return 0;
}
