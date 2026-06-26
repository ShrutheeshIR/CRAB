#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/MemoryBuffer.h> // Ensure MemoryBuffer is fully defined
#include <llvm/TargetParser/Host.h> // For getHostCPUName (LLVM 17+)
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/CodeGen/CodeGenAction.h>
#include <clang/Lex/PreprocessorOptions.h> // <-- Fixes your compiler error!

#include <memory>
#include <string>
#include <vector>
// Use C-linkage so Python's ctypes can easily find our engine functions
extern "C" {

    // Global JIT instance tracker
    static std::unique_ptr<llvm::orc::LLJIT> GlobalJIT;


    // 1. Initialize the JIT Engine once at startup
    bool init_jit_engine() {
    llvm::outs() << "=== JIT Engine Initializing ===\n";
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        llvm::InitializeNativeTargetAsmParser();

        auto JITExpect = llvm::orc::LLJITBuilder().create();
        if (!JITExpect) return false;
        
        GlobalJIT = std::move(*JITExpect);
        llvm::outs() << "=== JIT Engine Initialized ===\n";
        return true;
    }


    // 2. Compile a C++ string to machine code and return its memory address
    uint64_t compile_and_load_string(const char* cpp_source, const char* func_name, const char* include_paths[], int num_includes) {
        if (!GlobalJIT) return 0;

        llvm::outs() << "=== Compiling C++ Source ===\n";


        std::string host_cpu = llvm::sys::getHostCPUName().str();

        std::vector<std::string> args = {
            "-std=c++17",
            "-O3",
            "-triple", "x86_64-pc-linux-gnu",
            "-target-cpu", host_cpu // Dynamically passes your exact CPU (e.g., "alderlake")
        };

        for (int i = 0; i < num_includes; ++i) {
            args.push_back(std::string("-I") + include_paths[i]);
        }

        llvm::outs() << "=== Clang Arguments Done ===\n";

        // Configure Clang Frontend Execution
        clang::CompilerInstance Clang;
        auto DiagOpts = new clang::DiagnosticOptions();
        std::string DiagLog;
        llvm::raw_string_ostream DiagOS(DiagLog);
        Clang.createDiagnostics(new clang::TextDiagnosticPrinter(DiagOS, DiagOpts));

        // Create virtual file in memory
        std::unique_ptr<llvm::MemoryBuffer> Buffer = llvm::MemoryBuffer::getMemBufferCopy(cpp_source, "jit_source.cc");
        Clang.getPreprocessorOpts().addRemappedFile("jit_source.cc", Buffer.release());

        // Create compiler invocation
        std::vector<const char*> arg_pointers;
        for (const auto& arg : args) arg_pointers.push_back(arg.c_str());
        arg_pointers.push_back("jit_source.cc");

        llvm::outs() << "=== Creating Compiler Invocation ===\n";

        if (!clang::CompilerInvocation::CreateFromArgs(Clang.getInvocation(), arg_pointers, Clang.getDiagnostics())) {
            llvm::errs() << "=== CLANG INVOCATION FAILED ===\n";
            llvm::errs() << "Diagnostics Log:\n" << DiagLog << "\n"; // Print why arguments were rejected
            llvm::errs() << "Arguments passed:\n";
            for (auto* arg : arg_pointers) llvm::errs() << "  " << arg << "\n";
            llvm::errs() << "===============================\n";
            return 0; 
        }

        // Compile to LLVM IR Module
        clang::EmitLLVMOnlyAction Action;
        if (!Clang.ExecuteAction(Action)) {
            // FORCE CLANG TO SPIT OUT THE COMPILATION ERRORS TO TERMINAL
            llvm::errs() << "=== CLANG JIT COMPILATION FAILED ===\n";
            llvm::errs() << DiagLog << "\n";
            llvm::errs() << "====================================\n";
            return 0;
        }
        else {
            // Optional: Print successful compilation message
            llvm::outs() << "=== CLANG JIT COMPILATION SUCCEEDED ===\n";
        }

        llvm::outs() << "=== Taking LLVM IR Module ===\n";
        std::unique_ptr<llvm::Module> Mod = Action.takeModule();
        if (!Mod) return 0;

        // ThreadSafeContext management for ORCv2
        llvm::orc::ThreadSafeContext TSContext(std::make_unique<llvm::LLVMContext>());
        llvm::orc::ThreadSafeModule TSM(std::move(Mod), std::move(TSContext));

        // Inject into JIT Execution Engine
        if (GlobalJIT->addIRModule(std::move(TSM))) return 0;

        // Look up compiled function symbol address
        auto Sym = GlobalJIT->lookup(func_name);
        if (!Sym) {
            // Crucial: Clear the error state if lookup fails, 
            // otherwise LLVM will intentionally crash the program.
            llvm::consumeError(Sym.takeError()); 
            return 0;
        }

        return Sym->getValue();

    }
}