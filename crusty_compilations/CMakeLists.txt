cmake_minimum_required(VERSION 3.20)
project(crusty_jit_lib LANGUAGES C CXX)

# --------------------------------------------------------------------
# Find LLVM and Clang (only needed for include directories)
# --------------------------------------------------------------------
find_package(LLVM REQUIRED CONFIG PATHS /usr/lib/llvm-15 NO_DEFAULT_PATH)
list(APPEND CMAKE_PREFIX_PATH "${LLVM_DIR}")
find_package(Clang REQUIRED CONFIG)

# --------------------------------------------------------------------
# Source files for the JIT library
# --------------------------------------------------------------------
set(SRC_FILES
    src/jit/compiler.cc
    src/jit/object_cache.cc
    src/jit/session.cc
)

# --------------------------------------------------------------------
# Build a static library (no linking against LLVM/Clang libraries)
# --------------------------------------------------------------------
add_library(crusty_jit_lib STATIC ${SRC_FILES})

# Include directories – expose JIT headers and LLVM/Clang headers
target_include_directories(crusty_jit_lib PUBLIC
    ${LLVM_INCLUDE_DIRS}
    ${CLANG_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)

# Require C++17 (minimum needed for the JIT code)
target_compile_features(crusty_jit_lib PUBLIC cxx_std_17)

# Optional install rule (remove if not needed)
install(TARGETS crusty_jit_lib
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin)