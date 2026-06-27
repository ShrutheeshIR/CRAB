cmake_minimum_required(VERSION 3.20)
project(crusty_jit_lib LANGUAGES C CXX)

# --------------------------------------------------------------------
# Find LLVM and Clang
# --------------------------------------------------------------------
find_package(LLVM REQUIRED CONFIG PATHS /usr/lib/llvm-18 NO_DEFAULT_PATH)
list(APPEND CMAKE_PREFIX_PATH "${LLVM_DIR}")
find_package(Clang REQUIRED CONFIG)

message(STATUS "LLVM version: ${LLVM_PACKAGE_VERSION}")
message(STATUS "LLVM libraries dir: ${LLVM_LIBRARY_DIRS}")

# --------------------------------------------------------------------
# Collect LLVM/Clang link libraries
# --------------------------------------------------------------------
# llvm_map_components_to_libnames(
#     LLVM_LIBS
#     core support orcjit native nativecodegen
#     irreader passes target mc
#     executionengine runtimedyld object
# )

# set(CLANG_LIBS
#     clangDriver
#     clangFrontend
#     clangCodeGen
#     clangSerialization
#     clangParse
#     clangSema
#     clangAnalysis
#     clangEdit
#     clangAST
#     clangLex
#     clangBasic
# )

# --------------------------------------------------------------------
# Build the crusty_jit_lib static library
# --------------------------------------------------------------------
set(SRC_FILES
    src/jit/compiler.cc
    src/jit/object_cache.cc
    src/jit/session.cc
)

add_library(crusty_jit_lib STATIC ${SRC_FILES})

# Enable Position Independent Code (-fPIC) so it can be linked into shared libraries
set_target_properties(crusty_jit_lib PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(crusty_jit_lib PUBLIC
    ${LLVM_INCLUDE_DIRS}
    ${CLANG_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)

target_compile_features(crusty_jit_lib PUBLIC cxx_std_17)

# Link LLVM and Clang libraries to the target itself so consumers inherit them
target_link_libraries(crusty_jit_lib PUBLIC
    # ${LLVM_LIBS}
    # ${CLANG_LIBS}
    LLVM
    clang-cpp
)

# --------------------------------------------------------------------
# Example: C++ JIT Executable
# --------------------------------------------------------------------
add_executable(example_jit examples/example_jit.cpp)
target_link_libraries(example_jit PRIVATE crusty_jit_lib)

# --------------------------------------------------------------------
# C-API shared library (for Python ctypes)
# --------------------------------------------------------------------
add_library(crab_jit_capi SHARED examples/crab_jit_capi.cc)
target_link_libraries(crab_jit_capi PRIVATE crusty_jit_lib)

# --------------------------------------------------------------------
# Installation
# --------------------------------------------------------------------
install(TARGETS crusty_jit_lib example_jit crab_jit_capi
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
