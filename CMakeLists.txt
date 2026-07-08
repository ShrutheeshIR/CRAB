cmake_minimum_required(VERSION 3.15)
project(CRAB LANGUAGES CXX)

# Require C++17 (nanobind requirement)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen3
find_package(Eigen3 3.3 REQUIRED)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# Use FetchContent to fetch nanobind
include(FetchContent)
FetchContent_Declare(
  nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind.git
  # GIT_TAG        MAIN
  GIT_SUBMODULES_RECURSE ON
)
FetchContent_MakeAvailable(nanobind)

# Add python development component for nanobind
# nanobind_initialize()

# Define generated paths
set(SYMFORCE_GEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/generated/cpp")
set(GEN_FK_DIR "${SYMFORCE_GEN_DIR}/symforce/forward_kinematics_generated")
set(GEN_JAC_DIR "${SYMFORCE_GEN_DIR}/symforce/forward_kinematics_jacobian_generated")

# Check if the actual generated SymForce C++ files exist.
# If they do, compile them. If not, fallback to the placeholder.
if(EXISTS "${GEN_FK_DIR}/forward_kinematics_generated.cpp" AND EXISTS "${GEN_JAC_DIR}/forward_kinematics_jacobian_generated.cpp")
  set(SYMFORCE_SOURCES
    "${GEN_FK_DIR}/forward_kinematics_generated.cpp"
    "${GEN_JAC_DIR}/forward_kinematics_jacobian_generated.cpp"
  )
  message(STATUS "CRAB: Found generated SymForce files. Compiling with real kinematics/Jacobians.")
else()
  set(SYMFORCE_SOURCES
    src/generated_placeholder.cpp
  )
  message(STATUS "CRAB: SymForce generated files not found. Compiling with placeholder.")
endif()

# Define the C++ extension module
nanobind_add_module(crab_backend
  src/bindings.cpp
  src/collision_checker.cpp
  ${SYMFORCE_SOURCES}
)

# Include directories
target_include_directories(crab_backend PRIVATE
  src
  ${EIGEN3_INCLUDE_DIR}
  # Add the generated folder paths so the headers are found
  "${SYMFORCE_GEN_DIR}"
  "${GEN_FK_DIR}"
  "${GEN_JAC_DIR}"
)

# Link against Eigen and other necessary libraries
target_link_libraries(crab_backend PRIVATE
  Eigen3::Eigen
)

