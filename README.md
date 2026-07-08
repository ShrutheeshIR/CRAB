# 🦀 CRAB: Compiled Robotics with Accelerated Backend

> *Because eventually, all high-performance robotics code evolves into C++.*

CRAB is an ultra-fast, code-generated robotics library designed to perform Forward Kinematics (FK), Inverse Kinematics (IK), collision checking, and forward/inverse dynamics. By combining the flexibility of symbolic python prototyping (using **SymForce** and **SymPy**) with the runtime speed of compiled C++, CRAB generates highly optimized C++ kernels and wraps them back into Python using the modern and lightweight **nanobind** library.

---

## 🚀 The CRAB Paradigm: Why & How

### 1. The Core Philosophy
Traditional robotics frameworks are general-purpose. They introduce runtime overhead from conditional branching, dynamic memory allocation, and lack of specialized math optimization for specific robot topologies.

**CRAB** compiles your robot's kinematic and dynamic equations symbolically. SymForce simplifies these equations, performs **Common Subexpression Elimination (CSE)**, and writes a flat, branchless, static-sized C++ mathematical kernel specifically optimized for *your* robot.

```
+------------------+      Symbolic Math      +------------------------+
|    URDF/SRDF     | ----------------------> |    SymForce Models     |
|   Parser & IO    |                         | (FK, Dynamics, Cost)   |
+------------------+                         +------------------------+
                                                          |
                                                          | Codegen (C++)
                                                          v
+------------------+                         +------------------------+
|    Python API    | <---------------------- |   Optimized C++ Core   |
|  (nanobind bound)|    nanobind Wrapper     |    (Eigen-accelerated) |
+------------------+                         +------------------------+
```

### 2. What Makes CRAB Unique?
* **Differentiable & Binary Hybrid Collision Checking**:
  * **Differentiable Mode**: Computes smooth, continuous collision penalty functions and their analytical Jacobians using spherized geometry, allowing solvers to "push" the robot out of collisions.
  * **Binary Mode**: A highly optimized boolean checker that leverages C++ loops and **early-exits** immediately upon detecting the first colliding pair, achieving massive speedups.
* **Nanobind Integration**: Replaces heavy Pybind11 templates with a lean, modern C++ library for exposing Eigen structures and compiled functions to Python, resulting in faster compilation, smaller binary sizes, and minimal overhead.
* **Symbolic Dynamics Acceleration**: CRAB computes the equations of motion (e.g., Mass Matrix $M(q)$, Coriolis/Centrifugal terms $C(q,v)v$, and gravity $G(q)$) symbolically. The generated C++ solver evaluates these complex matrices in microseconds without dynamic allocation.

---

## 📁 Project Architecture & Components

```
crab/
├── robots/                     # URDF/SRDF descriptions of robots (e.g., Franka Emika Panda)
│   └── panda/
│       ├── panda_spherized.urdf # Robot URDF spherized with collision spheres
│       └── panda.srdf          # Self-collision exclusion pairs
├── crusty_model/               # Python symbolic representations of Robot, Link, Joint
│   ├── robot.py                # Kinematic tree traversal and validation
│   ├── link.py                 # Link dataclass & minimum enclosing sphere solver
│   ├── joint.py                # Joint dataclass (revolute, prismatic, fixed)
│   └── _bounding_sphere.py     # Welzl's minimum enclosing sphere algorithm
├── crusty_kinematics/          # Pure-Python symbolic kinematics & collision checking
│   ├── fk.py                   # Symbolic Forward Kinematics generator
│   └── collision_checker.py    # Symbolic collision distance & cost checker
├── crab_codegen/               # Code generation engine
│   ├── generate_math.py        # Generates optimized C++ source code using SymForce Codegen
│   └── templates/              # Jinja2/C++ code templates for custom solvers
└── src/                        # Hand-written & Generated C++ code and Python bindings
    ├── CMakeLists.txt          # Build system compiling the nanobind module
    ├── bindings.cpp            # Nanobind module definition exposing Eigen and functions
    ├── collision_helpers.hpp   # C++ definitions for external symbolic functions (sp.Function)
    └── collision_checker.cpp   # Custom C++ early-exiting binary & differentiable collision loops
```

---

## 🛠️ Step-by-Step Implementation Blueprint

### Step 1: Mapping `sp.Function` to C++ Definitions
In `crusty_kinematics/collision_checker.py`, we use symbolic functions such as:
```python
CCC = sp.Function("sphere_environment_collision_checker")
SSC = sp.Function("sphere_sphere_collision_checker")
```
When SymForce compiles these symbolic expressions, it writes them as native function calls in C++. To build successfully:
1. We configure SymForce `CppConfig` to inject extra header files:
   ```python
   config = CppConfig(extra_headers=["collision_helpers.hpp"])
   ```
2. We write `src/collision_helpers.hpp` defining these functions:
   ```cpp
   #pragma once
   #include <cmath>

   inline float sphere_sphere_collision_checker(float x1, float y1, float z1, float r1, 
                                                float x2, float y2, float z2, float r2) {
       float dx = x1 - x2;
       float dy = y1 - y2;
       float dz = z1 - z2;
       float dist = std::sqrt(dx*dx + dy*dy + dz*dz) - (r1 + r2);
       return dist < 0.0f ? 1.0f : 0.0f; // Binary return or smooth cost
   }
   ```

### Step 2: Differentiable vs. Binary Collision Loops
To allow early-exit, we do **not** compile the outer collision pair loop symbolically. A fully unrolled symbolic loop would prevent branching. Instead:
1. **CodeGen Piecewise Operations**: SymForce compiles the Forward Kinematics and pairwise distance calculations.
2. **C++ Loop Traversal**: We write the outer loop in C++ under `src/collision_checker.cpp`.

```cpp
// Binary Collision Check with Early Exit
bool is_collision_free(const Eigen::VectorXd& q) {
    // 1. Compute Forward Kinematics (generated by SymForce)
    auto link_poses = forward_kinematics_generated(q);
    
    // 2. Loop over collision pairs and return false immediately on collision
    for (const auto& [link_a, link_b] : allowed_collision_pairs) {
        for (const auto& sphere_a : link_a.spheres) {
            for (const auto& sphere_b : link_b.spheres) {
                float dist = compute_sphere_distance(link_poses, sphere_a, sphere_b);
                if (dist < 0.0f) {
                    return false; // EARLY EXIT: Instant return!
                }
            }
        }
    }
    return true;
}
```

### Step 3: Nanobind Binding Setup
We expose our high-performance C++ classes and functions to Python using `nanobind`. It has native support for Eigen mapping:

```cpp
#include <nanobind/nanobind.h>
#include <nanobind/eigen/matrix.h>
#include "collision_checker.hpp"

namespace nb = nanobind;

NB_MODULE(crab_backend, m) {
    m.def("is_collision_free", &is_collision_free, "q"_a,
          "Returns True if the robot configuration is collision free, False otherwise.");
    m.def("compute_collision_cost_and_jacobian", &compute_collision_cost_and_jacobian, "q"_a,
          "Computes differentiable collision cost and analytical Jacobian.");
}
```

---

## 📈 Roadmap & Milestones

- [x] **Phase 1: Foundation**: Parse URDF, extract spherized collision geometries, and build the symbolic kinematic tree.
- [ ] **Phase 2: CodeGen Infrastructure**: Write scripts to compile symbolic Forward Kinematics and Jacobians into C++ and verify Nanobind import.
- [ ] **Phase 3: Differentiable & Binary Collision Checking**: Write custom C++ loops with early-exits and compile differentiable analytical gradients.
- [ ] **Phase 4: Dynamics Compilation**: Implement symbolic dynamics (RNEA/ABA) and generate C++ implementations for mass matrix and joint torques.
- [ ] **Phase 5: Solvers**: Create C++ optimization-based IK and trajectory-optimization modules using nanobind.
- [ ] **Phase 6: Benchmark**: Profile CRAB against standard libraries (Pinocchio, PyBullet).

---

## 🧪 Running Existing Tests

To test the current symbolic representation of kinematics and spherized URDF parsing, run the following prototype scripts:

```bash
# Test URDF parsing and Link/Joint tree creation
python -m crusty_io.test_urdf

# Test Symbolic Forward Kinematics
python -m crusty_kinematics.test_fk

# Test Symbolic Collision Setup
python -m crusty_kinematics.test_collision
```
