"""End-to-end examples for CRAB's JIT backend.

fused_demo() is the recommended path: FK + the collision loop + cost/gradient are all
JIT-compiled into one -O3 translation unit per robot (crab_codegen.collision_kernel),
so there's no function-pointer boundary between FK and the collision math -- the
compiler can inline and fuse across the whole thing.

pointer_demo() is the older, more modular path: only FK/Jacobian are JIT'd
(crab_codegen.jit_wrapper), and the fixed, separately-compiled loop in
src/collision_checker.cpp calls them through a function pointer. Useful if you need
crab_backend's generic RobotSpherized-based API for something other than raw q -> cost.

fk_only_demo() is the smallest example: JITs just forward kinematics (sphere
positions), nothing else -- no Jacobian, no collision loop, no fused kernel.
"""

import numpy as np

from crab_codegen.generate_math import load_panda_robot
from crab_jit import build_fused_collision_kernel


def fused_demo():
    robot = load_panda_robot()
    kernel = build_fused_collision_kernel(robot)

    q = np.zeros(robot.nq)
    print("collision free:", kernel.is_collision_free(q))
    print("cost:", kernel.compute_collision_cost(q))
    cost, grad = kernel.compute_collision_cost_with_gradient(q)
    print("cost:", cost, "grad:", grad)


def fk_only_demo():
    """JITs *only* forward kinematics (sphere positions) -- no Jacobian, no collision
    loop, no fused kernel. Smallest possible example of crab_jit: one generated C++
    function, wrapped, compiled, called."""
    from crab_codegen.robot_spherized import build_robot_spherized
    from crab_jit import build_robot_fk_jit
    import crab_backend

    robot = load_panda_robot()
    spherized = build_robot_spherized(robot)
    fk_ptr, engine = build_robot_fk_jit(robot)

    q = np.zeros(robot.nq)
    spheres_world = crab_backend.forward_kinematics(q, spherized, fk_ptr)
    print("spheres_world:", spheres_world)

    engine.close()


def pointer_demo():
    from crab_codegen.robot_spherized import build_robot_spherized
    from crab_jit import build_robot_jit_functions
    import crab_backend

    robot = load_panda_robot()
    spherized = build_robot_spherized(robot)
    fk_ptr, jac_ptr, engine = build_robot_jit_functions(robot)

    q = np.zeros(robot.nq)
    spheres_world = crab_backend.forward_kinematics(q, spherized, fk_ptr)
    print("spheres_world:", spheres_world)
    print("collision free:", crab_backend.is_collision_free(q, spherized, fk_ptr))
    cost, grad = crab_backend.compute_collision_cost_with_gradient(q, spherized, 0.05, fk_ptr, jac_ptr)
    print("cost:", cost, "grad:", grad)

    engine.close()


if __name__ == "__main__":
    # fused_demo()
    fk_only_demo()
