import time

import numpy as np

from crab_codegen.ik_symforce_optimizer import build_ik_solver
from crab_codegen.ik_fused import build_fused_ik_solver
from crab_codegen.fixtures import load_panda_robot
from crusty_kinematics.derived import q_to_ee_pose, task_space_distance
from crab_jit import build_simple_jit_function

from crusty_model.robot import _joint_limits_by_index


def bench_ik_solve(robot, n_init: int = 50, n_goals_per_init: int = 10) -> None:
    """50x10: 50 random starting configurations (q_init), and for each one, 10 random
    goals -- so solve time gets measured across a spread of (start, goal) pairs
    instead of one lucky/unlucky combination, while still separating "does start
    config matter" from "does goal matter" in how the sweep is structured.

    Both solvers (SymForce's own Optimizer vs. the hand-fused JIT kernel) are timed
    over the exact same generated (q_init, goal_pose) pairs, so this is an
    apples-to-apples comparison rather than two separately-seeded runs. Timing alone
    isn't enough though -- a solver that's fast because it's not actually converging
    would win on speed and be useless, so distance-to-target gets checked for both
    too, using the same task_space_distance already reused elsewhere in this repo.
    """
    joint_limits = _joint_limits_by_index(robot)
    lower = np.array([lo for lo, _ in joint_limits])
    upper = np.array([hi for _, hi in joint_limits])

    ee_kernel = build_simple_jit_function(robot, q_to_ee_pose, name="q_to_ee")
    dist_kernel = build_simple_jit_function(robot, task_space_distance, name="task_space_distance",
                                             extra_inputs=[("goal_pose", 7)])

    # Built once, outside both loops -- this is the expensive one-time setup for each
    # (Factor/Optimizer tracing+codegen for the SymForce path, JIT compile for the
    # fused path). Only the per-pair solve calls below get timed.
    symforce_solver = build_ik_solver(robot)
    fused_solver = build_fused_ik_solver(robot)

    rng = np.random.default_rng(0)

    def random_q():
        return np.clip(rng.uniform(-2.5, 2.5, size=robot.nq), lower, upper)

    # Generate the whole (q_init, goal_pose) sweep once, up front, so both solvers see
    # identical problems in identical order.
    problems = []
    for _ in range(n_init):
        q_init = random_q()
        for _ in range(n_goals_per_init):
            goal_pose = ee_kernel(random_q())
            problems.append((q_init, goal_pose))

    # Warm-up: skip whatever's lazily initialized on each solver's very first call.
    for q_init, goal_pose in problems[:5]:
        symforce_solver.solve(goal_pose, q_init=q_init)
        fused_solver.solve(q_init, goal_pose)

    symforce_samples = []
    symforce_distances = []
    for q_init, goal_pose in problems:
        start = time.perf_counter()
        solved_q = symforce_solver.solve(goal_pose, q_init=q_init)
        symforce_samples.append(time.perf_counter() - start)
        symforce_distances.append(float(dist_kernel(solved_q, goal_pose)[0]))

    fused_samples = []
    fused_distances = []
    for q_init, goal_pose in problems:
        start = time.perf_counter()
        solved_q, _cost = fused_solver.solve(q_init, goal_pose)
        fused_samples.append(time.perf_counter() - start)
        fused_distances.append(float(dist_kernel(solved_q, goal_pose)[0]))

    symforce_us = np.array(symforce_samples) * 1e6
    fused_us = np.array(fused_samples) * 1e6
    symforce_dist = np.array(symforce_distances)
    fused_dist = np.array(fused_distances)

    print(f"IK solve over {n_init}x{n_goals_per_init} = {len(problems)} (q_init, goal) pairs:")
    print(f"  symforce Optimizer: median = {np.median(symforce_us):.2f} us/call, "
          f"mean = {np.mean(symforce_us):.2f} us/call, max = {np.max(symforce_us):.2f} us/call")
    print(f"  fused JIT kernel:   median = {np.median(fused_us):.2f} us/call, "
          f"mean = {np.mean(fused_us):.2f} us/call, max = {np.max(fused_us):.2f} us/call")
    print(f"  speedup (median): {np.median(symforce_us) / np.median(fused_us):.1f}x")
    print()
    print("distance to target (translation, meters), lower is better:")
    print(f"  symforce Optimizer: median = {np.median(symforce_dist):.6f}, "
          f"mean = {np.mean(symforce_dist):.6f}, max = {np.max(symforce_dist):.6f}")
    print(f"  fused JIT kernel:   median = {np.median(fused_dist):.6f}, "
          f"mean = {np.mean(fused_dist):.6f}, max = {np.max(fused_dist):.6f}")


if __name__ == "__main__":
    robot = load_panda_robot()
    bench_ik_solve(robot)
