"""High-level glue: SymForce codegen -> extern "C" wrapper source -> JIT-compiled
function pointers, ready to pass into crab_backend's fk_fn_ptr / jac_fn_ptr
parameters (src/collision_checker.hpp)."""

from typing import Optional, Tuple

from crab_codegen.generate_math import generate_jit_sources, load_panda_robot
from crab_codegen.jit_wrapper import (
    build_jit_source,
    build_fk_only_source,
    DEFAULT_FK_WRAPPER_NAME,
    DEFAULT_JAC_WRAPPER_NAME,
)

from .engine import CrabJitEngine


def build_robot_jit_functions(robot=None,
                               engine: Optional[CrabJitEngine] = None,
                               module_id: Optional[str] = None) -> Tuple[int, int, CrabJitEngine]:
    """Compiles this robot's forward-kinematics + position-Jacobian to native code at
    -O3 via crab::jit::ClangCompiler and returns (fk_fn_ptr, jac_fn_ptr, engine).

    The caller must keep `engine` alive for as long as the returned pointers are used:
    the LLJIT session owns the memory the pointers refer to.
    """
    robot = robot or load_panda_robot()
    generated = generate_jit_sources(robot)
    source = build_jit_source(generated)

    engine = engine or CrabJitEngine()
    module_id = module_id or f"crab_jit_{robot.name}_{generated['nq']}dof"
    engine.compile(source, module_id)

    fk_ptr = engine.lookup(DEFAULT_FK_WRAPPER_NAME)
    jac_ptr = engine.lookup(DEFAULT_JAC_WRAPPER_NAME)

    return fk_ptr, jac_ptr, engine


def build_robot_fk_jit(robot=None,
                        engine: Optional[CrabJitEngine] = None,
                        module_id: Optional[str] = None) -> Tuple[int, CrabJitEngine]:
    """FK-only counterpart to build_robot_jit_functions: JITs just sphere-position
    forward kinematics (no Jacobian, no collision loop) and returns (fk_fn_ptr, engine).
    """
    robot = robot or load_panda_robot()
    generated = generate_jit_sources(robot)
    source = build_fk_only_source(generated)

    engine = engine or CrabJitEngine()
    module_id = module_id or f"crab_jit_fk_{robot.name}_{generated['nq']}dof"
    engine.compile(source, module_id)

    fk_ptr = engine.lookup(DEFAULT_FK_WRAPPER_NAME)
    return fk_ptr, engine
