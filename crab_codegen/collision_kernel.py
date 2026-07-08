"""Path B: a hand-written, fused JIT translation unit per robot -- FK + the
collision-pair loop + cost/gradient math all in one compilation unit, so -O3 can
inline and fuse across what a function-pointer call boundary would otherwise prevent.

Sphere radii and which sphere-index pairs to check are known once the robot model is
loaded (they don't depend on q), so they're baked in as compile-time constants.

Exported functions use the same `void** inputs, void** outputs` ABI as Path A
(crab_jit.simple) -- see crab_jit.binder.JitFunction, the one binding mechanism used
for both paths (REDESIGN.md). This is what let `FusedCollisionKernel` stop being a
bespoke ctypes.CFUNCTYPE class and become a thin wrapper over three generic
JitFunctions instead.
"""

from crusty_model.robot import Robot
from crab_codegen.sphere_order import robot_sphere_list
from crab_codegen.codegen import build_and_generate_function_with_jacobian
from crusty_kinematics.derived import forward_kinematics_spheres

FK_IS_FREE = "jit_is_collision_free"
FK_COST = "jit_compute_collision_cost"
FK_COST_GRAD = "jit_compute_collision_cost_with_gradient"

FUSED_KERNEL_TEMPLATE = """\
#include <Eigen/Dense>
#include <cmath>

{fk_source}
{jac_source}

namespace crab_fused {{

constexpr int kNQ = {nq};
constexpr int kNSpheres = {n_spheres};
constexpr double kRadii[kNSpheres] = {{{radii}}};

struct Pair {{ int a; int b; }};
constexpr int kNumPairs = {n_pairs};
constexpr Pair kSelfPairs[{pairs_array_size}] = {{{pairs}}};

}}  // namespace crab_fused

// inputs[0] = q (kNQ doubles). outputs[0][0] = 1.0 if collision-free, 0.0 otherwise.
extern "C" void {fn_is_free}(void** inputs, void** outputs) {{
    Eigen::Matrix<double, crab_fused::kNQ, 1> q(
        Eigen::Map<const Eigen::Matrix<double, crab_fused::kNQ, 1>>(reinterpret_cast<const double*>(inputs[0])));
    Eigen::Matrix<double, crab_fused::kNSpheres * 4, 1> spheres_world =
        sym::{fk_func_name}<double>(q);

    double* out = reinterpret_cast<double*>(outputs[0]);
    out[0] = 1.0;

    for (int k = 0; k < crab_fused::kNumPairs; ++k) {{
        const auto& p = crab_fused::kSelfPairs[k];
        double dx = spheres_world(p.a * 4 + 0) - spheres_world(p.b * 4 + 0);
        double dy = spheres_world(p.a * 4 + 1) - spheres_world(p.b * 4 + 1);
        double dz = spheres_world(p.a * 4 + 2) - spheres_world(p.b * 4 + 2);
        double dist_sq = dx * dx + dy * dy + dz * dz;
        double r_sum = crab_fused::kRadii[p.a] + crab_fused::kRadii[p.b];
        if (dist_sq < r_sum * r_sum) {{
            out[0] = 0.0;
            return;
        }}
    }}

    for (int i = 0; i < crab_fused::kNSpheres; ++i) {{
        double z = spheres_world(i * 4 + 2);
        if (z < crab_fused::kRadii[i]) {{
            out[0] = 0.0;
            return;
        }}
    }}
}}

// inputs[0] = q (kNQ doubles), inputs[1] = margin (1 double). outputs[0][0] = cost.
extern "C" void {fn_cost}(void** inputs, void** outputs) {{
    Eigen::Matrix<double, crab_fused::kNQ, 1> q(
        Eigen::Map<const Eigen::Matrix<double, crab_fused::kNQ, 1>>(reinterpret_cast<const double*>(inputs[0])));
    double margin = *reinterpret_cast<const double*>(inputs[1]);
    Eigen::Matrix<double, crab_fused::kNSpheres * 4, 1> spheres_world =
        sym::{fk_func_name}<double>(q);

    double total_cost = 0.0;

    for (int k = 0; k < crab_fused::kNumPairs; ++k) {{
        const auto& p = crab_fused::kSelfPairs[k];
        double dx = spheres_world(p.a * 4 + 0) - spheres_world(p.b * 4 + 0);
        double dy = spheres_world(p.a * 4 + 1) - spheres_world(p.b * 4 + 1);
        double dz = spheres_world(p.a * 4 + 2) - spheres_world(p.b * 4 + 2);
        double dist = std::sqrt(dx * dx + dy * dy + dz * dz) -
                      (crab_fused::kRadii[p.a] + crab_fused::kRadii[p.b]);
        double penetration = margin - dist;
        if (penetration > 0.0) {{
            total_cost += (1.0 / 6.0) * penetration * penetration * penetration;
        }}
    }}

    for (int i = 0; i < crab_fused::kNSpheres; ++i) {{
        double dist = spheres_world(i * 4 + 2) - crab_fused::kRadii[i];
        double penetration = margin - dist;
        if (penetration > 0.0) {{
            total_cost += (1.0 / 6.0) * penetration * penetration * penetration;
        }}
    }}

    reinterpret_cast<double*>(outputs[0])[0] = total_cost;
}}

// inputs[0] = q (kNQ doubles), inputs[1] = margin (1 double).
// outputs[0][0] = cost, outputs[1] = grad (kNQ doubles).
extern "C" void {fn_cost_grad}(void** inputs, void** outputs) {{
    Eigen::Matrix<double, crab_fused::kNQ, 1> q(
        Eigen::Map<const Eigen::Matrix<double, crab_fused::kNQ, 1>>(reinterpret_cast<const double*>(inputs[0])));
    double margin = *reinterpret_cast<const double*>(inputs[1]);
    Eigen::Matrix<double, crab_fused::kNSpheres * 4, 1> spheres_world =
        sym::{fk_func_name}<double>(q);

    Eigen::Matrix<double, crab_fused::kNSpheres * 3, crab_fused::kNQ> jac;
    sym::{jac_func_name}<double>(q, &jac);

    Eigen::Matrix<double, crab_fused::kNQ, 1> grad =
        Eigen::Matrix<double, crab_fused::kNQ, 1>::Zero();
    double total_cost = 0.0;

    for (int k = 0; k < crab_fused::kNumPairs; ++k) {{
        const auto& p = crab_fused::kSelfPairs[k];
        Eigen::Vector3d pos_a(spheres_world(p.a * 4 + 0), spheres_world(p.a * 4 + 1), spheres_world(p.a * 4 + 2));
        Eigen::Vector3d pos_b(spheres_world(p.b * 4 + 0), spheres_world(p.b * 4 + 1), spheres_world(p.b * 4 + 2));
        Eigen::Vector3d diff = pos_a - pos_b;
        double dist_norm = diff.norm();
        if (dist_norm < 1e-6) continue;

        double dist = dist_norm - (crab_fused::kRadii[p.a] + crab_fused::kRadii[p.b]);
        double penetration = margin - dist;
        if (penetration > 0.0) {{
            total_cost += (1.0 / 6.0) * penetration * penetration * penetration;
            double dCost_dDist = -0.5 * penetration * penetration;
            Eigen::Vector3d dir = diff / dist_norm;
            auto J_a = jac.template block<3, crab_fused::kNQ>(p.a * 3, 0);
            auto J_b = jac.template block<3, crab_fused::kNQ>(p.b * 3, 0);
            grad += dCost_dDist * (dir.transpose() * (J_a - J_b)).transpose();
        }}
    }}

    for (int i = 0; i < crab_fused::kNSpheres; ++i) {{
        double dist = spheres_world(i * 4 + 2) - crab_fused::kRadii[i];
        double penetration = margin - dist;
        if (penetration > 0.0) {{
            total_cost += (1.0 / 6.0) * penetration * penetration * penetration;
            double dCost_dDist = -0.5 * penetration * penetration;
            auto J = jac.template block<3, crab_fused::kNQ>(i * 3, 0);
            grad += dCost_dDist * J.row(2).transpose();
        }}
    }}

    reinterpret_cast<double*>(outputs[0])[0] = total_cost;
    Eigen::Map<Eigen::Matrix<double, crab_fused::kNQ, 1>>(reinterpret_cast<double*>(outputs[1])) = grad;
}}
"""


def resolve_self_collision_pairs(spheres: list[dict], link_name_pairs) -> list[tuple[int, int]]:
    """Resolves (link_name_a, link_name_b) pairs into concrete sphere-index pairs,
    once, at codegen time -- the per-q loop in the fused kernel no longer needs to
    filter spheres by link_id at runtime."""
    pairs = []
    for link_a, link_b in link_name_pairs:
        idx_a = [i for i, s in enumerate(spheres) if s["link_name"] == link_a]
        idx_b = [i for i, s in enumerate(spheres) if s["link_name"] == link_b]
        for i in idx_a:
            for j in idx_b:
                pairs.append((i, j))
    return pairs


def build_fused_kernel_source(robot: Robot,
                               fn_is_free: str = FK_IS_FREE,
                               fn_cost: str = FK_COST,
                               fn_cost_grad: str = FK_COST_GRAD) -> dict:
    """Generates FK + its Jacobian, resolves this robot's collision-pair/radii data,
    and fills in FUSED_KERNEL_TEMPLATE. Returns {"source": str, "nq": int} ready for
    crab_jit.fused.build_fused_collision_kernel to compile and bind."""
    if not robot.allowed_collision_pairs:
        robot.compute_allowed_collision_pairs()

    generated = build_and_generate_function_with_jacobian(robot, forward_kinematics_spheres, "forward_kinematics_spheres")
    nq = generated["nq"]
    n_spheres = generated["n_out"] // 4

    spheres = robot_sphere_list(robot)
    pairs = resolve_self_collision_pairs(spheres, robot.allowed_collision_pairs)

    radii = ", ".join(repr(s["radius"]) for s in spheres)
    pairs_literal = ", ".join(f"{{{i}, {j}}}" for i, j in pairs)

    source = FUSED_KERNEL_TEMPLATE.format(
        fk_source=generated["value_source"],
        jac_source=generated["jac_source"],
        fk_func_name=generated["value_func_name"],
        jac_func_name=generated["jac_func_name"],
        nq=nq,
        n_spheres=n_spheres,
        radii=radii,
        n_pairs=len(pairs),
        pairs_array_size=max(len(pairs), 1),
        pairs=pairs_literal if pairs else "{0, 0}",
        fn_is_free=fn_is_free,
        fn_cost=fn_cost,
        fn_cost_grad=fn_cost_grad,
    )

    return {"source": source, "nq": nq}
