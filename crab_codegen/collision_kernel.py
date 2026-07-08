"""Generates a single, fully fused JIT translation unit per robot: FK + the
collision-pair loop + cost/gradient math all in one compilation unit, so -O3 can
inline and fuse across what used to be a function-pointer call boundary
(crab_codegen.jit_wrapper's FK/Jacobian sub-kernels, called from the fixed, separately
compiled loop in src/collision_checker.cpp).

Sphere radii and which sphere-index pairs to check are all known once the robot model
is loaded (they don't depend on q), so they're baked in as compile-time constants
instead of being looked up at runtime through crab::RobotSpherized.
"""

from crusty_model.robot import Robot
from crab_codegen.sphere_order import robot_sphere_list

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

extern "C" int {fn_is_free}(const double* q_ptr) {{
    Eigen::Matrix<double, crab_fused::kNQ, 1> q(
        Eigen::Map<const Eigen::Matrix<double, crab_fused::kNQ, 1>>(q_ptr));
    Eigen::Matrix<double, crab_fused::kNSpheres * 4, 1> spheres_world =
        sym::{fk_func_name}<double>(q);

    for (int k = 0; k < crab_fused::kNumPairs; ++k) {{
        const auto& p = crab_fused::kSelfPairs[k];
        double dx = spheres_world(p.a * 4 + 0) - spheres_world(p.b * 4 + 0);
        double dy = spheres_world(p.a * 4 + 1) - spheres_world(p.b * 4 + 1);
        double dz = spheres_world(p.a * 4 + 2) - spheres_world(p.b * 4 + 2);
        double dist_sq = dx * dx + dy * dy + dz * dz;
        double r_sum = crab_fused::kRadii[p.a] + crab_fused::kRadii[p.b];
        if (dist_sq < r_sum * r_sum) {{
            return 0;
        }}
    }}

    for (int i = 0; i < crab_fused::kNSpheres; ++i) {{
        double z = spheres_world(i * 4 + 2);
        if (z < crab_fused::kRadii[i]) {{
            return 0;
        }}
    }}
    return 1;
}}

extern "C" double {fn_cost}(const double* q_ptr, double margin) {{
    Eigen::Matrix<double, crab_fused::kNQ, 1> q(
        Eigen::Map<const Eigen::Matrix<double, crab_fused::kNQ, 1>>(q_ptr));
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

    return total_cost;
}}

extern "C" double {fn_cost_grad}(const double* q_ptr, double margin, double* grad_ptr) {{
    Eigen::Matrix<double, crab_fused::kNQ, 1> q(
        Eigen::Map<const Eigen::Matrix<double, crab_fused::kNQ, 1>>(q_ptr));
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

    Eigen::Map<Eigen::Matrix<double, crab_fused::kNQ, 1>>(grad_ptr) = grad;
    return total_cost;
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


def build_fused_kernel_source(generated: dict, robot: Robot,
                               fn_is_free: str = FK_IS_FREE,
                               fn_cost: str = FK_COST,
                               fn_cost_grad: str = FK_COST_GRAD) -> str:
    if not robot.allowed_collision_pairs:
        robot.compute_allowed_collision_pairs()

    spheres = robot_sphere_list(robot)
    pairs = resolve_self_collision_pairs(spheres, robot.allowed_collision_pairs)

    radii = ", ".join(repr(s["radius"]) for s in spheres)
    pairs_literal = ", ".join(f"{{{i}, {j}}}" for i, j in pairs)

    return FUSED_KERNEL_TEMPLATE.format(
        fk_source=generated["fk_source"],
        jac_source=generated["jac_source"],
        fk_func_name=generated["fk_func_name"],
        jac_func_name=generated["jac_func_name"],
        nq=generated["nq"],
        n_spheres=len(spheres),
        radii=radii,
        n_pairs=len(pairs),
        pairs_array_size=max(len(pairs), 1),
        pairs=pairs_literal if pairs else "{0, 0}",
        fn_is_free=fn_is_free,
        fn_cost=fn_cost,
        fn_cost_grad=fn_cost_grad,
    )
