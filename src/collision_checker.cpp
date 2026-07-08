#include "collision_checker.hpp"
#include "collision_helpers.hpp"
#include <iostream>

#ifdef HAVE_GENERATED_KINEMATICS
// Direct inclusion of SymForce-generated headers when available
#include <forward_kinematics_generated.h>
#include <forward_kinematics_jacobian_generated.h>
#else
namespace sym {

// Placeholder fallback when compilation occurs before SymForce codegen.
// Emulates SymForce signatures.
template <typename Scalar>
void ForwardKinematicsGenerated(const Eigen::Matrix<Scalar, 7, 1>& q, Eigen::Matrix<Scalar, Eigen::Dynamic, 1>* res) {
    // We assume 40 spheres for compilation layout
    res->resize(40 * 4);
    res->setZero();
    for (int i = 0; i < 40; ++i) {
        (*res)(i * 4 + 0) = 0.0; // x
        (*res)(i * 4 + 1) = 0.0; // y
        (*res)(i * 4 + 2) = static_cast<double>(q(i % 7)) * 0.1; // z (dummy dependency)
        (*res)(i * 4 + 3) = 0.1; // radius
    }
}

template <typename Scalar>
void ForwardKinematicsJacobianGenerated(const Eigen::Matrix<Scalar, 7, 1>& q, Eigen::Matrix<Scalar, Eigen::Dynamic, 1>* res) {
    res->resize(40 * 3 * 7);
    res->setZero();
}

} // namespace sym
#endif

namespace crab {

Eigen::VectorXd forward_kinematics(const Eigen::VectorXd& q, const RobotSpherized& robot, uintptr_t fk_fn_ptr) {
    Eigen::Matrix<double, Eigen::Dynamic, 1> spheres_world(robot.spheres.size() * 4);
    if (fk_fn_ptr != 0) {
        typedef void (*JITFunc)(void** inputs, void** outputs);
        auto fk_fn = reinterpret_cast<JITFunc>(fk_fn_ptr);
        void* inputs[1] = { (void*)q.data() };
        void* outputs[1] = { (void*)spheres_world.data() };
        fk_fn(inputs, outputs);
    } else {
        if (q.size() != 7) {
            throw std::runtime_error("Fallback static FK only supports 7-DoF. Provide JIT function pointer for other sizes.");
        }
        Eigen::Matrix<double, 7, 1> q_7 = q;
        sym::ForwardKinematicsGenerated(q_7, &spheres_world);
    }
    return spheres_world;
}

bool is_collision_free(const Eigen::VectorXd& q, const RobotSpherized& robot, uintptr_t fk_fn_ptr) {
    // 1. Compute sphere positions in the world frame via generated FK.
    Eigen::Matrix<double, Eigen::Dynamic, 1> spheres_world(robot.spheres.size() * 4);
    if (fk_fn_ptr != 0) {
        typedef void (*JITFunc)(void** inputs, void** outputs);
        auto fk_fn = reinterpret_cast<JITFunc>(fk_fn_ptr);
        void* inputs[1] = { (void*)q.data() };
        void* outputs[1] = { (void*)spheres_world.data() };
        fk_fn(inputs, outputs);
    } else {
        if (q.size() != 7) {
            throw std::runtime_error("Fallback static FK only supports 7-DoF. Provide JIT function pointer for other sizes.");
        }
        Eigen::Matrix<double, 7, 1> q_7 = q;
        sym::ForwardKinematicsGenerated(q_7, &spheres_world);
    }

    // Helper lambda to get a sphere's world coordinates from the flat output vector
    auto get_sphere_world = [&](int sphere_idx) {
        int offset = sphere_idx * 4;
        return Eigen::Vector3d(spheres_world(offset), spheres_world(offset + 1), spheres_world(offset + 2));
    };

    // 2. Loop over collision pairs and early exit as soon as any pair collides
    for (const auto& pair : robot.active_collision_pairs) {
        int link_a_id = pair.first;
        int link_b_id = pair.second;

        // Iterate through all spheres to find those belonging to link_a and link_b
        for (size_t i = 0; i < robot.spheres.size(); ++i) {
            if (robot.spheres[i].link_id != link_a_id) continue;
            Eigen::Vector3d pos_a = get_sphere_world(i);
            double r_a = robot.spheres[i].radius;

            for (size_t j = 0; j < robot.spheres.size(); ++j) {
                if (robot.spheres[j].link_id != link_b_id) continue;
                Eigen::Vector3d pos_b = get_sphere_world(j);
                double r_b = robot.spheres[j].radius;

                // Evaluate binary collision
                double dx = pos_a.x() - pos_b.x();
                double dy = pos_a.y() - pos_b.y();
                double dz = pos_a.z() - pos_b.z();
                double dist_sq = dx * dx + dy * dy + dz * dz;
                double r_sum = r_a + r_b;

                if (dist_sq < r_sum * r_sum) {
                    return false; // EARLY EXIT: Collision detected!
                }
            }
        }
    }

    // 3. Environment checks
    for (size_t i = 0; i < robot.spheres.size(); ++i) {
        Eigen::Vector3d pos = get_sphere_world(i);
        double r = robot.spheres[i].radius;
        if (sphere_environment_collision_checker(pos.x(), pos.y(), pos.z(), r) > 0.5f) {
            return false; // EARLY EXIT: Collision with environment!
        }
    }

    return true; // Collision free
}

double compute_collision_cost(const Eigen::VectorXd& q, const RobotSpherized& robot, double margin, uintptr_t fk_fn_ptr) {
    Eigen::Matrix<double, Eigen::Dynamic, 1> spheres_world(robot.spheres.size() * 4);
    if (fk_fn_ptr != 0) {
        typedef void (*JITFunc)(void** inputs, void** outputs);
        auto fk_fn = reinterpret_cast<JITFunc>(fk_fn_ptr);
        void* inputs[1] = { (void*)q.data() };
        void* outputs[1] = { (void*)spheres_world.data() };
        fk_fn(inputs, outputs);
    } else {
        if (q.size() != 7) {
            throw std::runtime_error("Fallback static FK only supports 7-DoF. Provide JIT function pointer for other sizes.");
        }
        Eigen::Matrix<double, 7, 1> q_7 = q;
        sym::ForwardKinematicsGenerated(q_7, &spheres_world);
    }

    auto get_sphere_world = [&](int sphere_idx) {
        int offset = sphere_idx * 4;
        return Eigen::Vector3d(spheres_world(offset), spheres_world(offset + 1), spheres_world(offset + 2));
    };

    double total_cost = 0.0;

    // Self-collisions
    for (const auto& pair : robot.active_collision_pairs) {
        int link_a_id = pair.first;
        int link_b_id = pair.second;

        for (size_t i = 0; i < robot.spheres.size(); ++i) {
            if (robot.spheres[i].link_id != link_a_id) continue;
            Eigen::Vector3d pos_a = get_sphere_world(i);
            double r_a = robot.spheres[i].radius;

            for (size_t j = 0; j < robot.spheres.size(); ++j) {
                if (robot.spheres[j].link_id != link_b_id) continue;
                Eigen::Vector3d pos_b = get_sphere_world(j);
                double r_b = robot.spheres[j].radius;

                double dist = (pos_a - pos_b).norm() - (r_a + r_b);
                double penetration = margin - dist;
                if (penetration > 0.0) {
                    // Cubic penalty for smooth second derivatives (C2 continuity)
                    total_cost += (1.0 / 6.0) * std::pow(penetration, 3);
                }
            }
        }
    }

    // Environment collisions
    for (size_t i = 0; i < robot.spheres.size(); ++i) {
        Eigen::Vector3d pos = get_sphere_world(i);
        double r = robot.spheres[i].radius;
        
        double dist = pos.z() - r;
        double penetration = margin - dist;
        if (penetration > 0.0) {
            total_cost += (1.0 / 6.0) * std::pow(penetration, 3);
        }
    }

    return total_cost;
}

double compute_collision_cost_with_gradient(const Eigen::VectorXd& q, 
                                            const RobotSpherized& robot, 
                                            double margin,
                                            Eigen::VectorXd& grad,
                                            uintptr_t fk_fn_ptr,
                                            uintptr_t jac_fn_ptr) {
    grad.setZero(robot.nq);
    
    // 1. Get sphere positions in the world frame
    Eigen::Matrix<double, Eigen::Dynamic, 1> spheres_world(robot.spheres.size() * 4);
    if (fk_fn_ptr != 0) {
        typedef void (*JITFunc)(void** inputs, void** outputs);
        auto fk_fn = reinterpret_cast<JITFunc>(fk_fn_ptr);
        void* inputs[1] = { (void*)q.data() };
        void* outputs[1] = { (void*)spheres_world.data() };
        fk_fn(inputs, outputs);
    } else {
        if (q.size() != 7) {
            throw std::runtime_error("Fallback static FK only supports 7-DoF. Provide JIT function pointer for other sizes.");
        }
        Eigen::Matrix<double, 7, 1> q_7 = q;
        sym::ForwardKinematicsGenerated(q_7, &spheres_world);
    }

    auto get_sphere_world = [&](int sphere_idx) {
        int offset = sphere_idx * 4;
        return Eigen::Vector3d(spheres_world(offset), spheres_world(offset + 1), spheres_world(offset + 2));
    };

    // 2. Get the Jacobian of all spheres with respect to joints q.
    Eigen::Matrix<double, Eigen::Dynamic, 1> jacobian_flat(robot.spheres.size() * 3 * robot.nq);
    if (jac_fn_ptr != 0) {
        typedef void (*JITFunc)(void** inputs, void** outputs);
        auto jac_fn = reinterpret_cast<JITFunc>(jac_fn_ptr);
        void* inputs[1] = { (void*)q.data() };
        void* outputs[1] = { (void*)jacobian_flat.data() };
        jac_fn(inputs, outputs);
    } else {
        if (q.size() != 7) {
            throw std::runtime_error("Fallback static Jacobian only supports 7-DoF. Provide JIT function pointer for other sizes.");
        }
        Eigen::Matrix<double, 7, 1> q_7 = q;
        sym::ForwardKinematicsJacobianGenerated(q_7, &jacobian_flat);
    }

    // Maps flat Jacobian memory to Eigen matrices
    auto get_sphere_jacobian = [&](int sphere_idx) {
        // Offset for this sphere: sphere_idx * 3 rows, robot.nq columns
        return Eigen::Map<const Eigen::Matrix<double, 3, Eigen::Dynamic>>(
            &jacobian_flat(sphere_idx * 3 * robot.nq),
            3, robot.nq
        );
    };

    double total_cost = 0.0;

    // Self-collisions
    for (const auto& pair : robot.active_collision_pairs) {
        int link_a_id = pair.first;
        int link_b_id = pair.second;

        for (size_t i = 0; i < robot.spheres.size(); ++i) {
            if (robot.spheres[i].link_id != link_a_id) continue;
            Eigen::Vector3d pos_a = get_sphere_world(i);
            double r_a = robot.spheres[i].radius;
            auto J_a = get_sphere_jacobian(i);

            for (size_t j = 0; j < robot.spheres.size(); ++j) {
                if (robot.spheres[j].link_id != link_b_id) continue;
                Eigen::Vector3d pos_b = get_sphere_world(j);
                double r_b = robot.spheres[j].radius;
                auto J_b = get_sphere_jacobian(j);

                Eigen::Vector3d diff = pos_a - pos_b;
                double dist_norm = diff.norm();
                
                // Avoid division by zero when spheres are exactly concentric
                if (dist_norm < 1e-6) continue; 

                double dist = dist_norm - (r_a + r_b);
                double penetration = margin - dist;
                if (penetration > 0.0) {
                    // Cost = 1/6 * penetration^3
                    total_cost += (1.0 / 6.0) * std::pow(penetration, 3);

                    // Derivative of cost w.r.t distance is: -0.5 * penetration^2
                    double dCost_dDist = -0.5 * std::pow(penetration, 2);

                    // Unit direction vector from sphere B to sphere A
                    Eigen::Vector3d dir = diff / dist_norm;

                    // Jacobian of distance w.r.t q:
                    // d(dist)/dq = dir^T * (J_a - J_b)
                    Eigen::Matrix<double, 1, Eigen::Dynamic> dDist_dq = dir.transpose() * (J_a - J_b);

                    // Accumulate gradient: dCost/dq = dCost/dDist * dDist/dq
                    grad += dCost_dDist * dDist_dq.transpose();
                }
            }
        }
    }

    // Environment collisions (e.g. ground plane z = 0)
    for (size_t i = 0; i < robot.spheres.size(); ++i) {
        Eigen::Vector3d pos = get_sphere_world(i);
        double r = robot.spheres[i].radius;
        auto J = get_sphere_jacobian(i);

        double dist = pos.z() - r;
        double penetration = margin - dist;
        if (penetration > 0.0) {
            // Cost = 1/6 * penetration^3
            total_cost += (1.0 / 6.0) * std::pow(penetration, 3);

            // Derivative w.r.t distance:
            double dCost_dDist = -0.5 * std::pow(penetration, 2);

            // Jacobian of distance w.r.t q:
            // Since dist = pos.z - r, d(dist)/dq is the 3rd row of the sphere's Jacobian J (Z-axis derivative)
            Eigen::Matrix<double, 1, Eigen::Dynamic> dDist_dq = J.row(2);

            grad += dCost_dDist * dDist_dq.transpose();
        }
    }

    return total_cost;
}

} // namespace crab
