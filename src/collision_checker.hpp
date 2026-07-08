#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <utility>
#include <Eigen/Core>

namespace crab {

/**
 * @brief Represents a collision sphere primitive on a robot link.
 */
struct Sphere {
    int link_id;     // ID of the link this sphere is attached to
    int index;       // Index of the sphere in the link's primitives
    double local_x;  // Local X offset in link frame
    double local_y;  // Local Y offset in link frame
    double local_z;  // Local Z offset in link frame
    double radius;   // Sphere radius
};

/**
 * @brief Represents a spherized robot in C++.
 */
struct RobotSpherized {
    std::string name;
    int nq; // Number of actuated joints
    std::vector<Sphere> spheres;
    std::vector<std::pair<std::string, std::string>> link_names;
    std::vector<std::pair<int, int>> active_collision_pairs; // Indices of links to check against each other
};

/**
 * @brief Computes the forward kinematics for the robot given joint configuration q.
 * 
 * @param q Active joint positions.
 * @param robot Spherized robot definition.
 * @param fk_fn_ptr Pointer to JIT-compiled FK function (optional).
 * @return Eigen::VectorXd Flat vector of sphere coordinates and radii.
 */
Eigen::VectorXd forward_kinematics(const Eigen::VectorXd& q, const RobotSpherized& robot, uintptr_t fk_fn_ptr = 0);

/**
 * @brief Binary collision checker. Returns true if the robot configuration is collision free.
 * Uses early exit to terminate check as soon as a single collision is found.
 * 
 * @param q Active joint positions.
 * @param robot Spherized robot definition.
 * @param fk_fn_ptr Pointer to JIT-compiled FK function (optional).
 * @return true if collision-free, false if in collision.
 */
bool is_collision_free(const Eigen::VectorXd& q, const RobotSpherized& robot, uintptr_t fk_fn_ptr = 0);

/**
 * @brief Computes the differentiable collision cost.
 * 
 * @param q Active joint positions.
 * @param robot Spherized robot definition.
 * @param margin Safety margin (epsilon).
 * @param fk_fn_ptr Pointer to JIT-compiled FK function (optional).
 * @return double Differentiable collision cost.
 */
double compute_collision_cost(const Eigen::VectorXd& q, const RobotSpherized& robot, double margin = 0.05, uintptr_t fk_fn_ptr = 0);

/**
 * @brief Computes the differentiable collision cost and its analytical gradient.
 * 
 * @param q Active joint positions.
 * @param robot Spherized robot definition.
 * @param margin Safety margin (epsilon).
 * @param grad Output gradient vector (w.r.t q).
 * @param fk_fn_ptr Pointer to JIT-compiled FK function (optional).
 * @param jac_fn_ptr Pointer to JIT-compiled Jacobian function (optional).
 * @return double Differentiable collision cost.
 */
double compute_collision_cost_with_gradient(const Eigen::VectorXd& q, 
                                            const RobotSpherized& robot, 
                                            double margin,
                                            Eigen::VectorXd& grad,
                                            uintptr_t fk_fn_ptr = 0,
                                            uintptr_t jac_fn_ptr = 0);

} // namespace crab
