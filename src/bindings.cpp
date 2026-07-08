#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
// #include <nanobind/eigen/matrix.h>
#include <nanobind/eigen/dense.h>
#include "collision_checker.hpp"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(crab_backend, m) {
    m.doc() = "C++ Accelerated Backend for CRAB (Compiled Robotics with Accelerated Backend)";

    // Bind Sphere struct
    nb::class_<crab::Sphere>(m, "Sphere")
        .def(nb::init<>())
        .def_rw("link_id", &crab::Sphere::link_id)
        .def_rw("index", &crab::Sphere::index)
        .def_rw("local_x", &crab::Sphere::local_x)
        .def_rw("local_y", &crab::Sphere::local_y)
        .def_rw("local_z", &crab::Sphere::local_z)
        .def_rw("radius", &crab::Sphere::radius);

    // Bind RobotSpherized struct
    nb::class_<crab::RobotSpherized>(m, "RobotSpherized")
        .def(nb::init<>())
        .def_rw("name", &crab::RobotSpherized::name)
        .def_rw("nq", &crab::RobotSpherized::nq)
        .def_rw("spheres", &crab::RobotSpherized::spheres)
        .def_rw("link_names", &crab::RobotSpherized::link_names)
        .def_rw("active_collision_pairs", &crab::RobotSpherized::active_collision_pairs);

    // Bind binary collision checker with early exit
    m.def("is_collision_free", &crab::is_collision_free, "q"_a, "robot"_a, "fk_fn_ptr"_a = 0,
          "Checks if the configuration q is collision free using an early-exiting loop. "
          "Pass a JIT-compiled FK function address (see crab_jit) as fk_fn_ptr to use a "
          "generated kernel instead of the built-in 7-DoF fallback.");

    // Bind differentiable collision cost function
    m.def("compute_collision_cost", &crab::compute_collision_cost, "q"_a, "robot"_a, "margin"_a = 0.05,
          "fk_fn_ptr"_a = 0,
          "Computes a smooth, differentiable scalar cost representing collision proximity.");

    m.def("forward_kinematics", &crab::forward_kinematics, "q"_a, "robot"_a, "fk_fn_ptr"_a = 0,
          "Computes the forward kinematics for the robot given joint configuration q.");

    // Bind differentiable collision cost with gradient, returning a Python tuple (cost, gradient_vector)
    m.def("compute_collision_cost_with_gradient", [](const Eigen::VectorXd& q,
                                                     const crab::RobotSpherized& robot,
                                                     double margin,
                                                     uintptr_t fk_fn_ptr,
                                                     uintptr_t jac_fn_ptr) {
        Eigen::VectorXd grad;
        double cost = crab::compute_collision_cost_with_gradient(q, robot, margin, grad, fk_fn_ptr, jac_fn_ptr);
        return std::make_pair(cost, grad);
    }, "q"_a, "robot"_a, "margin"_a = 0.05, "fk_fn_ptr"_a = 0, "jac_fn_ptr"_a = 0,
       "Computes both the differentiable collision cost and its analytical gradient w.r.t q. "
       "robot.nq determines the gradient size, so this works for any DoF count, not just 7.");
}
