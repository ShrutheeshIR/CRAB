#include <Eigen/Core>

namespace sym {

// Placeholder implementation of the generated Forward Kinematics function.
// This is to allow compilation of the nanobind module before the actual SymForce
// generator has run. Once the generator script runs, this file can be deleted
// or excluded from compilation, and replaced with the generated code.
void forward_kinematics_generated(const Eigen::Matrix<double, 7, 1>& q, double* primitive_xyzr_out) {
    // Fill with zero position and some default radius for each sphere.
    // Assuming a test layout where we have some spheres (e.g., 20 spheres).
    // Lay out 20 spheres, each with x=0, y=0, z=0, r=0.1
    for (int i = 0; i < 40; ++i) {
        primitive_xyzr_out[i * 4 + 0] = 0.0; // x
        primitive_xyzr_out[i * 4 + 1] = 0.0; // y
        primitive_xyzr_out[i * 4 + 2] = q(i % 7) * 0.1; // z (dummy dependency on q)
        primitive_xyzr_out[i * 4 + 3] = 0.1; // r
    }
}

// Placeholder implementation of the generated Forward Kinematics Jacobian.
void forward_kinematics_jacobian_generated(const Eigen::Matrix<double, 7, 1>& q, double* jacobian_out) {
    // Lay out zero Jacobian values.
    // Matrix size: (N_spheres * 3) x 7. Assuming 40 spheres max.
    for (int i = 0; i < 40 * 3 * 7; ++i) {
        jacobian_out[i] = 0.0;
    }
}

} // namespace sym
