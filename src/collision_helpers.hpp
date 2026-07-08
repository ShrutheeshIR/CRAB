#pragma once
#include <cmath>
#include <algorithm>

/**
 * @brief Custom C++ definition for the SymPy function 'sphere_sphere_collision_checker'.
 * 
 * In binary/differentiable evaluations, this function determines if two spheres are in collision.
 * 
 * @param x1, y1, z1, r1 Position and radius of the first sphere.
 * @param x2, y2, z2, r2 Position and radius of the second sphere.
 * @return float 1.0f if colliding, 0.0f otherwise (or smooth distance/cost).
 */
inline float sphere_sphere_collision_checker(float x1, float y1, float z1, float r1,
                                             float x2, float y2, float z2, float r2) {
    float dx = x1 - x2;
    float dy = y1 - y2;
    float dz = z1 - z2;
    float dist_sq = dx * dx + dy * dy + dz * dz;
    float r_sum = r1 + r2;
    
    if (dist_sq < r_sum * r_sum) {
        return 1.0f; // Collision
    }
    return 0.0f; // No collision
}

/**
 * @brief Custom C++ definition for the SymPy function 'sphere_environment_collision_checker'.
 * 
 * In binary/differentiable evaluations, this checks a sphere against environmental obstacles.
 * 
 * @param x, y, z, r Position and radius of the sphere.
 * @return float Collision metric (e.g. penetration distance or binary value).
 */
inline float sphere_environment_collision_checker(float x, float y, float z, float r) {
    // Simple ground plane collision check (e.g. z < r is a collision)
    if (z < r) {
        return 1.0f;
    }
    return 0.0f;
}
