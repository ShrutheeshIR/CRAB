import numpy as np
import random


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def norm(v):
    return np.linalg.norm(v)


def sphere_contains(ball, sphere, eps=1e-9):
    """
    ball   = (center, radius)
    sphere = (center, radius)
    """
    c, R = ball
    p, r = sphere
    return norm(p - c) + r <= R + eps


# -----------------------------------------------------------------------------
# Solve support set exactly
# -----------------------------------------------------------------------------

def ball_from_support(support):
    """
    Compute the unique smallest enclosing sphere determined by the support set.

    This implementation solves the KKT system with Newton iteration on λ.

    support = [(center, radius), ...]
    """

    if len(support) == 0:
        return np.zeros(3), -np.inf

    if len(support) == 1:
        c, r = support[0]
        return c.copy(), r

    dim = len(support[0][0])

    P = np.stack([p for p, _ in support])
    R = np.array([r for _, r in support])

    # Initial guess
    center = P.mean(axis=0)
    radius = max(norm(center - p) + r for p, r in support)

    for _ in range(100):

        d = center - P
        dist = np.linalg.norm(d, axis=1)
        dist = np.maximum(dist, 1e-12)

        f = dist + R - radius

        if np.max(np.abs(f)) < 1e-10:
            break

        J = d / dist[:, None]

        A = np.zeros((len(support) + 1, dim + 1))
        b = np.zeros(len(support) + 1)

        A[:-1, :-1] = J
        A[:-1, -1] = -1
        b[:-1] = -f

        # Gauge condition
        A[-1, -1] = 1
        b[-1] = 0

        x, *_ = np.linalg.lstsq(A, b, rcond=None)

        center += x[:-1]
        radius += x[-1]

    return center, radius


# -----------------------------------------------------------------------------
# Welzl recursion
# -----------------------------------------------------------------------------

def welzl(spheres, support, n):

    if n == 0 or len(support) == len(spheres[0][0]) + 1:
        return ball_from_support(support)

    sphere = spheres[n - 1]

    ball = welzl(spheres, support, n - 1)

    if sphere_contains(ball, sphere):
        return ball

    support.append(sphere)
    ball = welzl(spheres, support, n - 1)
    support.pop()

    return ball


# -----------------------------------------------------------------------------
# Public interface
# -----------------------------------------------------------------------------

def minimum_enclosing_sphere_of_spheres(spheres):
    """
    spheres : list of (center, radius)

    center : ndarray(d)
    radius : float
    """

    spheres = [(np.asarray(c, dtype=float), float(r)) for c, r in spheres]

    random.shuffle(spheres)

    return welzl(spheres, [], len(spheres))


# -----------------------------------------------------------------------------
# Example
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    spheres = [
        (np.array([0.0, 0.0, 0.0]), 1.0),
        (np.array([4.0, 0.0, 0.0]), 0.5),
        (np.array([1.0, 3.0, 0.0]), 0.7),
        (np.array([2.0, 1.0, 2.0]), 0.3),
    ]

    center, radius = minimum_enclosing_sphere_of_spheres(spheres)

    print("center =", center)
    print("radius =", radius)

    # Verify
    for p, r in spheres:
        assert norm(center - p) + r <= radius + 1e-8