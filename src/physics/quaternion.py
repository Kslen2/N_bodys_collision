import numpy as np
import numba as nb

@nb.njit(cache=True)
def quat_multiply(q1, q2):
    """
    Multiply two quaternions.
    q = [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])

@nb.njit(cache=True)
def quat_normalize(q):
    """
    Normalize a quaternion.
    """
    norm = np.sqrt(np.sum(q**2))
    if norm > 1e-9:
        return q / norm
    return np.array([1.0, 0.0, 0.0, 0.0])

@nb.njit(cache=True)
def quat_from_angular_velocity(omega, dt):
    """
    Create a rotation quaternion from an angular velocity vector.
    omega: [omega_x, omega_y, omega_z]
    """
    omega_norm = np.sqrt(np.sum(omega**2))
    theta = omega_norm * dt
    if theta < 1e-12 or omega_norm < 1e-18:
        return np.array([1.0, 0.0, 0.0, 0.0])

    half_theta = 0.5 * theta
    w = np.cos(half_theta)
    s = np.sin(half_theta) / omega_norm
    x = omega[0] * s
    y = omega[1] * s
    z = omega[2] * s
    return np.array([w, x, y, z])

@nb.njit(cache=True)
def update_orientations(q, omega, dt):
    """
    Update an array of quaternions based on angular velocities.
    q: (N, 4) array of quaternions
    omega: (N, 3) array of angular velocities
    """
    new_q = np.empty_like(q)
    for i in nb.prange(q.shape[0]):
        dq = quat_from_angular_velocity(omega[i], dt)
        new_q[i] = quat_normalize(quat_multiply(dq, q[i]))
    return new_q
