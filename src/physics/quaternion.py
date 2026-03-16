import math
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

@nb.njit(cache=True, parallel=True)
def update_orientations(q, omega, dt):
    """
    Update an array of quaternions based on angular velocities.
    q: (N, 4) array of quaternions
    omega: (N, 3) array of angular velocities
    """
    new_q = np.empty_like(q)
    n = q.shape[0]
    for i in nb.prange(n):
        ox = omega[i, 0]
        oy = omega[i, 1]
        oz = omega[i, 2]

        omega_norm = math.sqrt(ox * ox + oy * oy + oz * oz)
        theta = omega_norm * dt
        if theta < 1e-12 or omega_norm < 1e-18:
            dq_w = 1.0
            dq_x = 0.0
            dq_y = 0.0
            dq_z = 0.0
        else:
            half_theta = 0.5 * theta
            dq_w = math.cos(half_theta)
            s = math.sin(half_theta) / omega_norm
            dq_x = ox * s
            dq_y = oy * s
            dq_z = oz * s

        q_w = q[i, 0]
        q_x = q[i, 1]
        q_y = q[i, 2]
        q_z = q[i, 3]

        w = dq_w * q_w - dq_x * q_x - dq_y * q_y - dq_z * q_z
        x = dq_w * q_x + dq_x * q_w + dq_y * q_z - dq_z * q_y
        y = dq_w * q_y - dq_x * q_z + dq_y * q_w + dq_z * q_x
        z = dq_w * q_z + dq_x * q_y - dq_y * q_x + dq_z * q_w

        norm = math.sqrt(w * w + x * x + y * y + z * z)
        if norm > 1e-9:
            inv_norm = 1.0 / norm
            new_q[i, 0] = w * inv_norm
            new_q[i, 1] = x * inv_norm
            new_q[i, 2] = y * inv_norm
            new_q[i, 3] = z * inv_norm
        else:
            new_q[i, 0] = 1.0
            new_q[i, 1] = 0.0
            new_q[i, 2] = 0.0
            new_q[i, 3] = 0.0
    return new_q
