#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

# TODO: Add docstrings


def generate_trajectory_coeffs(positions, velocities, accelerations, start_time, stop_time):
    """
    """

    assert (len(positions) == 2)
    assert (len(velocities) == 2)
    assert (len(accelerations) == 2)

    t_0, t_f = start_time, stop_time
    p_0, p_f = positions[0], positions[1]
    v_0, v_f = velocities[0], velocities[1]
    a_0, a_f = accelerations[0], accelerations[1]

    coef_matrix = np.array([[1, t_0, t_0**(2), t_0**(3), t_0**(4), t_0**(5)],
                           [0, 1, 2 * t_0, 3 * t_0 **
                               (2), 4 * t_0 ** (3), 5 * t_0 ** (4)],
                           [0, 0, 2, 6 * t_0, 12 * t_0 **
                               (2), 20 * t_0 ** (3)],
                           [1, t_f, t_f ** (2), t_f ** (
                               3), t_f ** (4), t_f ** (5)],
                           [0, 1, 2 * t_f, 3 * t_f **
                               (2), 4 * t_f ** (3), 5 * t_f ** (4)],
                           [0, 0, 2, 6 * t_f, 12 * t_f ** (2), 20 * t_f ** (3)]])
    boundary_conditions = np.array([p_0, v_0, a_0, p_f, v_f, a_f]).T

    parameters = np.linalg.inv(coef_matrix)  @ boundary_conditions

    return parameters


def create_three_phase_trajectory(positions, velocities, accelerations, time, delta=0.2):
    """
    """
    # Decompose values
    t0, tf = time[0], time[-1]
    g0, gf = positions[0], positions[-1]
    g0_dot, gf_dot = velocities[0], velocities[-1]
    g0_ddot, gf_ddot = accelerations[0], accelerations[-1]

    #
    g02_dot = (gf - g0) / (tf - t0 - 2 * delta)
    g02 = g0 + delta * g02_dot
    gf2 = g0 + (tf - t0 - 3 * delta) * g02_dot
    position_trajectory, velocity_trajectory, acceleration_trajectory = [], [], []

    # Generate coefficients
    coeffs_phase1 = generate_trajectory_coeffs(
        [g0, g02], [g0_dot, g02_dot], [g0_ddot, 0], t0, t0 + 2 * delta)
    coeffs_phase2 = generate_trajectory_coeffs(
        [g02, gf2], [g02_dot, g02_dot], [0, 0], t0 + 2 * delta, tf - t0 - 2 * delta)
    coeffs_phase3 = generate_trajectory_coeffs(
        [gf2, gf], [g02_dot, gf_dot], [0, gf_ddot], tf - t0 - 2 * delta, tf)
    assert(len(coeffs_phase1) == len(coeffs_phase2) == len(coeffs_phase3) == 6)

    # Compute position trajectory
    for ts in time:
        if(ts < t0 + 2 * delta):
            current_val = coeffs_phase1[0] + coeffs_phase1[1] * ts + coeffs_phase1[2] * pow(
                ts, 2) + coeffs_phase1[3] * pow(ts, 3) + coeffs_phase1[4] * pow(ts, 4) + coeffs_phase1[5] * pow(ts, 5)
        elif(ts < tf - t0 - 2 * delta):
            current_val = coeffs_phase2[0] + coeffs_phase2[1] * ts
        else:
            current_val = coeffs_phase3[0] + coeffs_phase3[1] * ts + coeffs_phase3[2] * pow(
                ts, 2) + coeffs_phase3[3] * pow(ts, 3) + coeffs_phase3[4] * pow(ts, 4) + coeffs_phase3[5] * pow(ts, 5)
        position_trajectory.append(current_val)

    # Compute velocity trajectory
    for ts in time:
        if(ts < t0 + 2 * delta):
            current_val = coeffs_phase1[1] + 2 * coeffs_phase1[2] * ts + 6 * coeffs_phase1[3] * pow(
                ts, 2) + 12 * coeffs_phase1[4] * pow(ts, 3) + 20 * coeffs_phase1[5] * pow(ts, 4)
        elif(ts < tf - t0 - 2 * delta):
            current_val = coeffs_phase2[1]
        else:
            current_val = coeffs_phase1[1] + 2 * coeffs_phase1[2] * ts + 6 * coeffs_phase1[3] * pow(
                ts, 2) + 12 * coeffs_phase1[4] * pow(ts, 3) + 20 * coeffs_phase1[5] * pow(ts, 4)
        velocity_trajectory.append(current_val)

    # Compute acceleration trajectory
    for ts in time:
        if(ts < t0 + 2 * delta):
            current_val = 2 * coeffs_phase1[2] + 6 * coeffs_phase1[3] * ts + 12 * \
                coeffs_phase1[4] * pow(ts, 2) + 20 * \
                coeffs_phase1[5] * pow(ts, 3)
        elif(ts < tf - t0 - 2 * delta):
            current_val = 0.0
        else:
            current_val = 2 * coeffs_phase1[2] + 6 * coeffs_phase1[3] * ts + 12 * \
                coeffs_phase1[4] * pow(ts, 2) + 20 * \
                coeffs_phase1[5] * pow(ts, 3)
        acceleration_trajectory.append(current_val)

    trajectories = {
        "position": position_trajectory,
        "velocity": velocity_trajectory,
        "acceleration": acceleration_trajectory
    }

    return trajectories
