#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sympy
from pprint import pprint


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

    coef_matrix = np.array([[1, t_0, t_0 ^ (2), t_0 ^ (3), t_0 ^ (4), t_0 ^ (5)],
                           [0, 1, 2 * t_0, 3 * t_0 ^
                               (2), 4 * t_0 ^ (3), 5 * t_0 ^ (4)],
                           [0, 0, 2, 6*t_0, 12 * t_0 ^ (2), 20 * t_0 ^ (3)],
                           [1, t_f, t_f ^ (2), t_f ^ (
                               3), t_f ^ (4), t_f ^ (5)],
                           [0, 1, 2 * t_f, 3 * t_f ^
                               (2), 4 * t_f ^ (3), 5 * t_f ^ (4)],
                           [0, 0, 2, 6*t_f, 12 * t_f ^ (2), 20 * t_f ^ (3)]])
    boundary_conditions = np.array([p_0, v_0, a_0, p_f, v_f, a_f]).T

    parameters = np.linalg.inv(coef_matrix)  @ boundary_conditions

    return parameters


def position_trajectory_expr(coeffs):
    """
    """
    assert (len(coeffs) == 6)

    t = sympy.symbols("t")
    coeffs_round = [round(x, 2) for x in coeffs]
    a0, a1, a2 = coeffs_round[0], coeffs_round[1], coeffs_round[2]
    a3, a4, a5 = coeffs_round[3], coeffs_round[4], coeffs_round[5]
    g = a0 + a1 * t + a2 * sympy.Pow(t, 2) + \
        a3*sympy.Pow(t, 3) + a4*sympy.Pow(t, 4) + a5*sympy.Pow(t, 5)

    return g


def velocity_trajectory_expr(coeffs):
    """
    """
    assert (len(coeffs) == 6)

    t = sympy.symbols("t")
    coeffs_round = [round(x, 2) for x in coeffs]
    a0, a1, a2 = coeffs_round[0], coeffs_round[1], coeffs_round[2]
    a3, a4, a5 = coeffs_round[3], coeffs_round[4], coeffs_round[5]
    g = a1 + 2*a2 * t + \
        3*a3*sympy.Pow(t, 2) + 4*a4*sympy.Pow(t, 3) + 5*a5*sympy.Pow(t, 4)

    return g


if (__name__ == "__main__"):

    P_A = [1.0, 1.0, 1.0]
    V_A = [1.0, 0.0, 0.2]
    A_A = [0.1, 0.05, 0.03]

    P_B = [1.5, 1.5, 1.0]
    V_B = [2.0, 0.1, 0.3]
    A_B = [0.0, 0.0, 0.0]

    t_0, t_f = 0, 20  # in [sec]

    x_vals, x_dot_vals, x_ddot_vals = [P_A[0], P_B[0]], [
        V_A[0], V_B[0]], [A_A[0], A_B[0]]
    y_vals, y_dot_vals, y_ddot_vals = [P_A[1], P_B[1]], [
        V_A[1], V_B[1]], [A_A[1], A_B[1]]
    z_vals, z_dot_vals, z_ddot_vals = [P_A[2], P_B[2]], [
        V_A[2], V_B[2]], [A_A[2], A_B[2]]

    x_parameters = generate_trajectory_coeffs(
        x_vals, x_dot_vals, x_ddot_vals, t_0, t_f)
    y_parameters = generate_trajectory_coeffs(
        y_vals, y_dot_vals, y_ddot_vals, t_0, t_f)
    z_parameters = generate_trajectory_coeffs(
        z_vals, z_dot_vals, z_ddot_vals, t_0, t_f)

    print(f"x(t) = {position_trajectory_expr(x_parameters)}")
    print("")
    print(f"y(t) = {position_trajectory_expr(y_parameters)}")
    print("")
    print(f"z(t) = {position_trajectory_expr(z_parameters)}")
