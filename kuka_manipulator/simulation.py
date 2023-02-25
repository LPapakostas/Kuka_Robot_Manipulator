#!/usr/bin/python
# -*- coding: utf-8 -*-
import sympy
import numpy as np
import matplotlib.pyplot as plt

from kuka_manipulator.trajectory import generate_trajectory_coeffs, position_trajectory_expr, velocity_trajectory_expr

if (__name__ == "__main__"):
    # Define initial_conditions
    P_A = [0.7, -0.4, 0.95]
    V_A = [1.0, 0.0, 0.2]
    A_A = [0.1, 0.05, 0.03]

    P_B = [0.75, 0.2, 0.95]
    V_B = [2.0, 0.1, 0.3]
    A_B = [0.0, 0.0, 0.0]

    T_0, T_F, t_step = 0, 5, 0.01  # in [sec]

    P_x_conditions = [P_A[0], P_B[0]]
    P_y_conditions = [P_A[1], P_B[1]]
    P_z_conditions = [P_A[2], P_B[2]]

    V_x_conditions = [V_A[0], V_B[0]]
    V_y_conditions = [V_A[1], V_B[1]]
    V_z_conditions = [V_A[2], V_B[2]]

    A_x_conditions = [A_A[0], A_B[0]]
    A_y_conditions = [A_A[1], A_B[1]]
    A_z_conditions = [A_A[2], A_B[2]]

    # Generate Trajectories
    traj_x_coeffs = generate_trajectory_coeffs(
        P_x_conditions, V_x_conditions, A_x_conditions, T_0, T_F)
    traj_x = position_trajectory_expr(traj_x_coeffs)
    traj_vx = velocity_trajectory_expr(traj_x_coeffs)

    traj_y_coeffs = generate_trajectory_coeffs(
        P_y_conditions, V_y_conditions, A_y_conditions, T_0, T_F)
    traj_y = position_trajectory_expr(traj_y_coeffs)
    traj_vy = velocity_trajectory_expr(traj_y_coeffs)

    traj_z_coeffs = generate_trajectory_coeffs(
        P_z_conditions, V_z_conditions, A_z_conditions, T_0, T_F)
    traj_z = position_trajectory_expr(traj_z_coeffs)
    traj_vz = velocity_trajectory_expr(traj_z_coeffs)

    # Generate time and trajectory points
    t = sympy.symbols("t")
    time = np.arange(T_0, T_F, t_step)
    traj_x_pts = np.array([traj_x.subs(t, time_val) for time_val in time])
    vel_x_pts = np.array([traj_vx.subs(t, time_val) for time_val in time])
    traj_y_pts = np.array([traj_y.subs(t, time_val) for time_val in time])
    vel_y_pts = np.array([traj_vy.subs(t, time_val) for time_val in time])
    traj_y_pts = np.array([traj_z.subs(t, time_val) for time_val in time])
    vel_y_pts = np.array([traj_vz.subs(t, time_val) for time_val in time])
    print("Trajectory points are generated...")

    # Create plot for Position and Velocity
    fig, axs = plt.subplots(3, 2)
    axs[0, 0].plot(time, traj_x_pts)
    axs[0, 0].set_xlabel('t [sec]')
    axs[0, 0].set_ylabel('Position (x-axis) (m)')
    axs[0, 0].xaxis.grid()
    axs[0, 0].yaxis.grid()
    fig.set_dpi(200)
    plt.show()
