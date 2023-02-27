#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import sympy
import matplotlib.pyplot as plt
import pickle
import os
from matplotlib.ticker import AutoMinorLocator

from kuka_manipulator.inverse_kinematics import compute_inverse_kinematics
from kuka_manipulator.display import read_forward_kinematics, read_jacobian, read_inverse_jacobian

if (__name__ == "__main__"):

    print("Initializing parameters....")
    T_0, T_F, t_step = 0, 20, 0.01  # in [sec]
    time = np.arange(T_0, T_F, t_step)

    # Define initial_conditions
    P_A = [0.7, -0.4, 0.95]
    P_B = [0.7, 0.4, 0.95]
    V = [0.2, 0.3, 0.1]

    P_x_start, P_x_final = [P_A[0], P_B[0]]
    P_y_start, P_y_final = [P_A[1], P_B[1]]
    P_z_start, P_z_final = [P_A[2], P_B[2]]
    V_x, V_y, V_z = V[0], V[1], V[2]

    # Obtain all link positions in respect to q1, q2, q3
    q_syms = list(sympy.symbols("q1:4"))
    fk = read_forward_kinematics(subs=True)
    assert(len(fk) == 6)

    link1_pos_eq = (fk[0])[:3, 3]
    link2_pos_eq = (fk[1])[:3, 3]
    link3_pos_eq = (fk[2])[:3, 3]
    link4_pos_eq = (fk[3])[:3, 3]
    link5_pos_eq = (fk[4])[:3, 3]
    eef_pos_eq = (fk[-1])[0:3, 3]

    # Obtain Jacobian and inverse Jacobian
    j = read_jacobian(subs=True)
    fk_diff = j[0:3, 0:3]
    j_inv = read_inverse_jacobian(subs=True)
    ik_diff = j_inv[0:3, 0:3]

    print("Initializing Task-Space Trajectory Computations ....")
    traj_x_ref, traj_y_ref, traj_z_ref = [P_x_start], [P_y_start], [P_z_start]
    lambda_x = (P_x_final - P_x_start) / T_F
    lambda_y = (P_y_final - P_y_start) / T_F
    lambda_z = (P_z_final - P_z_start) / T_F

    # Compute (Reference) EEF Position Trajectories
    for _ in range(1, len(time)):
        last_x, last_y, last_z = traj_x_ref[-1], traj_y_ref[-1], traj_z_ref[-1]
        traj_x_ref.append(last_x + lambda_x * t_step)
        traj_y_ref.append(last_y + lambda_y * t_step)
        traj_z_ref.append(last_z + lambda_z * t_step)

    # Compute `q`'s through inverse kinematics
    q1_vals, q2_vals, q3_vals = [], [], []
    assert(len(traj_x_ref) == len(traj_y_ref) == len(traj_z_ref))
    for (x, y, z) in zip(traj_x_ref, traj_y_ref, traj_z_ref):
        current_joint_values = compute_inverse_kinematics(x, y, z)
        q1_vals.append(current_joint_values[0])  # q1
        q2_vals.append(current_joint_values[1])  # q2
        q3_vals.append(current_joint_values[2])  # q3

    assert(len(q1_vals) == len(q2_vals) == len(q3_vals))

    # Forward Kinematics to obtain X,Y,Z Link 1 position
    link1_pos_x, link1_pos_y, link1_pos_z = [], [], []
    for (q1, q2, q3) in zip(q1_vals, q2_vals, q3_vals):
        current_link1_pos = link1_pos_eq.subs("q1", q1)
        current_link1_pos = current_link1_pos.subs("q2", q2)
        current_link1_pos = current_link1_pos.subs("q3", q3)

        link1_pos_x.append(current_link1_pos[0])
        link1_pos_y.append(current_link1_pos[1])
        link1_pos_z.append(current_link1_pos[2])

    # Forward Kinematics to obtain X,Y,Z Link 2 position
    link2_pos_x, link2_pos_y, link2_pos_z = [], [], []
    for (q1, q2, q3) in zip(q1_vals, q2_vals, q3_vals):
        current_link2_pos = link2_pos_eq.subs("q1", q1)
        current_link2_pos = current_link2_pos.subs("q2", q2)
        current_link2_pos = current_link2_pos.subs("q3", q3)

        link2_pos_x.append(current_link2_pos[0])
        link2_pos_y.append(current_link2_pos[1])
        link2_pos_z.append(current_link2_pos[2])

    # Forward Kinematics to obtain X,Y,Z Link 3 position
    link3_pos_x, link3_pos_y, link3_pos_z = [], [], []
    for (q1, q2, q3) in zip(q1_vals, q2_vals, q3_vals):
        current_link3_pos = link3_pos_eq.subs("q1", q1)
        current_link3_pos = current_link3_pos.subs("q2", q2)
        current_link3_pos = current_link3_pos.subs("q3", q3)

        link3_pos_x.append(current_link3_pos[0])
        link3_pos_y.append(current_link3_pos[1])
        link3_pos_z.append(current_link3_pos[2])

    # Forward Kinematics to obtain X,Y,Z Link 4 position
    link4_pos_x, link4_pos_y, link4_pos_z = [], [], []
    for (q1, q2, q3) in zip(q1_vals, q2_vals, q3_vals):
        current_link4_pos = link4_pos_eq.subs("q1", q1)
        current_link4_pos = current_link4_pos.subs("q2", q2)
        current_link4_pos = current_link4_pos.subs("q3", q3)

        link4_pos_x.append(current_link4_pos[0])
        link4_pos_y.append(current_link4_pos[1])
        link4_pos_z.append(current_link4_pos[2])

    # Forward Kinematics to obtain X,Y,Z Link 5 position
    link5_pos_x, link5_pos_y, link5_pos_z = [], [], []
    for (q1, q2, q3) in zip(q1_vals, q2_vals, q3_vals):
        current_link5_pos = link5_pos_eq.subs("q1", q1)
        current_link5_pos = current_link5_pos.subs("q2", q2)
        current_link5_pos = current_link5_pos.subs("q3", q3)

        link5_pos_x.append(current_link5_pos[0])
        link5_pos_y.append(current_link5_pos[1])
        link5_pos_z.append(current_link5_pos[2])

    # Forward Kinematics to obtain X,Y,Z EEF position
    traj_x, traj_y, traj_z = [], [], []
    for (q1, q2, q3) in zip(q1_vals, q2_vals, q3_vals):
        current_p_eef = eef_pos_eq.subs("q1", q1)
        current_p_eef = current_p_eef.subs("q2", q2)
        current_p_eef = current_p_eef.subs("q3", q3)

        traj_x.append(current_p_eef[0])
        traj_y.append(current_p_eef[1])
        traj_z.append(current_p_eef[2])

    # Compute (Reference) EEF Velocity Trajectories
    traj_vx_ref, traj_vy_ref, traj_vz_ref = [], [], []
    for _ in range(0, len(time)):
        traj_vx_ref.append(V_x)
        traj_vy_ref.append(V_y)
        traj_vz_ref.append(V_z)

    # Compute joint velocities through inverse Jacobian
    qdot1_vals, qdot2_vals, qdot3_vals = [], [], []
    for i in range(0, len(time)):
        q1, q2, q3 = q1_vals[i], q2_vals[i], q3_vals[i]
        vx, vy, vz = traj_vx_ref[i], traj_vy_ref[i], traj_vz_ref[i]
        v = np.array([vx, vy, vz])
        qdot = sympy.Matrix(ik_diff @ v)
        qdot_vals = qdot.subs("q1", q1)
        qdot_vals = qdot_vals.subs("q2", q2)
        qdot_vals = qdot_vals.subs("q3", q3)
        qdot_vals = [float(x) for x in qdot_vals]

        qdot1_vals.append(qdot_vals[0])
        qdot2_vals.append(qdot_vals[1])
        qdot3_vals.append(qdot_vals[2])

    # Compute velocities through Jacobian multiplication
    traj_vx, traj_vy, traj_vz = [], [], []
    for i in range(0, len(time)):
        q1, q2, q3 = q1_vals[i], q2_vals[i], q3_vals[i]
        qdot1, qdot2, qdot3 = qdot1_vals[i], qdot2_vals[i], qdot3_vals[i]
        qdot = np.array([qdot1, qdot2, qdot3])
        v = sympy.Matrix(fk_diff @ qdot)
        v = ((v.subs("q1", q1)).subs("q2", q2)).subs("q3", q3)

        traj_vx.append(v[0])
        traj_vy.append(v[1])
        traj_vz.append(v[2])

    print("Saving values ....")

    # Save reference trajectory into pickle file
    refence_trajectory = {
        "x": traj_x_ref,
        "y": traj_y_ref,
        "z": traj_z_ref
    }
    reference_trajectory_save_path = os.getcwd(
    ) + "/kuka_manipulator/simulation/reference_trajectory.pickle"
    with open(reference_trajectory_save_path, 'wb') as outf:
        outf.write(pickle.dumps(refence_trajectory))

    q_values = {
        "q1": q1_vals,
        "q2": q2_vals,
        "q3": q3_vals
    }
    joint_angles_save_path = os.getcwd(
    ) + "/kuka_manipulator/simulation/joint_angles.pickle"
    with open(joint_angles_save_path, 'wb') as outf:
        outf.write(pickle.dumps(q_values))

    # Save Link 1 position
    link1_position = {
        "x": link1_pos_x,
        "y": link1_pos_y,
        "z": link1_pos_z
    }
    link1_position_save_path = os.getcwd(
    ) + "/kuka_manipulator/simulation/link1_position.pickle"
    with open(link1_position_save_path, 'wb') as outf:
        outf.write(pickle.dumps(link1_position))

    # Save Link 2 position
    link2_position = {
        "x": link2_pos_x,
        "y": link2_pos_y,
        "z": link2_pos_z
    }
    link2_position_save_path = os.getcwd(
    ) + "/kuka_manipulator/simulation/link2_position.pickle"
    with open(link2_position_save_path, 'wb') as outf:
        outf.write(pickle.dumps(link2_position))

    # Save Link 3 position
    link3_position = {
        "x": link3_pos_x,
        "y": link3_pos_y,
        "z": link3_pos_z
    }
    link3_position_save_path = os.getcwd(
    ) + "/kuka_manipulator/simulation/link3_position.pickle"
    with open(link3_position_save_path, 'wb') as outf:
        outf.write(pickle.dumps(link3_position))

    # Save Link 4 position
    link4_position = {
        "x": link4_pos_x,
        "y": link4_pos_y,
        "z": link4_pos_z
    }
    link4_position_save_path = os.getcwd(
    ) + "/kuka_manipulator/simulation/link4_position.pickle"
    with open(link4_position_save_path, 'wb') as outf:
        outf.write(pickle.dumps(link4_position))

    # Save Link 5 position
    link5_position = {
        "x": link5_pos_x,
        "y": link5_pos_y,
        "z": link5_pos_z
    }
    link5_position_save_path = os.getcwd(
    ) + "/kuka_manipulator/simulation/link5_position.pickle"
    with open(link5_position_save_path, 'wb') as outf:
        outf.write(pickle.dumps(link5_position))

    # Save End effector position
    actual_trajectory = {
        "x": traj_x,
        "y": traj_y,
        "z": traj_z
    }
    actual_trajectory_save_path = os.getcwd(
    ) + "/kuka_manipulator/simulation/actual_trajectory.pickle"
    with open(actual_trajectory_save_path, 'wb') as outf:
        outf.write(pickle.dumps(actual_trajectory))

    # Save simulation time
    simulation_time_save_path = os.getcwd(
    ) + "/kuka_manipulator/simulation/simulation_time.pickle"
    with open(simulation_time_save_path, 'wb') as outf:
        outf.write(pickle.dumps(time))

    # *==== Cartesian Space Plots ====*

    # Create plot for EEF X position
    # fig, ax = plt.subplots(figsize=(15, 15))
    # ax.plot(time, traj_x_ref, label="Reference")
    # ax.plot(time, traj_x, label="Actual")
    # ax.set_xlabel('time [sec]')
    # ax.set_ylabel('Position [m]')
    # ax.set_title("End Effector Position (X-axis)")
    # ax.grid(which='minor', color='black', linewidth=0.2)
    # ax.tick_params(which='minor', bottom=False, left=False)
    # ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    # ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    # ax.legend()
    # fig.set_dpi(200)
    # plt.show()

    # Create plot for EEF Y position
    # fig, ax = plt.subplots(figsize=(15, 15))
    # ax.plot(time, traj_y_ref, label="Reference")
    # ax.plot(time, traj_y, label="Actual")
    # ax.set_xlabel('time [sec]')
    # ax.set_ylabel('Position [m]')
    # ax.set_title("End Effector Position (Y-axis)")
    # ax.grid(which='minor', color='black', linewidth=0.2)
    # ax.tick_params(which='minor', bottom=False, left=False)
    # ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    # ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    # ax.legend()
    # fig.set_dpi(200)
    # plt.show()

    # Create plot for EEF Z position
    # fig, ax = plt.subplots(figsize=(15, 15))
    # ax.plot(time, traj_z_ref, label="Reference")
    # ax.plot(time, traj_z, label="Actual")
    # ax.set_xlabel('time [sec]')
    # ax.set_ylabel('Position [m]')
    # ax.set_title("End Effector Position (Z-axis)")
    # ax.grid(which='minor', color='black', linewidth=0.2)
    # ax.tick_params(which='minor', bottom=False, left=False)
    # ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    # ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    # ax.legend()
    # fig.set_dpi(200)
    # plt.show()

    # Create Plot for EEF Linear Velocity (X-axis)
    # fig, ax = plt.subplots(figsize=(15, 15))
    # ax.plot(time, traj_vx_ref, label="Reference")
    # ax.plot(time, traj_vx, label="Actual")
    # ax.set_xlabel('time [sec]')
    # ax.set_ylabel('Velocity [m/s]')
    # ax.set_title("End Effector Velocity (X-axis)")
    # ax.grid(which='minor', color='black', linewidth=0.2)
    # ax.tick_params(which='minor', bottom=False, left=False)
    # ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    # ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    # ax.legend()
    # fig.set_dpi(200)
    # plt.show()

    # Create Plot for EEF Linear Velocity (Y-axis)
    # fig, ax = plt.subplots(figsize=(15, 15))
    # ax.plot(time, traj_vy_ref, label="Reference")
    # ax.plot(time, traj_vy, label="Actual")
    # ax.set_xlabel('time [sec]')
    # ax.set_ylabel('Velocity [m/s]')
    # ax.set_title("End Effector Velocity (Y-axis)")
    # ax.grid(which='minor', color='black', linewidth=0.2)
    # ax.tick_params(which='minor', bottom=False, left=False)
    # ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    # ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    # ax.legend()
    # fig.set_dpi(200)
    # plt.show()

    # Create Plot for EEF Linear Velocity (Z-axis)
    # fig, ax = plt.subplots(figsize=(15, 15))
    # ax.plot(time, traj_vz_ref, label="Reference")
    # ax.plot(time, traj_vz, label="Actual")
    # ax.set_xlabel('time [sec]')
    # ax.set_ylabel('Velocity [m/s]')
    # ax.set_title("End Effector Velocity (Z-axis)")
    # ax.grid(which='minor', color='black', linewidth=0.2)
    # ax.tick_params(which='minor', bottom=False, left=False)
    # ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    # ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    # ax.legend()
    # fig.set_dpi(200)
    # plt.show()

    # *==== Joint Space Plots ====*

    # Plot Joint 1 angles (rad)
    # fig, ax = plt.subplots(figsize=(15, 15))
    # ax.plot(time, q1_vals)
    # ax.set_xlabel('Time [sec]')
    # ax.set_ylabel('Angle [rad]]')
    # ax.set_title("Joint 1 Angle")
    # ax.grid(which='minor', color='black', linewidth=0.2)
    # ax.tick_params(which='minor', bottom=False, left=False)
    # ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    # ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    # fig.set_dpi(200)
    # plt.show()

    # Plot Joint 2 angles (rad)
    # fig, ax = plt.subplots(figsize=(15, 15))
    # ax.plot(time, q2_vals)
    # ax.set_xlabel('Time [sec]')
    # ax.set_ylabel('Angle [rad]]')
    # ax.set_title("Joint 2 Angle")
    # ax.grid(which='minor', color='black', linewidth=0.2)
    # ax.tick_params(which='minor', bottom=False, left=False)
    # ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    # ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    # fig.set_dpi(200)
    # plt.show()

    # Plot Joint 3 angles (rad)
    # fig, ax = plt.subplots(figsize=(15, 15))
    # ax.plot(time, q3_vals)
    # ax.set_xlabel('Time [sec]')
    # ax.set_ylabel('Angle [rad]]')
    # ax.set_title("Joint 3 Angle")
    # ax.grid(which='minor', color='black', linewidth=0.2)
    # ax.tick_params(which='minor', bottom=False, left=False)
    # ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    # ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    # fig.set_dpi(200)
    # plt.show()

    # Plot Joint 1 angular velocity (rad/s)
    # fig, ax = plt.subplots(figsize=(15, 15))
    # ax.plot(time, qdot1_vals)
    # ax.set_xlabel('Time [sec]')
    # ax.set_ylabel('Angular Velocity [rad/sec]')
    # ax.set_title("Joint 1 Angular Velocity")
    # ax.grid(which='minor', color='black', linewidth=0.2)
    # ax.tick_params(which='minor', bottom=False, left=False)
    # ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    # ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    # fig.set_dpi(200)
    # plt.show()

    # Plot Joint 2 angular velocity (rad/s)
    # fig, ax = plt.subplots(figsize=(15, 15))
    # ax.plot(time, qdot2_vals)
    # ax.set_xlabel('Time [sec]')
    # ax.set_ylabel('Angular Velocity [rad/sec]')
    # ax.set_title("Joint 2 Angular Velocity")
    # ax.grid(which='minor', color='black', linewidth=0.2)
    # ax.tick_params(which='minor', bottom=False, left=False)
    # ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    # ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    # fig.set_dpi(200)
    # plt.show()

    # Plot Joint 3 angular velocity (rad/s)
    # fig, ax = plt.subplots(figsize=(15, 15))
    # ax.plot(time, qdot3_vals)
    # ax.set_xlabel('Time [sec]')
    # ax.set_ylabel('Angular Velocity [rad/sec]')
    # ax.set_title("Joint 3 Angular Velocity")
    # ax.grid(which='minor', color='black', linewidth=0.2)
    # ax.tick_params(which='minor', bottom=False, left=False)
    # ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    # ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    # fig.set_dpi(200)
    # plt.show()

    # Create Plot for X-Y trajectory of EEF
    # fig, ax = plt.subplots(figsize=(15, 15))
    # ax.plot(traj_x_ref, traj_y_ref, label="Reference")
    # ax.plot(traj_x, traj_y, label="Actual")
    # ax.set_xlabel('Position (X) [m]')
    # ax.set_ylabel('Position (Y) [m]')
    # ax.set_title("End Effector Trajectory")
    # ax.grid(which='minor', color='black', linewidth=0.2)
    # ax.tick_params(which='minor', bottom=False, left=False)
    # ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    # ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    # ax.legend()
    # fig.set_dpi(200)
    # plt.show()
