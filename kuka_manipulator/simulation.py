#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sympy
import pickle
import os
from copy import deepcopy

from kuka_manipulator.inverse_kinematics import compute_inverse_kinematics
from kuka_manipulator.display import (
    read_forward_kinematics,
    read_jacobian,
    read_inverse_jacobian,
)
from kuka_manipulator.trajectory import create_three_phase_trajectory

SAVE = True

if __name__ == "__main__":
    # TODO: Parse files from yaml

    print("Initializing parameters....")
    T, t_step = 10, 0.01
    T_0, T_F = 0, 2 * T  # in [sec]
    time1 = np.arange(T_0, T, t_step)
    time2 = np.arange(T, T_F, t_step)
    time = np.concatenate((time1, time2))

    # Define initial/boundary conditions
    P_A = [0.7, -0.5, 0.8]
    P_B = [1.0, 0.5, 0.8]
    V_A = [0.0, 0.0, 0.0]
    V_B = [0.0, 0.0, 0.0]
    A_A = [0.0, 0.0, 0.0]
    A_B = [0.0, 0.0, 0.0]

    P_x_conditions_time1 = [P_A[0], P_B[0]]
    V_x_conditions_time1 = [V_A[0], V_B[0]]
    A_x_conditions_time1 = [A_A[0], A_B[0]]

    P_x_conditions_time2 = [P_B[0], P_A[0]]
    V_x_conditions_time2 = [V_B[0], V_A[0]]
    A_x_conditions_time2 = [A_B[0], A_A[0]]

    P_y_conditions_time1 = [P_A[1], P_B[1]]
    V_y_conditions_time1 = [V_A[1], V_B[1]]
    A_y_conditions_time1 = [A_A[1], A_B[1]]

    P_y_conditions_time2 = [P_B[1], P_A[1]]
    V_y_conditions_time2 = [V_B[1], V_A[1]]
    A_y_conditions_time2 = [A_B[1], A_A[1]]

    P_z_conditions_time1 = [P_A[2], P_B[2]]
    V_z_conditions_time1 = [V_A[2], V_B[2]]
    A_z_conditions_time1 = [A_A[2], A_B[2]]

    P_z_conditions_time2 = [P_B[2], P_A[2]]
    V_z_conditions_time2 = [V_B[2], V_A[2]]
    A_z_conditions_time2 = [A_B[2], A_A[2]]

    # Obtain all link positions in respect to q1, q2, q3
    q_syms = list(sympy.symbols("q1:4"))
    fk = read_forward_kinematics(subs=True)
    assert len(fk) == 6

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

    # *==== Compute Reference Trajectories ====*

    # Compute Reference (periodic) trajectory [x]
    x_reference_trajectories_time1 = create_three_phase_trajectory(
        P_x_conditions_time1, V_x_conditions_time1, A_x_conditions_time1, time1
    )
    x_reference_trajectories_time2 = create_three_phase_trajectory(
        P_x_conditions_time2, V_x_conditions_time2, A_x_conditions_time2, time2
    )

    traj_x_ref = deepcopy(x_reference_trajectories_time1["position"])
    traj_x_ref.extend(x_reference_trajectories_time2["position"])

    traj_vx_ref = deepcopy(x_reference_trajectories_time1["velocity"])
    traj_vx_ref.extend(x_reference_trajectories_time2["velocity"])

    # Compute Reference (periodic) trajectory [y]
    y_reference_trajectories_time1 = create_three_phase_trajectory(
        P_y_conditions_time1, V_y_conditions_time1, A_y_conditions_time1, time1
    )
    y_reference_trajectories_time2 = create_three_phase_trajectory(
        P_y_conditions_time2, V_y_conditions_time2, A_y_conditions_time2, time2
    )

    traj_y_ref = deepcopy(y_reference_trajectories_time1["position"])
    traj_y_ref.extend(y_reference_trajectories_time2["position"])

    traj_vy_ref = deepcopy(y_reference_trajectories_time1["velocity"])
    traj_vy_ref.extend(y_reference_trajectories_time2["velocity"])

    # Compute Reference (periodic) trajectory [z]
    z_reference_trajectories_time1 = create_three_phase_trajectory(
        P_z_conditions_time1, V_z_conditions_time1, A_z_conditions_time1, time1
    )
    z_reference_trajectories_time2 = create_three_phase_trajectory(
        P_z_conditions_time2, V_z_conditions_time2, A_z_conditions_time2, time2
    )

    traj_z_ref = deepcopy(z_reference_trajectories_time1["position"])
    traj_z_ref.extend(z_reference_trajectories_time2["position"])

    traj_vz_ref = deepcopy(z_reference_trajectories_time1["velocity"])
    traj_vz_ref.extend(z_reference_trajectories_time2["velocity"])

    # Compute `q`'s through inverse kinematics
    q1_vals, q2_vals, q3_vals = [], [], []
    assert len(traj_x_ref) == len(traj_y_ref) == len(traj_z_ref)
    for x, y, z in zip(traj_x_ref, traj_y_ref, traj_z_ref):
        current_joint_values = compute_inverse_kinematics(x, y, z)
        q1_vals.append(current_joint_values[0])  # q1
        q2_vals.append(current_joint_values[1])  # q2
        q3_vals.append(current_joint_values[2])  # q3

    assert len(q1_vals) == len(q2_vals) == len(q3_vals)

    # Forward Kinematics to obtain X,Y,Z Link 1 position
    link1_pos_x, link1_pos_y, link1_pos_z = [], [], []
    for q1, q2, q3 in zip(q1_vals, q2_vals, q3_vals):
        current_link1_pos = link1_pos_eq.subs("q1", q1)
        current_link1_pos = current_link1_pos.subs("q2", q2)
        current_link1_pos = current_link1_pos.subs("q3", q3)

        link1_pos_x.append(current_link1_pos[0])
        link1_pos_y.append(current_link1_pos[1])
        link1_pos_z.append(current_link1_pos[2])

    # Forward Kinematics to obtain X,Y,Z Link 2 position
    link2_pos_x, link2_pos_y, link2_pos_z = [], [], []
    for q1, q2, q3 in zip(q1_vals, q2_vals, q3_vals):
        current_link2_pos = link2_pos_eq.subs("q1", q1)
        current_link2_pos = current_link2_pos.subs("q2", q2)
        current_link2_pos = current_link2_pos.subs("q3", q3)

        link2_pos_x.append(current_link2_pos[0])
        link2_pos_y.append(current_link2_pos[1])
        link2_pos_z.append(current_link2_pos[2])

    # Forward Kinematics to obtain X,Y,Z Link 3 position
    link3_pos_x, link3_pos_y, link3_pos_z = [], [], []
    for q1, q2, q3 in zip(q1_vals, q2_vals, q3_vals):
        current_link3_pos = link3_pos_eq.subs("q1", q1)
        current_link3_pos = current_link3_pos.subs("q2", q2)
        current_link3_pos = current_link3_pos.subs("q3", q3)

        link3_pos_x.append(current_link3_pos[0])
        link3_pos_y.append(current_link3_pos[1])
        link3_pos_z.append(current_link3_pos[2])

    # Forward Kinematics to obtain X,Y,Z Link 4 position
    link4_pos_x, link4_pos_y, link4_pos_z = [], [], []
    for q1, q2, q3 in zip(q1_vals, q2_vals, q3_vals):
        current_link4_pos = link4_pos_eq.subs("q1", q1)
        current_link4_pos = current_link4_pos.subs("q2", q2)
        current_link4_pos = current_link4_pos.subs("q3", q3)

        link4_pos_x.append(current_link4_pos[0])
        link4_pos_y.append(current_link4_pos[1])
        link4_pos_z.append(current_link4_pos[2])

    # Forward Kinematics to obtain X,Y,Z Link 5 position
    link5_pos_x, link5_pos_y, link5_pos_z = [], [], []
    for q1, q2, q3 in zip(q1_vals, q2_vals, q3_vals):
        current_link5_pos = link5_pos_eq.subs("q1", q1)
        current_link5_pos = current_link5_pos.subs("q2", q2)
        current_link5_pos = current_link5_pos.subs("q3", q3)

        link5_pos_x.append(current_link5_pos[0])
        link5_pos_y.append(current_link5_pos[1])
        link5_pos_z.append(current_link5_pos[2])

    # Forward Kinematics to obtain X,Y,Z EEF position
    traj_x, traj_y, traj_z = [], [], []
    for q1, q2, q3 in zip(q1_vals, q2_vals, q3_vals):
        current_p_eef = eef_pos_eq.subs("q1", q1)
        current_p_eef = current_p_eef.subs("q2", q2)
        current_p_eef = current_p_eef.subs("q3", q3)

        traj_x.append(current_p_eef[0])
        traj_y.append(current_p_eef[1])
        traj_z.append(current_p_eef[2])

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

    if SAVE:
        print("Saving values ....")

        # Save reference position trajectory
        refence_position_trajectory = {
            "x": traj_x_ref,
            "y": traj_y_ref,
            "z": traj_z_ref,
        }
        reference_position_trajectory_save_path = (
            os.getcwd()
            + "/kuka_manipulator/simulation/reference_position_trajectory.pickle"
        )
        with open(reference_position_trajectory_save_path, "wb") as outf:
            outf.write(pickle.dumps(refence_position_trajectory))

        # Save reference velocity trajectory
        refence_velocity_trajectory = {
            "x": traj_vx_ref,
            "y": traj_vy_ref,
            "z": traj_vz_ref,
        }
        reference_velocity_trajectory_save_path = (
            os.getcwd()
            + "/kuka_manipulator/simulation/reference_velocity_trajectory.pickle"
        )
        with open(reference_velocity_trajectory_save_path, "wb") as outf:
            outf.write(pickle.dumps(refence_velocity_trajectory))

        # Save joint angles
        q_values = {"q1": q1_vals, "q2": q2_vals, "q3": q3_vals}
        joint_angles_save_path = (
            os.getcwd() + "/kuka_manipulator/simulation/joint_angles.pickle"
        )
        with open(joint_angles_save_path, "wb") as outf:
            outf.write(pickle.dumps(q_values))

        # Save Link 1 position
        link1_position = {"x": link1_pos_x, "y": link1_pos_y, "z": link1_pos_z}
        link1_position_save_path = (
            os.getcwd() + "/kuka_manipulator/simulation/link1_position.pickle"
        )
        with open(link1_position_save_path, "wb") as outf:
            outf.write(pickle.dumps(link1_position))

        # Save Link 2 position
        link2_position = {"x": link2_pos_x, "y": link2_pos_y, "z": link2_pos_z}
        link2_position_save_path = (
            os.getcwd() + "/kuka_manipulator/simulation/link2_position.pickle"
        )
        with open(link2_position_save_path, "wb") as outf:
            outf.write(pickle.dumps(link2_position))

        # Save Link 3 position
        link3_position = {"x": link3_pos_x, "y": link3_pos_y, "z": link3_pos_z}
        link3_position_save_path = (
            os.getcwd() + "/kuka_manipulator/simulation/link3_position.pickle"
        )
        with open(link3_position_save_path, "wb") as outf:
            outf.write(pickle.dumps(link3_position))

        # Save Link 4 position
        link4_position = {"x": link4_pos_x, "y": link4_pos_y, "z": link4_pos_z}
        link4_position_save_path = (
            os.getcwd() + "/kuka_manipulator/simulation/link4_position.pickle"
        )
        with open(link4_position_save_path, "wb") as outf:
            outf.write(pickle.dumps(link4_position))

        # Save Link 5 position
        link5_position = {"x": link5_pos_x, "y": link5_pos_y, "z": link5_pos_z}
        link5_position_save_path = (
            os.getcwd() + "/kuka_manipulator/simulation/link5_position.pickle"
        )
        with open(link5_position_save_path, "wb") as outf:
            outf.write(pickle.dumps(link5_position))

        # Save End effector position
        actual_trajectory = {"x": traj_x, "y": traj_y, "z": traj_z}
        actual_trajectory_save_path = (
            os.getcwd() + "/kuka_manipulator/simulation/actual_trajectory.pickle"
        )
        with open(actual_trajectory_save_path, "wb") as outf:
            outf.write(pickle.dumps(actual_trajectory))

        # Save End Effector Velocities
        end_effector_linear_velocity = {"vx": traj_vx, "vy": traj_vy, "vz": traj_vz}
        end_effector_linear_velocity_save_path = (
            os.getcwd() + "/kuka_manipulator/simulation/eef_linear_velocity.pickle"
        )
        with open(end_effector_linear_velocity_save_path, "wb") as outf:
            outf.write(pickle.dumps(end_effector_linear_velocity))

        # Save Joint angular velocities
        joint_angular_velocity = {"q1": qdot1_vals, "q2": qdot2_vals, "q3": qdot3_vals}
        joint_angular_velocity_save_path = (
            os.getcwd() + "/kuka_manipulator/simulation/joint_angular_velocity.pickle"
        )
        with open(joint_angular_velocity_save_path, "wb") as outf:
            outf.write(pickle.dumps(joint_angular_velocity))

        # Save simulation time
        simulation_time_save_path = (
            os.getcwd() + "/kuka_manipulator/simulation/simulation_time.pickle"
        )
        with open(simulation_time_save_path, "wb") as outf:
            outf.write(pickle.dumps(time))
