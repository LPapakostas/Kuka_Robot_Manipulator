#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import pickle
import matplotlib.pyplot as plt

# *==== CONSTANTS ====*

SIMULATION_TIME_READ_PATH = (
    os.getcwd() + "/kuka_manipulator/simulation/simulation_time.pickle"
)

EEF_POSITION_READ_PATH = (
    os.getcwd() + "/kuka_manipulator/simulation/actual_trajectory.pickle"
)

JOINTS_ANGLES_READ_PATH = (
    os.getcwd() + "/kuka_manipulator/simulation/joint_angles.pickle"
)

EEF_LINEAR_VELOCITY_READ_PATH = (
    os.getcwd() + "/kuka_manipulator/simulation/eef_linear_velocity.pickle"
)

JOINTS_ANGULAR_VELOCITY_READ_PATH = (
    os.getcwd() + "/kuka_manipulator/simulation/joint_angular_velocity.pickle"
)

POSITION_REFERENCE_TRAJECTORY_READ_PATH = (
    os.getcwd() + "/kuka_manipulator/simulation/reference_position_trajectory.pickle"
)

VELOCITY_REFERENCE_TRAJECTORY_READ_PATH = (
    os.getcwd() + "/kuka_manipulator/simulation/reference_velocity_trajectory.pickle"
)

ACCELERATION_REFERENCE_TRAJECTORY_READ_PATH = (
    os.getcwd()
    + "/kuka_manipulator/simulation/reference_acceleration_trajectory.pickle"
)

if __name__ == "__main__":
    print("Read saved values.....")

    with open(SIMULATION_TIME_READ_PATH, "rb") as f:
        simulation_time = pickle.load(f)

    with open(EEF_POSITION_READ_PATH, "rb") as f:
        eef_position = pickle.load(f)

    with open(JOINTS_ANGLES_READ_PATH, "rb") as f:
        joint_angles = pickle.load(f)

    with open(EEF_LINEAR_VELOCITY_READ_PATH, "rb") as f:
        eef_velocity = pickle.load(f)

    with open(JOINTS_ANGULAR_VELOCITY_READ_PATH, "rb") as f:
        joint_angular_velocity = pickle.load(f)

    with open(POSITION_REFERENCE_TRAJECTORY_READ_PATH, "rb") as f:
        position_reference = pickle.load(f)

    with open(VELOCITY_REFERENCE_TRAJECTORY_READ_PATH, "rb") as f:
        velocity_reference = pickle.load(f)

    # Decompose values
    eef_position_x = eef_position["x"]
    eef_position_y = eef_position["y"]
    eef_position_z = eef_position["z"]

    q1 = joint_angles["q1"]
    q2 = joint_angles["q2"]
    q3 = joint_angles["q3"]

    eef_velocity_x = eef_velocity["vx"]
    eef_velocity_y = eef_velocity["vy"]
    eef_velocity_z = eef_velocity["vz"]

    q1_dot = joint_angular_velocity["q1"]
    q2_dot = joint_angular_velocity["q2"]
    q3_dot = joint_angular_velocity["q3"]

    position_ref_x = position_reference["x"]
    position_ref_y = position_reference["y"]
    position_ref_z = position_reference["z"]

    velocity_ref_x = velocity_reference["x"]
    velocity_ref_y = velocity_reference["y"]
    velocity_ref_z = velocity_reference["z"]

    print("Start plotting data.....")

    # *==== Plot Reference values ====*

    # Create plot for Reference Trajectory (x)
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.plot(simulation_time, position_ref_x)
    ax.plot(simulation_time, eef_position_x)
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("Position [m]")
    ax.set_title("Reference Position Trajectory (X-axis)")
    ax.grid(which="major", color="black", linewidth=0.2)
    ax.grid(which="minor", color="black", linewidth=0.2)
    ax.minorticks_on()
    fig.set_dpi(200)
    plt.show()

    # # Create plot for Reference Trajectory (y)
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.plot(simulation_time, position_ref_y)
    ax.plot(simulation_time, eef_position_y)
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("Position [m]")
    ax.set_title("Reference Position Trajectory (Y-axis)")
    ax.grid(which="major", color="black", linewidth=0.2)
    ax.grid(which="minor", color="black", linewidth=0.2)
    ax.minorticks_on()
    fig.set_dpi(200)
    plt.show()

    # Create plot for Reference Trajectory (z)
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.plot(simulation_time, position_ref_z)
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("Position [m]")
    ax.set_title("Reference Position Trajectory (Z-axis)")
    ax.grid(which="major", color="black", linewidth=0.2)
    ax.grid(which="minor", color="black", linewidth=0.2)
    ax.minorticks_on()
    ax.set_ybound(upper=1.0, lower=0.6)
    fig.set_dpi(200)
    plt.show()

    # Create plot for Reference Velocity Trajectory (x)
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.plot(simulation_time, velocity_ref_x)
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("Velocity [m/s]")
    ax.set_title("Reference Velocity Trajectory (X-axis)")
    ax.grid(which="major", color="black", linewidth=0.2)
    ax.grid(which="minor", color="black", linewidth=0.2)
    ax.minorticks_on()
    fig.set_dpi(200)
    plt.show()

    # Create plot for Reference Velocity Trajectory (y)
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.plot(simulation_time, velocity_ref_y)
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("Velocity [m/s]")
    ax.set_title("Reference Velocity Trajectory (Y-axis)")
    ax.grid(which="major", color="black", linewidth=0.2)
    ax.grid(which="minor", color="black", linewidth=0.2)
    ax.minorticks_on()
    fig.set_dpi(200)
    plt.show()

    # Create plot for Reference Velocity Trajectory (z)
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.plot(simulation_time, velocity_ref_z)
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("Velocity [m/s]")
    ax.set_title("Reference Velocity Trajectory (Z-axis)")
    ax.grid(which="major", color="black", linewidth=0.2)
    ax.grid(which="minor", color="black", linewidth=0.2)
    ax.minorticks_on()
    fig.set_dpi(200)
    plt.show()

    # *==== Cartesian Space Plots ====*

    # # Create plot for EEF X position
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.plot(simulation_time, eef_position_x)
    ax.set_xlabel("time [sec]")
    ax.set_ylabel("Position [m]")
    ax.set_title("End Effector Position (X-axis)")
    ax.grid(which="major", color="black", linewidth=0.2)
    ax.grid(which="minor", color="black", linewidth=0.2)
    ax.minorticks_on()
    fig.set_dpi(200)
    plt.show()

    # Create plot for EEF Y position
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.plot(simulation_time, eef_position_y)
    ax.set_xlabel("time [sec]")
    ax.set_ylabel("Position [m]")
    ax.set_title("End Effector Position (Y-axis)")
    ax.grid(which="major", color="black", linewidth=0.2)
    ax.grid(which="minor", color="black", linewidth=0.2)
    ax.minorticks_on()
    fig.set_dpi(200)
    plt.show()

    # Create plot for EEF Z position
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.plot(simulation_time, eef_position_z)
    ax.set_xlabel("time [sec]")
    ax.set_ylabel("Position [m]")
    ax.set_title("End Effector Position (Z-axis)")
    ax.grid(which="major", color="black", linewidth=0.2)
    ax.grid(which="minor", color="black", linewidth=0.2)
    ax.minorticks_on()
    ax.set_ybound(upper=1.0, lower=0.6)
    fig.set_dpi(200)
    plt.show()

    # Create Plot for EEF Linear Velocity (X-axis)
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.plot(simulation_time, eef_velocity_x)
    ax.set_xlabel("time [sec]")
    ax.set_ylabel("Velocity [m/s]")
    ax.set_title("End Effector Velocity (X-axis)")
    ax.grid(which="major", color="black", linewidth=0.2)
    ax.grid(which="minor", color="black", linewidth=0.2)
    ax.minorticks_on()
    # ax.set_ybound(upper=0.5, lower=0.0)
    fig.set_dpi(200)
    plt.show()

    # Create Plot for EEF Linear Velocity (Y-axis)
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.plot(simulation_time, eef_velocity_y)
    ax.set_xlabel("time [sec]")
    ax.set_ylabel("Velocity [m/s]")
    ax.set_title("End Effector Velocity (Y-axis)")
    ax.grid(which="major", color="black", linewidth=0.2)
    ax.grid(which="minor", color="black", linewidth=0.2)
    ax.minorticks_on()
    # ax.set_ybound(upper=0.5, lower=0.0)
    fig.set_dpi(200)
    plt.show()

    # Create Plot for EEF Linear Velocity (Z-axis)
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.plot(simulation_time, eef_velocity_z)
    ax.set_xlabel("time [sec]")
    ax.set_ylabel("Velocity [m/s]")
    ax.set_title("End Effector Velocity (z-axis)")
    ax.grid(which="major", color="black", linewidth=0.2)
    ax.grid(which="minor", color="black", linewidth=0.2)
    ax.minorticks_on()
    ax.set_ybound(upper=-0.1, lower=0.1)
    fig.set_dpi(200)
    plt.show()

    # *==== Joint Space Plots ====*

    # Plot Joint 1 angles (rad)
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.plot(simulation_time, q1)
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("Angle [rad]")
    ax.set_title("Joint 1 Angle")
    ax.grid(which="major", color="black", linewidth=0.2)
    ax.grid(which="minor", color="black", linewidth=0.2)
    ax.minorticks_on()
    fig.set_dpi(200)
    plt.show()

    # Plot Joint 2 angles (rad)
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.plot(simulation_time, q2)
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("Angle [rad]")
    ax.set_title("Joint 2 Angle")
    ax.grid(which="major", color="black", linewidth=0.2)
    ax.grid(which="minor", color="black", linewidth=0.2)
    ax.minorticks_on()
    fig.set_dpi(200)
    plt.show()

    # Plot Joint 3 angles (rad)
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.plot(simulation_time, q3)
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("Angle [rad]")
    ax.set_title("Joint 3 Angle")
    ax.grid(which="major", color="black", linewidth=0.2)
    ax.grid(which="minor", color="black", linewidth=0.2)
    ax.minorticks_on()
    fig.set_dpi(200)
    plt.show()

    # Plot Joint 1 angular velocity (rad/s)
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.plot(simulation_time, q1_dot)
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("Angular Velocity [rad/sec]")
    ax.set_title("Joint 1 Angular Velocity")
    ax.grid(which="major", color="black", linewidth=0.2)
    ax.grid(which="minor", color="black", linewidth=0.2)
    ax.minorticks_on()
    fig.set_dpi(200)
    plt.show()

    # Plot Joint 2 angular velocity (rad/s)
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.plot(simulation_time, q2_dot)
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("Angular Velocity [rad/sec]")
    ax.set_title("Joint 2 Angular Velocity")
    ax.grid(which="major", color="black", linewidth=0.2)
    ax.grid(which="minor", color="black", linewidth=0.2)
    ax.minorticks_on()
    fig.set_dpi(200)
    plt.show()

    # Plot Joint 3 angular velocity (rad/s)
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.plot(simulation_time, q3_dot)
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("Angular Velocity [rad/sec]")
    ax.set_title("Joint 3 Angular Velocity")
    ax.grid(which="major", color="black", linewidth=0.2)
    ax.grid(which="minor", color="black", linewidth=0.2)
    ax.minorticks_on()
    fig.set_dpi(200)
    plt.show()
