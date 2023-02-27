#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# *==== CONSTANTS ====*

SIMULATION_TIME_READ_PATH = os.getcwd(
) + "/kuka_manipulator/simulation/simulation_time.pickle"

EEF_POSITION_READ_PATH = os.getcwd(
) + "/kuka_manipulator/simulation/actual_trajectory.pickle"

JOINTS_ANGLES_READ_PATH = os.getcwd(
) + "/kuka_manipulator/simulation/joint_angles.pickle"

if (__name__ == "__main__"):

    print("Read saved values.....")

    with open(SIMULATION_TIME_READ_PATH, 'rb') as f:
        simulation_time = pickle.load(f)

    with open(EEF_POSITION_READ_PATH, 'rb') as f:
        eef_position = pickle.load(f)

    with open(JOINTS_ANGLES_READ_PATH, 'rb') as f:
        joint_angles = pickle.load(f)

    # Decompose values
    eef_position_x = eef_position["x"]
    eef_position_y = eef_position["y"]
    eef_position_z = eef_position["z"]

    q1 = joint_angles["q1"]
    q2 = joint_angles["q2"]
    q3 = joint_angles["q3"]

    print("Start plotting data.....")

    # *==== Cartesian Space Plots ====*

    # Create plot for EEF X position
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.plot(simulation_time, eef_position_x)
    ax.set_xlabel('time [sec]')
    ax.set_ylabel('Position [m]')
    ax.set_title("End Effector Position (X-axis)")
    ax.grid(which='minor', color='black', linewidth=0.2)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_ybound(upper=1.0, lower=0.5)
    fig.set_dpi(200)
    plt.show()

    # Create plot for EEF Y position
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.plot(simulation_time, eef_position_y)
    ax.set_xlabel('time [sec]')
    ax.set_ylabel('Position [m]')
    ax.set_title("End Effector Position (Y-axis)")
    ax.grid(which='minor', color='black', linewidth=0.2)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    fig.set_dpi(200)
    plt.show()

    # Create plot for EEF Z position
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.plot(simulation_time, eef_position_z)
    ax.set_xlabel('time [sec]')
    ax.set_ylabel('Position [m]')
    ax.set_title("End Effector Position (Z-axis)")
    ax.grid(which='minor', color='black', linewidth=0.2)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    fig.set_dpi(200)
    plt.show()

    # *==== Joint Space Plots ====*

    # Plot Joint 1 angles (rad)
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.plot(simulation_time, q1)
    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Angle [rad]')
    ax.set_title("Joint 1 Angle")
    ax.grid(which='minor', color='black', linewidth=0.2)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    fig.set_dpi(200)
    plt.show()

    # Plot Joint 2 angles (rad)
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.plot(simulation_time, q2)
    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Angle [rad]')
    ax.set_title("Joint 2 Angle")
    ax.grid(which='minor', color='black', linewidth=0.2)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    fig.set_dpi(200)
    plt.show()

    # Plot Joint 3 angles (rad)
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.plot(simulation_time, q3)
    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Angle [rad]')
    ax.set_title("Joint 3 Angle")
    ax.grid(which='minor', color='black', linewidth=0.2)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    fig.set_dpi(200)
    plt.show()
