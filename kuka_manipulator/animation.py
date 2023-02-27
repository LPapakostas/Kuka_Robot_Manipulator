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

REFERENCE_TRAJECTORY_READ_PATH = os.getcwd(
) + "/kuka_manipulator/simulation/reference_trajectory.pickle"

LINK1_POSITION_READ_PATH = os.getcwd(
) + "/kuka_manipulator/simulation/link1_position.pickle"

LINK2_POSITION_READ_PATH = os.getcwd(
) + "/kuka_manipulator/simulation/link2_position.pickle"

LINK3_POSITION_READ_PATH = os.getcwd(
) + "/kuka_manipulator/simulation/link3_position.pickle"

LINK4_POSITION_READ_PATH = os.getcwd(
) + "/kuka_manipulator/simulation/link4_position.pickle"

LINK5_POSITION_READ_PATH = os.getcwd(
) + "/kuka_manipulator/simulation/link5_position.pickle"

EEF_POSITION_READ_PATH = os.getcwd(
) + "/kuka_manipulator/simulation/actual_trajectory.pickle"


if (__name__ == "__main__"):

    # Read all saved values
    print("Read saved values")

    with open(SIMULATION_TIME_READ_PATH, 'rb') as f:
        simuation_time = pickle.load(f)

    with open(REFERENCE_TRAJECTORY_READ_PATH, 'rb') as f:
        refence_trajectory = pickle.load(f)

    with open(LINK1_POSITION_READ_PATH, 'rb') as f:
        link1_position = pickle.load(f)

    with open(LINK2_POSITION_READ_PATH, 'rb') as f:
        link2_position = pickle.load(f)

    with open(LINK3_POSITION_READ_PATH, 'rb') as f:
        link3_position = pickle.load(f)

    with open(LINK4_POSITION_READ_PATH, 'rb') as f:
        link4_position = pickle.load(f)

    with open(LINK5_POSITION_READ_PATH, 'rb') as f:
        link5_position = pickle.load(f)

    with open(EEF_POSITION_READ_PATH, 'rb') as f:
        eef_position = pickle.load(f)

    # Decompose values
    refence_trajectory_x = refence_trajectory["x"]
    refence_trajectory_y = refence_trajectory["y"]

    link1_position_x = link1_position["x"]
    link1_position_y = link1_position["y"]

    link2_position_x = link2_position["x"]
    link2_position_y = link2_position["y"]

    link3_position_x = link3_position["x"]
    link3_position_y = link3_position["y"]

    link4_position_x = link4_position["x"]
    link4_position_y = link4_position["y"]

    link5_position_x = link5_position["x"]
    link5_position_y = link5_position["y"]

    eef_position_x = eef_position["x"]
    eef_position_y = eef_position["y"]

    dt = simuation_time[1] - simuation_time[0]
    step = int(5 / dt)

    print("Start trajectory animation (2D).... ")

    for td in range(0, len(simuation_time), step):

        print(td)

        # Denote joints in XY plane
        o_positions_x = [0, link1_position_x[td], link2_position_x[td],
                         link3_position_x[td], link4_position_x[td], link5_position_x[td]]
        o_positions_y = [0, link1_position_y[td], link2_position_y[td],
                         link3_position_y[td], link4_position_y[td], link5_position_y[td]]

        fig, ax = plt.subplots(figsize=(15, 15))
        ax.plot(refence_trajectory_x, refence_trajectory_y)
        ax.plot(o_positions_x, o_positions_y, 'o')
        ax.plot(eef_position_x[td], eef_position_y[td], 'x')

        ax.plot([0, link1_position_x[td]], [
                0, link1_position_y[td]])  # 0 --> 1 Link

        ax.plot([link1_position_x[td], link2_position_x[td]],
                [link1_position_y[td], link2_position_y[td]])  # 1 --> 2 Link

        ax.plot([link2_position_x[td], link3_position_x[td]],
                [link2_position_y[td], link3_position_y[td]])  # 2 --> 3 Link

        ax.plot([link3_position_x[td], link4_position_x[td]],
                [link3_position_y[td], link4_position_y[td]])  # 3 --> 4 Link

        ax.plot([link4_position_x[td], link5_position_x[td]],
                [link4_position_y[td], link5_position_y[td]])  # 4 --> 5 Link

        ax.plot([link5_position_x[td], eef_position_x[td]],
                [link5_position_y[td], eef_position_y[td]])  # 5 --> EEF Link

        ax.set_xlabel('Position (X) [m]')
        ax.set_ylabel('Position (Y) [m]')
        ax.set_title("End Effector Trajectory Animation")
        ax.grid(which='minor', color='black', linewidth=0.2)
        ax.tick_params(which='minor', bottom=False, left=False)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        fig.set_dpi(200)
        plt.show()

    # Print final position of EEF
    td = -1
    o_positions_x = [0, link1_position_x[td], link2_position_x[td],
                     link3_position_x[td], link4_position_x[td], link5_position_x[td]]
    o_positions_y = [0, link1_position_y[td], link2_position_y[td],
                     link3_position_y[td], link4_position_y[td], link5_position_y[td]]

    fig, ax = plt.subplots(figsize=(15, 15))
    ax.plot(refence_trajectory_x, refence_trajectory_y)
    ax.plot(o_positions_x, o_positions_y, 'o')
    ax.plot(eef_position_x[td], eef_position_y[td], 'x')
    ax.plot([0, link1_position_x[td]], [
            0, link1_position_y[td]])  # 0 --> 1 Link
    ax.plot([link1_position_x[td], link2_position_x[td]],
            [link1_position_y[td], link2_position_y[td]])  # 1 --> 2 Link
    ax.plot([link2_position_x[td], link3_position_x[td]],
            [link2_position_y[td], link3_position_y[td]])  # 2 --> 3 Link
    ax.plot([link3_position_x[td], link4_position_x[td]],
            [link3_position_y[td], link4_position_y[td]])  # 3 --> 4 Link
    ax.plot([link4_position_x[td], link5_position_x[td]],
            [link4_position_y[td], link5_position_y[td]])  # 4 --> 5 Link
    ax.plot([link5_position_x[td], eef_position_x[td]],
            [link5_position_y[td], eef_position_y[td]])  # 5 --> EEF Link
    ax.set_xlabel('Position (X) [m]')
    ax.set_ylabel('Position (Y) [m]')
    ax.set_title("End Effector Trajectory Animation")
    ax.grid(which='minor', color='black', linewidth=0.2)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    fig.set_dpi(200)
    plt.show()
