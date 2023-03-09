#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# *==== CONSTANTS ====*
SIMULATION_TIME_READ_PATH = (
    os.getcwd() + "/kuka_manipulator/simulation/simulation_time.pickle"
)

REFERENCE_POSITION_TRAJECTORY_READ_PATH = (
    os.getcwd() + "/kuka_manipulator/simulation/reference_position_trajectory.pickle"
)

LINK1_POSITION_READ_PATH = (
    os.getcwd() + "/kuka_manipulator/simulation/link1_position.pickle"
)

LINK2_POSITION_READ_PATH = (
    os.getcwd() + "/kuka_manipulator/simulation/link2_position.pickle"
)

LINK3_POSITION_READ_PATH = (
    os.getcwd() + "/kuka_manipulator/simulation/link3_position.pickle"
)

LINK4_POSITION_READ_PATH = (
    os.getcwd() + "/kuka_manipulator/simulation/link4_position.pickle"
)

LINK5_POSITION_READ_PATH = (
    os.getcwd() + "/kuka_manipulator/simulation/link5_position.pickle"
)

EEF_POSITION_READ_PATH = (
    os.getcwd() + "/kuka_manipulator/simulation/actual_trajectory.pickle"
)


if __name__ == "__main__":
    # Read all saved values
    print("Read saved values")

    with open(SIMULATION_TIME_READ_PATH, "rb") as f:
        simuation_time = pickle.load(f)

    with open(REFERENCE_POSITION_TRAJECTORY_READ_PATH, "rb") as f:
        refence_trajectory = pickle.load(f)

    with open(LINK1_POSITION_READ_PATH, "rb") as f:
        link1_position = pickle.load(f)

    with open(LINK2_POSITION_READ_PATH, "rb") as f:
        link2_position = pickle.load(f)

    with open(LINK3_POSITION_READ_PATH, "rb") as f:
        link3_position = pickle.load(f)

    with open(LINK4_POSITION_READ_PATH, "rb") as f:
        link4_position = pickle.load(f)

    with open(LINK5_POSITION_READ_PATH, "rb") as f:
        link5_position = pickle.load(f)

    with open(EEF_POSITION_READ_PATH, "rb") as f:
        eef_position = pickle.load(f)

    # Decompose values
    refence_trajectory_x = refence_trajectory["x"]
    refence_trajectory_y = refence_trajectory["y"]
    refence_trajectory_z = refence_trajectory["z"]

    link1_position_x = link1_position["x"]
    link1_position_y = link1_position["y"]
    link1_position_z = link1_position["z"]

    link2_position_x = link2_position["x"]
    link2_position_y = link2_position["y"]
    link2_position_z = link2_position["z"]

    link3_position_x = link3_position["x"]
    link3_position_y = link3_position["y"]
    link3_position_z = link3_position["z"]

    link4_position_x = link4_position["x"]
    link4_position_y = link4_position["y"]
    link4_position_z = link4_position["z"]

    link5_position_x = link5_position["x"]
    link5_position_y = link5_position["y"]
    link5_position_z = link5_position["z"]

    eef_position_x = eef_position["x"]
    eef_position_y = eef_position["y"]
    eef_position_z = eef_position["z"]

    dt = simuation_time[1] - simuation_time[0]
    step = int(5 / dt)
    # 0, 5, 10, 15 and 20s
    desired_times = [0, 500, 1000, 1500, -1]

    print("Start trajectory animation (2D).... ")

    for td in desired_times:
        joint_positions_x = [
            0,
            link1_position_x[td],
            link2_position_x[td],
            link3_position_x[td],
            link4_position_x[td],
            link5_position_x[td],
        ]
        joint_positions_y = [
            0,
            link1_position_y[td],
            link2_position_y[td],
            link3_position_y[td],
            link4_position_y[td],
            link5_position_y[td],
        ]

        fig, ax = plt.subplots(figsize=(15, 15))
        ax.plot(
            refence_trajectory_x, refence_trajectory_y, label="Reference Trajectory"
        )
        ax.plot(joint_positions_x, joint_positions_y, "o", ms=5, label="Joints")
        ax.plot(
            eef_position_x[td],
            eef_position_y[td],
            "x",
            ms=10,
            label="End Effector Position",
        )
        ax.plot(
            [0, link1_position_x[td]],
            [0, link1_position_y[td]],
            color="black",
            alpha=0.6,
        )  # 0 --> 1 Link
        ax.plot(
            [link1_position_x[td], link2_position_x[td]],
            [link1_position_y[td], link2_position_y[td]],
            color="black",
            alpha=0.6,
        )  # 1 --> 2 Link
        ax.plot(
            [link2_position_x[td], link3_position_x[td]],
            [link2_position_y[td], link3_position_y[td]],
            color="black",
            alpha=0.6,
        )  # 2 --> 3 Link
        ax.plot(
            [link3_position_x[td], link4_position_x[td]],
            [link3_position_y[td], link4_position_y[td]],
            color="black",
            alpha=0.6,
        )  # 3 --> 4 Link
        ax.plot(
            [link4_position_x[td], link5_position_x[td]],
            [link4_position_y[td], link5_position_y[td]],
            color="black",
            alpha=0.6,
        )  # 4 --> 5 Link
        ax.plot(
            [link5_position_x[td], eef_position_x[td]],
            [link5_position_y[td], eef_position_y[td]],
            color="black",
            alpha=0.6,
        )  # 5 --> EEF Link
        ax.set_xlabel("Position (X) [m]")
        ax.set_ylabel("Position (Y) [m]")
        ax.set_title("End Effector Trajectory Animation")
        ax.legend(loc="upper left")
        fig.set_dpi(200)
        plt.show()

    print("Start trajectory animation (3D)....")

    for td in desired_times:
        joint_positions_x = [
            0,
            link1_position_x[td],
            link2_position_x[td],
            link3_position_x[td],
            link4_position_x[td],
            link5_position_x[td],
        ]
        joint_positions_y = [
            0,
            link1_position_y[td],
            link2_position_y[td],
            link3_position_y[td],
            link4_position_y[td],
            link5_position_y[td],
        ]
        joint_positions_z = [
            0,
            link1_position_z[td],
            link2_position_z[td],
            link3_position_z[td],
            link4_position_z[td],
            link5_position_z[td],
        ]

        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.plot(
            refence_trajectory_x,
            refence_trajectory_y,
            refence_trajectory_z,
            label="Reference Trajectory",
        )
        ax.plot(
            joint_positions_x,
            joint_positions_y,
            joint_positions_z,
            "o",
            ms=5,
            label="Joints",
        )
        ax.plot(
            eef_position_x[td],
            eef_position_y[td],
            eef_position_z[td],
            "x",
            ms=10,
            label="End Effector Position",
        )
        ax.plot(
            [0, link1_position_x[td]],
            [0, link1_position_y[td]],
            [0, link1_position_z[td]],
            color="black",
            alpha=0.6,
        )  # 0 --> 1 Link

        ax.plot(
            [link1_position_x[td], link2_position_x[td]],
            [link1_position_y[td], link2_position_y[td]],
            [link1_position_z[td], link2_position_z[td]],
            color="black",
            alpha=0.6,
        )  # 1 --> 2 Link

        ax.plot(
            [link2_position_x[td], link3_position_x[td]],
            [link2_position_y[td], link3_position_y[td]],
            [link2_position_z[td], link3_position_z[td]],
            color="black",
            alpha=0.6,
        )  # 2 --> 3 Link

        ax.plot(
            [link3_position_x[td], link4_position_x[td]],
            [link3_position_y[td], link4_position_y[td]],
            [link3_position_z[td], link4_position_z[td]],
            color="black",
            alpha=0.6,
        )  # 3 --> 4 Link

        ax.plot(
            [link4_position_x[td], link5_position_x[td]],
            [link4_position_y[td], link5_position_y[td]],
            [link4_position_z[td], link5_position_z[td]],
            color="black",
            alpha=0.6,
        )  # 4 --> 5 Link

        ax.plot(
            [link5_position_x[td], eef_position_x[td]],
            [link5_position_y[td], eef_position_y[td]],
            [link5_position_z[td], eef_position_z[td]],
            color="black",
            alpha=0.6,
        )  # 5 --> EEF Link
        ax.set_xlabel("Position (X) [m]")
        ax.set_ylabel("Position (Y) [m]")
        ax.set_zlabel("Position (Z) [m]")
        ax.set_title("End Effector Trajectory Animation")
        ax.legend()
        fig.set_dpi(200)
        plt.show()
