#!/usr/bin/python
# -*- coding: utf-8 -*-
from math import sqrt, pow, pi
from math import sin, cos, atan2
from typing import List

# *==== CONSTANTS ====*

L0 = 0.810  # in [m]
L1 = 0.200  # in [m]
L2 = 0.600  # in [m]
L3 = 0.030  # in [m]
L4 = 0.140  # in [m]
L5 = 0.550  # in [m]
L6 = 0.100  # in [m]
L7 = 0.100  # in [m]


def compute_inverse_kinematics(x: float, y: float, z: float) -> List[float]:
    """
    Calculate inverse kinematics for specific arm manipulator

    Parameters
    ----------
    x : `double`
        End Effector x-axis coordinate
    y : `double`
        End Effector y-axis coordinate
    z : `double`
        End Effector y-axis coordinate

    Returns
    -------
    [q1, q2, q3] : `List`
        Computed angles for joints 1,2,3
    """

    # Define constants for `q1` joint angle
    L = L3 + L6
    K = sqrt(pow(x, 2) + pow(y, 2) - pow(L, 2))
    # Compute q1
    q1 = atan2((K * y - L * x), K * x + L * y)

    # Define constants for `q2` joint angle
    R = (
        pow((K - L1), 2) + pow((z - L0), 2) - pow(L2, 2) - pow(L4, 2) - pow(L5 + L7, 2)
    ) / (2 * L2)
    Q = pow(L5 + L7, 2) + pow(L4, 2) - pow(R, 2)
    # Compute q3
    q3 = 2 * atan2(-(L5 + L7) + sqrt(Q), R + L4) + 2 * pi

    # Define constants for `q3` joint angle
    A = L2 + L4 * cos(q3) - (L5 + L7) * sin(q3)
    B = L4 * sin(q3) + (L5 + L7) * cos(q3)
    # Compute q2
    q2 = atan2(A * (K - L1) - B * (z - L0), A * (z - L0) + B * (K - L1))

    q_lst = [q1, q2, q3]
    return q_lst


if __name__ == "__main__":
    P_A = [0.7, -0.4, 0.95]
    q_lst = compute_inverse_kinematics(P_A[0], P_A[1], P_A[2])

    print(q_lst)
