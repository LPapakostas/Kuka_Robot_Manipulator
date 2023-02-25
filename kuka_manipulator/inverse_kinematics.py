#!/usr/bin/python
# -*- coding: utf-8 -*-
from math import sqrt, pow, atan, acos
from math import sin, cos
from numpy import arctan2

# *==== CONSTANTS ====*

L0 = 0.810  # in [m]
L1 = 0.200  # in [m]
L2 = 0.600  # in [m]
L3 = 0.030  # in [m]
L4 = 0.140  # in [m]
L5 = 0.550  # in [m]
L6 = 0.100  # in [m]
L7 = 0.100  # in [m]


def compute_inverse_kinematics(x, y, z):
    """
    """
    # Define constants for `q1` joint angle
    L = L3 + L6
    K = sqrt(pow(x, 2) + pow(y, 2) - pow(L, 2))
    # Compute q1
    q1 = arctan2((y*K - x*L), (x*K+y*L))

    # Define constants for `q2` joint angle
    Q = z + K - L0 - L1
    R = pow((K - L1), 2) + pow((z-L0), 2) - pow(L4, 2) - pow(L5+L7, 2)
    # Compute q2
    q2 = acos((Q - sqrt(Q-2*R))/(2*L2))

    # Define constants for `q3` joint angle
    A = L0 + L4*sin(q2) + (L5 + L7 + L2)*cos(q2) - z
    B = L4*cos(q2) - (L5 + L7)*sin(q2)
    C = L0 + (L2-L4)*sin(q2) - (L5+L7)*cos(q2) - z
    # Compute q3
    q3 = 2*atan((-B + sqrt(pow(B, 2) - A*C)) / A)

    q_lst = [q1, q2, q3]
    return q_lst


if (__name__ == "__main__"):
    P_A = [0.7, -0.4, 0.95]
    q_lst = compute_inverse_kinematics(P_A[0], P_A[1], P_A[2])

    print(q_lst)
