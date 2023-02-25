#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import pickle
import sympy
from pprint import pprint

from kuka_manipulator.helper import compute_det_two, compute_det_three

# *==== Static Parameter Definition ====*
L0 = 0.810  # in [m]
L1 = 0.200  # in [m]
L2 = 0.600  # in [m]
L3 = 0.030  # in [m]
L4 = 0.140  # in [m]
L5 = 0.550  # in [m]
L6 = 0.100  # in [m]
L7 = 0.100  # in [m]
L = [L0, L1, L2, L3, L4, L5, L6, L7]

# *==== Variables ====*
JACOBIAN_READ_PATH = os.getcwd() + "/kuka_manipulator/cached_matrices/jacobian.pickle"
JACOBIAN_DET_READ_PATH = os.getcwd(
) + "/kuka_manipulator/cached_matrices/jacobian_det.pickle"
HOMOGENOUS_TF_READ_PATH = os.getcwd(
) + "/kuka_manipulator/cached_matrices/homogenous_transformations.pickle"
INVERSE_JACOBIAN_READ_PATH = os.getcwd(
) + "/kuka_manipulator/cached_matrices/inverse_jacobian.pickle"

# *==== Methods ====*


def read_jacobian(subs=False) -> sympy.Matrix:
    """
    """
    with open(JACOBIAN_READ_PATH, 'rb') as f:
        J = pickle.load(f)

    if (subs):
        l_list = list(sympy.symbols("l0:8"))
        for l in range(0, len(l_list)):
            J = J.subs(l_list[l], L[l])
        J = sympy.simplify(J)

    return J


def read_inverse_jacobian(subs=False) -> sympy.Matrix:
    """
    """
    with open(INVERSE_JACOBIAN_READ_PATH, "rb") as f:
        J_inv = pickle.load(f)

    if (subs):
        l_list = list(sympy.symbols("l0:8"))
        for l in range(0, len(l_list)):
            J_inv = J_inv.subs(l_list[l], L[l])
        J_inv = sympy.simplify(J_inv)

    return J_inv


def read_determinant_jacobian() -> sympy.core.mul.Mul:
    """
    """
    with open(JACOBIAN_DET_READ_PATH, 'rb') as f:
        det_J = pickle.load(f)

    return det_J


def read_forward_kinematics(subs=False):

    with open(HOMOGENOUS_TF_READ_PATH, "rb") as f:
        h_tf = pickle.load(f)

    if (subs):
        l_list = list(sympy.symbols("l0:8"))

        for i, matrix in enumerate(h_tf):
            for l in range(0, len(l_list)):
                matrix = matrix.subs(l_list[l], L[l])
            h_tf[i] = sympy.simplify(matrix)

    return h_tf


if (__name__ == "__main__"):
    q = list(sympy.symbols("q1:4"))

    J = read_jacobian()
    det_J = read_determinant_jacobian()
    j_l = sympy.trigsimp(J[:3, :3])

    c_11 = sympy.simplify(j_l[1, 1] * j_l[2, 2] - j_l[1, 2] * j_l[2, 1])
    D_11 = sympy.simplify(c_11 / det_J)

    c_21 = sympy.simplify(j_l[0, 1] * j_l[2, 2] - j_l[0, 2] * j_l[2, 1])
    D_12 = -sympy.simplify(c_21 / det_J)

    c_31 = sympy.simplify(j_l[0, 1] * j_l[1, 2] - j_l[1, 1] * j_l[0, 2])
    D_31 = sympy.simplify(c_31 / det_J)

    c_12 = sympy.simplify(j_l[1, 0] * j_l[2, 2] - j_l[2, 0] * j_l[1, 2])
    D_21 = -sympy.simplify(c_12 / det_J)

    c_22 = sympy.simplify(j_l[0, 0] * j_l[2, 2] - j_l[2, 0] * j_l[0, 2])
    D_22 = sympy.simplify(c_22 / det_J)

    c_32 = sympy.simplify(j_l[0, 0] * j_l[1, 2] - j_l[1, 0] * j_l[0, 2])
    D_23 = -sympy.simplify(c_32 / det_J)

    c_13 = sympy.simplify(j_l[1, 0] * j_l[2, 1] - j_l[2, 0] * j_l[1, 1])
    D_31 = sympy.simplify(c_13 / det_J)

    c_23 = sympy.simplify(j_l[0, 0] * j_l[2, 1] - j_l[2, 0] * j_l[0, 1])
    D_23 = -sympy.simplify(c_23 / det_J)

    c_33 = sympy.simplify(j_l[0, 0] * j_l[1, 1] - j_l[1, 0] * j_l[0, 1])
    D_33 = sympy.simplify(c_33 / det_J)
    pprint(D_33)
