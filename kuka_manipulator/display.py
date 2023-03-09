#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import pickle
import sympy
from typing import List

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


def read_jacobian(subs: bool = False) -> sympy.Matrix:
    """
    Read Jacobian matrix in symbolic form

    Parameters
    ----------
    subs : `bool`
        Substitute symbolic values of `l` related parameters 
    """

    with open(JACOBIAN_READ_PATH, 'rb') as f:
        J = pickle.load(f)

    if (subs):
        l_list = list(sympy.symbols("l0:8"))
        for l in range(0, len(l_list)):
            J = J.subs(l_list[l], L[l])
        J = sympy.simplify(J)

    return J


def read_inverse_jacobian(subs: bool = False) -> sympy.Matrix:
    """
    Read inverse Jacobian matrix in symbolic form

    Parameters
    ----------
    subs : `bool`
        Substitute symbolic values of `l` related parameters 
    """
    with open(INVERSE_JACOBIAN_READ_PATH, "rb") as f:
        J_inv = pickle.load(f)

    if (subs):
        l_list = list(sympy.symbols("l0:8"))
        for l in range(0, len(l_list)):
            J_inv = J_inv.subs(l_list[l], L[l])
        J_inv = sympy.simplify(J_inv)

    return J_inv


def read_determinant_jacobian(subs=False) -> sympy.core.mul.Mul:
    """
    Read saved Jacobian determinant in symbolic form.

    Parameters
    ----------
    subs : `bool`
        Substitute symbolic values of `l` related parameters 
    """
    with open(JACOBIAN_DET_READ_PATH, 'rb') as f:
        det_J = pickle.load(f)

    if (subs):
        l_list = list(sympy.symbols("l0:8"))
        for l in range(0, len(l_list)):
            det_J = det_J.subs(l_list[l], L[l])
        det_J = sympy.simplify(det_J)

    return det_J


def read_forward_kinematics(subs=False) -> List[sympy.Matrix]:
    """
    Read forward kinematics equations in symbolic form

    Parameters
    ----------
    subs : `bool`
        Substitute symbolic values of `l` related parameters 
    """

    with open(HOMOGENOUS_TF_READ_PATH, "rb") as f:
        h_tf = pickle.load(f)

    if (subs):
        l_list = list(sympy.symbols("l0:8"))

        for i, matrix in enumerate(h_tf):
            for l in range(0, len(l_list)):
                matrix = matrix.subs(l_list[l], L[l])
            h_tf[i] = sympy.simplify(matrix)

    return h_tf
