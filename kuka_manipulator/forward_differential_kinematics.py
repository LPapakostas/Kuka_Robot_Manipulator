#!/usr/bin/python
# -*- coding: utf-8 -*-

import sympy
import numpy as np
from kuka_manipulator.forward_kinematics import compute_kuka_forward_kinematics
from kuka_manipulator.helper import skew_symmetric
from typing import List
from pprint import pprint

# *==== Constants ====*

DH_BASE = np.array([[0], [0], [1]])


# *==== Methods ====*


def compute_jacobian_matrix(q_list: List[sympy.Symbol], l_list: List[sympy.Symbol]):

    # Compute and decompose homogenous matrices
    dh_homogenous_matrices = compute_kuka_forward_kinematics(q_list, l_list)
    assert(len(dh_homogenous_matrices) == 6)
    A_0_1, A_0_2, A_0_3 = dh_homogenous_matrices[0], dh_homogenous_matrices[1], dh_homogenous_matrices[2]
    A_0_4, A_0_5, A_0_E = dh_homogenous_matrices[3], dh_homogenous_matrices[4], dh_homogenous_matrices[5]

    # Compute J_L1 and J_A1
    R_0_0 = sympy.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    b_0 = R_0_0 @ DH_BASE
    b_0_skew = skew_symmetric(b_0)
    p_0_E = A_0_E[0:3, 3]

    j_l1 = sympy.simplify(b_0_skew @ p_0_E)
    j_a1 = b_0

    # Compute J_L2 and J_A2
    R_0_1 = A_0_1[0:3, 0:3]
    b_1 = R_0_1 @ DH_BASE
    b_1_skew = skew_symmetric(b_1)
    p_0_1 = A_0_1[0:3, 3]
    p_1_E = p_0_E - p_0_1

    j_l2 = sympy.simplify(b_1_skew @ p_1_E)
    j_a2 = b_1

    # Compute J_L3 and J_A3
    R_0_2 = A_0_2[0:3, 0:3]
    b_2 = R_0_2 @ DH_BASE
    b_2_skew = skew_symmetric(b_2)
    p_0_2 = A_0_2[0:3, 3]
    p_2_E = p_0_E - p_0_2

    j_l3 = sympy.simplify(b_2_skew @ p_2_E)
    j_a3 = b_2

    pass


if (__name__ == "__main__"):
    q_list = list(sympy.symbols("q1:7"))
    l_list = list(sympy.symbols("l0:8"))

    compute_jacobian_matrix(q_list, l_list)
