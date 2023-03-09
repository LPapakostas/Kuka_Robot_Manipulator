#!/usr/bin/python
# -*- coding: utf-8 -*-

import pickle
from pprint import pprint
from typing import List
import sympy
import os
from sympy import sin, cos

# *==== Constants ====*

INV_JACOBIAN_SAVE_PATH = os.getcwd(
) + "/kuka_manipulator/cached_matrices/inverse_jacobian.pickle"

# *==== Methods ====*


def compute_inv_jacobian(q_list: List[sympy.Symbol], l_list: List[sympy.Symbol]) -> sympy.Matrix:
    """
    Compute Inverse Jacobian of Kuka Maninpulator

    Parameters
    ----------
    q_list : `List`
        List of joint angles in symbolic form
    l_list : `List`
        List of link lengths in symbolic form

    Returns
    -------
    J_inv : `sympy.Matrix`
        Inverse Jacobian matrix in symbolic form
    """
    assert(len(q_list) == 3)
    assert(len(l_list) == 8)

    q1, q2, q3 = q_list[0], q_list[1], q_list[2]

    l0, l1, l2 = l_list[0], l_list[1], l_list[2]
    l3, l4, l5 = l_list[3], l_list[4], l_list[5]
    l6, l7 = l_list[6], l_list[7]

    # Define constants
    L = l3 + l6
    p1 = l1 + l2 * sin(q2) + l4 * sin(q2 + q3) + l5 * \
        cos(q2 + q3) + l7 * cos(q2 + q3)
    p2 = l4 * sin(q2 + q3) + l5 * cos(q2 + q3) + l7 * cos(q2 + q3)
    p3 = l4 * sin(q3) + l5 * cos(q3) + l7 * cos(q3)
    p4 = l5 * sin(q2 + q3) + l7 * sin(q2 + q3) - l4 * cos(q2 + q3)
    p5 = l2 * cos(q2) + l4 * cos(q2 + q3) - l5 * \
        sin(q2 + q3) - l7 * sin(q2 + q3)

    # Define inverse jacobian matrix elements
    d11 = -sin(q1) / p1
    d12 = cos(q1) / p1
    d13 = 0
    d21 = (p2 * (p1 * cos(q1) - L * sin(q1))) / (l2 * p1 * p3)
    d22 = (p2 * (p1 * sin(q1) + L * cos(q1))) / (l2 * p1 * p3)
    d23 = -p4 / (l2 * p3)
    d31 = ((l1 - p1) * (p1 * cos(q1) - L * sin(q1))) / (l2 * p1 * p3)
    d32 = ((l1 - p1) * (p1 * sin(q1) + L * cos(q1))) / (l2 * p1 * p3)
    d33 = -p5 / (l2 * p3)

    # Create inverse jacobian matrix
    J_inv = sympy.Matrix([[d11, d12, d13], [d21, d22, d23], [d31, d32, d33]])
    return J_inv


if (__name__ == "__main__"):

    q_list = list(sympy.symbols("q1:4"))
    l_list = list(sympy.symbols("l0:8"))
    J_inv = compute_inv_jacobian(q_list, l_list)

    # Save inverse Jacobian matrix
    with open(INV_JACOBIAN_SAVE_PATH, 'wb') as outf:
        outf.write(pickle.dumps(J_inv))

    pprint(J_inv)
