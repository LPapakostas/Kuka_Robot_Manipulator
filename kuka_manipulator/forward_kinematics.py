#!/usr/bin/python
# -*- coding: utf-8 -*-

import sympy
from typing import List, Dict
from pprint import pprint
import os
import pickle

from kuka_manipulator.helper import dh_homogenous

# *==== Constants ====*
HOMOGENOUS_TRANS_SAVE_PATH = (
    os.getcwd() + "/kuka_manipulator/cached_matrices/homogenous_transformations.pickle"
)

# *==== Methods ====*


def generate_DH_table(
    q_list: List[sympy.Symbol], l_list: List[sympy.Symbol]
) -> Dict[str, Dict[str, sympy.Symbol]]:
    """
    Generate table of DH parameters

    Parameters
    ----------
    q_list : `List`
        List of joint angles in symbolic form
    l_list : `List`
        List of link lengths in symbolic form

    Returns
    -------
    dh_parameter_table : `Dict`
        Denavit-Hartenberg parameters table
    """

    assert len(q_list) == 6
    assert len(l_list) == 8

    q1, q2, q3 = q_list[0], q_list[1], q_list[2]
    q4, q5, q6 = q_list[3], q_list[4], q_list[5]

    l0, l1, l2 = l_list[0], l_list[1], l_list[2]
    l3, l4, l5 = l_list[3], l_list[4], l_list[5]
    l6, l7 = l_list[6], l_list[7]

    link_0_params = {"theta": q1, "d": l0, "alpha": -sympy.pi / 2, "a": l1}
    link_1_params = {"theta": q2 - sympy.pi / 2, "d": l3, "alpha": 0, "a": l2}
    link_2_params = {"theta": q3, "d": 0, "alpha": -sympy.pi / 2, "a": l4}
    link_3_params = {"theta": q4, "d": l5, "alpha": sympy.pi / 2, "a": 0}
    link_4_params = {"theta": q5, "d": l6, "alpha": -sympy.pi / 2, "a": 0}
    link_5_params = {"theta": q6, "d": l7, "alpha": 0, "a": 0}

    dh_parameter_table = {
        "0": link_0_params,
        "1": link_1_params,
        "2": link_2_params,
        "3": link_3_params,
        "4": link_4_params,
        "5": link_5_params,
    }

    return dh_parameter_table


def compute_DH_transformation_matrices(
    q_list: List[sympy.Symbol], l_list: List[sympy.Symbol]
) -> List[sympy.Matrix]:
    """
    Compute Homogenous Tranformation matrix, using DH parameters

    Parameters
    ----------
    q_list : `List`
        List of joint angles in symbolic form
    l_list : `List`
        List of link lengths in symbolic form

    Returns
    -------
    homogenous_tranformations : List
        Homogenous transformation of each link in symbolic form
    """

    dh_parameter_table = generate_DH_table(q_list, l_list)
    dh_homogenous_matrices = []

    for i, (_, link_parameters) in enumerate(dh_parameter_table.items()):
        current_theta = link_parameters["theta"]
        current_d = link_parameters["d"]
        current_alpha = link_parameters["alpha"]
        current_a = link_parameters["a"]
        current_homogenous_matrix = dh_homogenous(
            current_theta, current_d, current_alpha, current_a
        )
        dh_homogenous_matrices.append(current_homogenous_matrix)

    return dh_homogenous_matrices


def compute_kuka_forward_kinematics(
    q_list: List[sympy.Symbol], l_list: List[sympy.Symbol]
) -> List[sympy.Matrix]:
    """
    Compute forward kinematics matrix for Kuka Manipulator

    Parameters
    ----------
    q_list : `List`
        List of joint angles in symbolic form
    l_list : `List`
        List of link lengths in symbolic form

    Returns
    -------
    homogenous_tranformations : List
        Homogenous transformation of sequential links in symbolic form
    """

    assert len(q_list) == 6
    assert len(l_list) == 8

    DH_matrices = compute_DH_transformation_matrices(q_list, l_list)
    A_0_E = sympy.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    homogenous_tranformations = []

    for i, matrix in enumerate(DH_matrices):
        # Substitute q4, q5, q4 with 0
        if i > 2:
            matrix = matrix.subs(q_list[i], 0)
        A_0_E = sympy.simplify(A_0_E @ matrix)
        homogenous_tranformations.append(A_0_E)

    return homogenous_tranformations


if __name__ == "__main__":
    q_list = list(sympy.symbols("q1:7"))
    l_list = list(sympy.symbols("l0:8"))
    homogenous_tranformations = compute_kuka_forward_kinematics(q_list, l_list)

    # Save Homogenous Transformations
    with open(HOMOGENOUS_TRANS_SAVE_PATH, "wb") as outf:
        outf.write(pickle.dumps(homogenous_tranformations))

    A_0_E = homogenous_tranformations[-1]
    pprint(A_0_E)
