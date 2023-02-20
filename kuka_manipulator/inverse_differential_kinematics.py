#!/usr/bin/python
# -*- coding: utf-8 -*-

import pickle
from pprint import pprint
import numpy as np
import sympy
import os


JACOBIAN_READ_PATH = os.getcwd() + "/kuka_manipulator/cached_matrices/jacobian.pickle"
INV_JACOBIAN_SAVE_PATH = os.getcwd(
) + "/kuka_manipulator/cached_matrices/inverse_jacobian.pickle"
JACOBIAN_DETERMINANT_SAVE_PATH = os.getcwd(
) + "/kuka_manipulator/cached_matrices/jacobian_det.pickle"


def compute_inverse_jacobian_matrix(j: sympy.Matrix) -> sympy.Matrix:
    """
    """

    # Compute inverse Jacobian matrix
    j_l = sympy.simplify(j[:3, :3])
    A = sympy.Matrix(3, 3, sympy.symbols('A:3:3'))
    det_j = A.det().subs(zip(list(A), list(j_l)))
    det_j = sympy.simplify(sympy.det(j_l))

    # Compute co-factor matrix
    c_11 = sympy.simplify(j_l[1, 1] * j_l[2, 2] - j_l[1, 2] * j_l[2, 1])
    c_12 = sympy.simplify(j_l[1, 0] * j_l[2, 2] - j_l[2, 0] * j_l[1, 2])
    c_13 = sympy.simplify(j_l[1, 0] * j_l[2, 1] - j_l[2, 0] * j_l[1, 1])
    c_21 = sympy.simplify(j_l[0, 1] * j_l[2, 2] - j_l[0, 2] * j_l[2, 1])
    c_22 = sympy.simplify(j_l[0, 0] * j_l[2, 2] - j_l[2, 0] * j_l[0, 2])
    c_23 = sympy.simplify(j_l[0, 0] * j_l[2, 1] - j_l[2, 0] * j_l[0, 1])
    c_31 = sympy.simplify(j_l[0, 1] * j_l[1, 2] - j_l[1, 1] * j_l[0, 2])
    c_32 = sympy.simplify(j_l[0, 0] * j_l[1, 2] - j_l[1, 0] * j_l[0, 2])
    c_33 = sympy.simplify(j_l[0, 0] * j_l[1, 1] - j_l[1, 0] * j_l[0, 1])
    coeffs = sympy.Matrix(
        [[c_11, c_12, c_13], [c_21, c_22, c_23], [c_31, c_32, c_33]])

    mul = np.array([[1, -1, 1], [-1, 1, -1], [1, -1, 1]])

    adjoint = (mul @ coeffs).transpose()
    j_l_inv = sympy.simplify(adjoint / det_j)
    pprint(j_l_inv)

    return j_l_inv


def read_jacobian() -> sympy.Matrix:
    """
    """
    with open(JACOBIAN_READ_PATH, 'rb') as f:
        J = pickle.load(f)

    return J


if (__name__ == "__main__"):
    J = read_jacobian()

    J_inv = compute_inverse_jacobian_matrix(J)

    # Save inverse Jacobian matrix
    with open(INV_JACOBIAN_SAVE_PATH, 'wb') as outf:
        outf.write(pickle.dumps(J_inv))

    pprint(J_inv)
