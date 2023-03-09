#!/usr/bin/python
# -*- coding: utf-8 -*-
import sympy
import os
import pickle

from kuka_manipulator.display import read_jacobian
from kuka_manipulator.helper import compute_det_two

# *==== Constants ====*

JACOBIAN_DETERMINANT_SAVE_PATH = os.getcwd(
) + "/kuka_manipulator/cached_matrices/jacobian_det.pickle"

# *==== Methods ====*


def compute_jacobian_determinant() -> sympy.core.mul.Mul:
    """
    Compute determinant of Jabobian matrix

    Returns
    -------
    det_JL : `sympy.core.mul.Mul`
        Jacobian determinant equation in symbolic form
    """

    # Read jacobian saved matrix
    J = read_jacobian(False)
    J_l = J[:3, :3]

    # Calculate 3x3 determinant through analytic way
    J_l1 = J_l[1:3, 1:3]
    det_Jl1 = sympy.simplify(compute_det_two(J_l1))

    J_l2 = sympy.Matrix([[J_l[0, 1], J_l[0, 2]], [J_l[2, 1], J_l[2, 2]]])
    det_Jl2 = sympy.simplify(compute_det_two(J_l2))

    a, b = J_l[0, 0], J_l[1, 0]
    det_Jl = sympy.simplify(a * det_Jl1 - b * det_Jl2)

    return det_Jl


if (__name__ == "__main__"):
    det_Jl = compute_jacobian_determinant()

    # Save inverse Jacobian matrix
    with open(JACOBIAN_DETERMINANT_SAVE_PATH, 'wb') as outf:
        outf.write(pickle.dumps(det_Jl))
