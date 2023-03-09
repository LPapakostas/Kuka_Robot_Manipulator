#!/usr/bin/python
# -*- coding: utf-8 -*-
import sympy


def rot_x(theta: sympy.Symbol) -> sympy.Matrix:
    """
    Compute Rotation matrix along x-axis
    """
    si = sympy.sin(theta)
    ci = sympy.cos(theta)

    rx = sympy.Matrix([[1, 0, 0, 0], [0, ci, -si, 0], [0, si, ci, 0], [0, 0, 0, 1]])

    return rx


def rot_y(theta: sympy.Symbol) -> sympy.Matrix:
    """
    Compute Rotation matrix along y-axis
    """
    si = sympy.sin(theta)
    ci = sympy.cos(theta)

    ry = sympy.Matrix([[ci, 0, si, 0], [0, 1, 0, 0], [-si, 0, ci, 0], [0, 0, 0, 1]])
    return ry


def rot_z(theta: sympy.Symbol) -> sympy.Matrix:
    """
    Compute Rotation matrix along z-axis
    """
    si = sympy.sin(theta)
    ci = sympy.cos(theta)

    rz = sympy.Matrix([[ci, -si, 0, 0], [si, ci, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    return rz


def tra_x(d: sympy.Symbol) -> sympy.Matrix:
    """
    Compute Translation matrix along x-axis
    """
    trax = sympy.Matrix([[1, 0, 0, d], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    return trax


def tra_y(d: sympy.Symbol) -> sympy.Matrix:
    """
    Compute Translation matrix along y-axis
    """
    tray = sympy.Matrix([[1, 0, 0, 0], [0, 1, 0, d], [0, 0, 1, 0], [0, 0, 0, 1]])
    return tray


def tra_z(d: sympy.Symbol) -> sympy.Matrix:
    """
    Compute Translation matrix along z-axis
    """
    traz = sympy.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, d], [0, 0, 0, 1]])
    return traz


def dh_homogenous(
    theta: sympy.Symbol, d: sympy.Symbol, alpha: sympy.Symbol, a: sympy.Symbol
) -> sympy.Matrix:
    """
    Compute Homogenous Transformation matrix from Denavit-Hartenberg parameters
    """

    c_th = sympy.cos(theta)
    s_th = sympy.sin(theta)

    c_a = sympy.cos(alpha)
    s_a = sympy.sin(alpha)

    dh_homogenous_matrix = sympy.Matrix(
        [
            [c_th, -s_th * c_a, s_th * s_a, a * c_th],
            [s_th, c_th * c_a, -c_th * s_a, a * s_th],
            [0, s_a, c_a, d],
            [0, 0, 0, 1],
        ]
    )

    return dh_homogenous_matrix


def homogenous_inverse(T: sympy.Matrix) -> sympy.Matrix:
    """
    Compute inverse homogenous matrix
    """

    R = T[0:3, 0:3]
    P = T[0:3, 3]

    T_inv = sympy.Matrix([[R.T, -R.T @ P], [0, 0, 0, 1]])
    return T_inv


def skew_symmetric(vector: sympy.Matrix) -> sympy.Matrix:
    """
    Compute skew symmetric matrix for a given vector
    """

    # Check if vector is properly declared
    dim_a, dim_b = vector.rows, vector.cols
    assert dim_a + dim_b == 4

    a1, a2, a3 = vector[0], vector[1], vector[2]

    skew_matrix = sympy.Matrix([[0, -a3, a2], [a3, 0, -a1], [-a2, a1, 0]])

    return skew_matrix


def compute_det_two(m: sympy.Matrix) -> sympy.core.mul.Mul:
    """
    Create equation for 2x2 matrix determinant
    """

    A = sympy.Matrix(2, 2, sympy.symbols("A:2:2"))
    det = A.det().subs(zip(list(A), list(m)))
    det = sympy.simplify(det)

    return det
