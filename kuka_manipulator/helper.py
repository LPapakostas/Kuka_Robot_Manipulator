#!/usr/bin/python
# -*- coding: utf-8 -*-
import sympy


def rot_x(theta: sympy.Symbol) -> sympy.Matrix:
    """
    Compute Rotation matrix along x-axis
    """
    si = sympy.sin(theta)
    ci = sympy.cos(theta)

    rx = sympy.Matrix([[1, 0, 0, 0],
                       [0, ci, -si, 0],
                       [0, si, ci, 0],
                       [0, 0, 0, 1]])

    return rx


def rot_y(theta: sympy.Symbol) -> sympy.Matrix:
    """
    Compute Rotation matrix along y-axis
    """
    si = sympy.sin(theta)
    ci = sympy.cos(theta)

    ry = sympy.Matrix([[ci, 0, si, 0],
                       [0, 1, 0, 0],
                       [-si, 0, ci, 0],
                       [0, 0, 0, 1]])
    return ry


def rot_z(theta: sympy.Symbol) -> sympy.Matrix:
    """
    Compute Rotation matrix along z-axis
    """
    si = sympy.sin(theta)
    ci = sympy.cos(theta)

    rz = sympy.Matrix([[ci, -si, 0, 0],
                       [si, ci, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])

    return rz


def tra_x(d: sympy.Symbol) -> sympy.Matrix:
    """
    Compute Translation matrix along x-axis
    """
    trax = sympy.Matrix([[1, 0, 0, d],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
    return trax


def tra_y(d: sympy.Symbol) -> sympy.Matrix:
    """
    Compute Translation matrix along y-axis
    """
    tray = sympy.Matrix([[1, 0, 0, 0],
                         [0, 1, 0, d],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
    return tray


def tra_z(d: sympy.Symbol) -> sympy.Matrix:
    """
    Compute Translation matrix along z-axis
    """
    traz = sympy.Matrix([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, d],
                         [0, 0, 0, 1]])
    return traz


def dh_homogenous(theta: sympy.Symbol, d: sympy.Symbol, alpha: sympy.Symbol, a: sympy.Symbol) -> sympy.Matrix:
    """
    Compute Homogenous Transformation matrix from Denavit-Hartenberg parameters 
    """

    c_th = sympy.cos(theta)
    s_th = sympy.sin(theta)

    c_a = sympy.cos(alpha)
    s_a = sympy.sin(alpha)

    dh_homogenous_matrix = sympy.Matrix([
        [c_th, -s_th * c_a, s_th * s_a, a * c_th],
        [s_th, c_th * c_a, -c_th * s_a, a * s_th],
        [0, s_a, c_a, d],
        [0, 0, 0, 1]])

    return dh_homogenous_matrix
