#Module containing containing Iterative Soft-Thresholding with Cooling solver for linear L1-regularized problems
from math import isnan
import pySolver


class ISTCsolver(pySolver.Solver):
	"""ISTC solver to solve: linear problem 1/2*| y - Am |_2 + lambda*| m |_1"""
