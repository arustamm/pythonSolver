# Module containing the definition of the operator necessary for the solver class
# It takes vector objects from the pyVector class

from __future__ import division, print_function, absolute_import
import time
from copy import deepcopy
import numpy as np
from pyVector import vector, superVector
import pyproximal as pp
import pyOperator as Op
import pyProblem as Problem
import sep_util

class ProxOperator:

    def __init__(self):
        raise NotImplementedError("This is an abstract class")

    def prox(self, vector: vector, tau):
        raise NotImplementedError("This is an abstract class")

class ProxOperatorExplicit(ProxOperator):
    """ 
        Proximal operator based on the explicit form (pyproximal)
    """

    # Default class methods/functions
    def __init__(self, prox_operator, epsilon=1):
        self.prox_op = prox_operator
        self.epsilon = epsilon

    def prox(self, input: vector, output:vector, tau):
        arr_in = input.getNdArray()
        arr_out = output.getNdArray()
        # epsilon scaling to make it equivalent to regularized problem formulation
        res = self.prox_op.prox(arr_in, tau * self.epsilon * self.epsilon / 2)
        arr_out[:] = res.reshape(arr_in.shape)[:]

class ProxOperatorImplicit(ProxOperator):
    """ 
        Proximal operator based on the implicit form (by solving the actual optimization problem)
        The proximal operator is defined as the solution of the following optimization problem:
            min ||Ax - b||^2 + epsilon * ||x - u||^2, or
            min ||f(x) - b||^2 + epsilon * ||x - u||^2
        where A is a linear, or f(x) is a non-linear operator,
        u is the vector at which the proximal operator is evaluated, 
        b is the data and epsilon is a regularization parameter
    """

    def __init__(self, model, data, op, solver, epsilon, warm=True):
        self.epsilon = epsilon
        if isinstance(op, Op.NonLinearOperator):
            self.problem = Problem.ProblemL2NonLinearReg(model, data, op, self.epsilon, prior_model=None)
        elif isinstance(op, Op.Operator):
            self.problem = Problem.ProblemL2LinearReg(model, data, op, self.epsilon, prior_model=None)
        else:
            raise TypeError("Provided operator should be of Operator class")
        self.solver = solver
        self.warm = warm

    def prox(self, input: vector, output:vector, tau):
        # if we choose not to start with the previous estimate of the model
        if not self.warm:
            self.problem.model.zero()
        # scale epsilon by the current step size tau
        if tau < 0:
            raise RuntimeError("Error in the proximal evaluation: step size is negative!")
        self.problem.epsilon = np.sqrt(1/(tau * self.epsilon))
        # set the prior in ||x - u||^2 regularization term
        self.problem.prior_model = input.clone()
        self.solver.run(self.problem, verbose=True)
        output.copy(self.problem.model)


    