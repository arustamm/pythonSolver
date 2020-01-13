#!/usr/bin/env python3
import sys, os

sys.path.insert(0, "../../python")
import pyVector as Vec
import pyOperator as Op
from pyLinearSolver import LSQRsolver
from pyProblem import ProblemL2Linear
from pyStopper import BasicStopper
import numpy as np


class MatMult_incore(Op.Operator):
    """Operator class to perform matrix-vector multiplication"""

    def __init__(self, A, domain, range):
        """Constructor for the class: A = matrix to use; domain = domain vector; range = range vector"""
        if not isinstance(domain, Vec.vector): raise TypeError("ERROR! Domain vector not a vector object")
        if (not isinstance(range, Vec.vector)): raise TypeError("ERROR! Range vector not a vector object")
        # Setting domain and range of operator and matrix to use during application of the operator
        self.setDomainRange(domain, range)
        self.A = np.matrix(A)
        return

    def forward(self, add, model, data):
        """Method to compute d = A m"""
        self.checkDomainRange(model, data)
        if (not isinstance(model, Vec.vector)): raise TypeError("ERROR! Model vector not a vector object")
        if (not isinstance(data, Vec.vector)): raise TypeError("ERROR! Data vector not a vector object")
        if (not add): data.zero()
        model_arr = model.getNdArray()
        data_arr = data.getNdArray()
        data_arr += np.matmul(self.A, model_arr)
        return

    def adjoint(self, add, model, data):
        """Method to compute m = A d"""
        self.checkDomainRange(model, data)
        if (not isinstance(model, Vec.vector)): raise TypeError("ERROR! Model vector not a vector object")
        if (not isinstance(data, Vec.vector)): raise TypeError("ERROR! Data vector not a vector object")
        if (not add): model.zero()
        model_arr = model.getNdArray()
        data_arr = data.getNdArray()
        model_arr += np.matmul(self.A.H, data_arr)
        return
    

if __name__ == "__main__":
    
    # same as scipy (https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html)
    model_vec = Vec.vectorIC(np.zeros(2, dtype=float))
    data_vec = Vec.vectorIC(np.array([1., 0.01, -1.], dtype=float))
    A = np.array([[1., 0.], [1., 1.], [0., 1.]], dtype=float)
    MatMult = MatMult_incore(A, model_vec, data_vec)

    L2Prob = ProblemL2Linear(model_vec, data_vec, MatMult)
    LSQR = LSQRsolver(BasicStopper(niter=1000))
    LSQR.setDefaults()
    
    LSQR.run(L2Prob, verbose=True)
    
    print(L2Prob.model)  # should be near [1,-1]
