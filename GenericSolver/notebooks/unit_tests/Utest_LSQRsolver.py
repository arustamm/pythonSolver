#!/usr/bin/env python3
import sys
sys.path.insert(0, "../../python")
import pyVector
import pyOperator
from pyLinearSolver import LSQRsolver
from pyProblem import ProblemL2Linear
from pyStopper import BasicStopper
import numpy as np

if __name__ == "__main__":
    # same as scipy (https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html)
    model = pyVector.vectorIC(np.zeros(2, dtype=float))
    data = pyVector.vectorIC(np.array([1., 0.01, -1.], dtype=float))
    A = pyOperator.MatrixOp(np.array([[1., 0.], [1., 1.], [0., 1.]], dtype=float), model, data)
    
    L2Prob = ProblemL2Linear(model, data, A)
    LSQR = LSQRsolver(BasicStopper(niter=1000))
    LSQR.setDefaults()
    
    LSQR.run(L2Prob, verbose=True)
    
    print(L2Prob.model)  # should be near [1,-1]
