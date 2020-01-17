#!/usr/bin/env python3
import sys
sys.path.insert(0, "../../python")
import pyVector
import pyOperator
from pyLinearSolver import LSQRsolver, LCGsolver
from pyProblem import ProblemL2Linear
from pyStopper import BasicStopper
import numpy as np
try:
    from scipy.sparse.linalg import lsqr
except ImportError:
    import subprocess
    import sys
    subprocess.call([sys.executable, "-m", "pip", "install", "--user", "scipy"])
    from scipy.sparse.linalg import lsqr
    
    
if __name__ == "__main__":
    # same as scipy (https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html)
    model = pyVector.vectorIC(np.zeros(2, dtype=float))
    data = pyVector.vectorIC(np.array([1., 0.01, -1.], dtype=float))
    A = pyOperator.MatrixOp(np.array([[1., 0.], [1., 1.], [0., 1.]], dtype=float), model, data)
    
    # CG for benchmarking
    L2ProbCG = ProblemL2Linear(model, data, A)
    CG = LCGsolver(BasicStopper(niter=1000))
    CG.setDefaults()
    CG.run(L2ProbCG, verbose=False)
    print('CG result: \t\t', L2ProbCG.model.getNdArray())  # should be near [1, -1]
    
    # LSQR from scipy
    print('scipy result: \t',
          lsqr(np.array([[1., 0.], [1., 1.], [0., 1.]], dtype=float), np.array([1., 0.01, -1.], dtype=float))[0])
    
    # LSQR
    L2Prob = ProblemL2Linear(model, data, A)
    LSQR = LSQRsolver(BasicStopper(niter=1000))
    LSQR.setDefaults()
    LSQR.run(L2Prob, verbose=False)
    print('LSQR result: \t', L2Prob.model.getNdArray())  # should be near [1, -1]
