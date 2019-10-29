import sys
import os
import numpy as np

import GenericSolver.python.pyVector as pyVector
import GenericSolver.python.pyOperator as pyOperator

# First test on scaling a vector
x = pyVector.vectorIC(np.ones((100, 200)))
y = x.clone()

x * y  # x should be 1.
x + y  # x should be 2.
y * 2  # y should be 2.


# superVector from vectors
sv_from_vectors = pyVector.superVector(x.clone(), x.clone())

# superVector from list of vectors
sv_from_list = pyVector.superVector([x.clone(), x.clone()])

# superVector from superVector and vector
sv_from_sv_and_v = pyVector.superVector(sv_from_vectors, x.clone())

# the same operations of add and mul are working also for superVector
sv_from_list * 2  # it should be 4
sv_from_list * sv_from_vectors  # it should be 8

# Test operators on superVectors
a = sv_from_vectors.clone()
b = sv_from_vectors.clone()
S = pyOperator.scalingOp(a, 2)
S.forward(False, a, b)  # b should be 4
