import sys
import os
import numpy as np

import GenericSolver.python.pyVector as pyVector
import GenericSolver.python.pyOperator as pyOperator

# First test on scaling a vector
x = pyVector.vectorIC(np.ones((100, 200)))
y = x.clone()
S = pyOperator.scalingOp(x, .5)
S.forward(False, x, y)

# Test add operator
Z = pyOperator.ZeroOp(x, x)
I = pyOperator.IdentityOp(x)
sumOp = I + Z
sumOp.forward(False, x, y)
x.isDifferent(y)

# Test prod operator
I2 = I*2
I2.forward(False, x, y)
z = x.clone()
z * 2
y.isDifferent(z)

prod = I * Z
prod.forward(False, x, y)
z = x.clone()
z.zero()
y.isDifferent(z)

# Test combinations
combo = I + I + Z
combo.forward(False, x, y)

comboS = S * S + Z
comboS.forward(False, x, y)

comboS.adjoint(False, z, x)
comboS.H.forward(False, x, z)

V = pyOperator.Vstack(I, Z)
z = pyVector.superVector(x.clone(), x.clone())
z.zero()
V.forward(False, x, z)
V.H.forward(False, y, z)

H = pyOperator.Hstack(I, Z)
a = z.clone()
a.zero()
b = a.clone()
H.forward(False, a, y)
H.H.forward(False, b, y)  # b should be equal to a

# Test inversion x = A / y
y = pyVector.vectorIC(np.ones((200, 1)))
y * 10
x = pyVector.vectorIC(np.ones((200, 1)))
A = pyOperator.scalingOp(x, 10)

y_hat = y.clone()
y_hat.zero()
A.forward(False, x, y_hat)
y.isDifferent(y_hat)

x_hat = A / y
x.isDifferent(x_hat)