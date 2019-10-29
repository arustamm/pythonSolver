# Module containing the definition of the operator necessary for the solver class
# It takes vector objects from the pyVector class

from __future__ import division, print_function, absolute_import
import warnings

try:
    from pyVector import vector, superVector
except ImportError:
    from GenericSolver.python.pyVector import vector, superVector
import time
from copy import deepcopy
import numpy as np


# TODO rename model and data to x and y respectively
class Operator:
    """Abstract python operator class"""

    # Default class methods/functions
    def __init__(self, domain, range):
        """Generic class for operator"""
        self.domain = domain.cloneSpace()
        self.range = range.cloneSpace()

    def __del__(self):
        """Default destructor"""
        return

    # unary operators
    def __add__(self, other):  # self + other
        if isinstance(other, Operator):
            return _sumOperator(self, other)
        else:
            raise TypeError('Argument must be an Operator')

    def __sub__(self, other):  # self - other
        self.__add__(-other)

    def __neg__(self):  # -self
        return _scaledOperator(self, -1)

    def __mul__(self, other):  # self * other
        return self.dot(other)

    def __matmul__(self, other):  # self @ other
        if np.isscalar(other):
            raise ValueError("Scalar operands are not allowed, use '*' instead")
        return self.__mul__(other)

    # # TODO implement
    def __invert__(self, other):
        """x = A / y through CG"""
        raise NotImplementedError

    # main function for all kinds of multiplication
    def dot(self, other):
        """Matrix-matrix or matrix-vector or matrix-scalar multiplication."""
        if isinstance(other, Operator):  # A * B
            return _prodOperator(self, other)
        elif np.isscalar(other):  # A * c | # TODO I want to call also c * A (but in the latter __mul__ is of scalar)
            return _scaledOperator(self, other)
        elif isinstance(other, vector) or isinstance(other, superVector):  # A * x
            temp = self.range.clone().zero()
            self.forward(False, other, temp)
            return temp
        else:
            raise TypeError('Expected Operator, (super)Vector or scalar, got %r' % other)

    # TODO not useful
    def setDomainRange(self, domain, range):
        """Function to set (cloning space) domain and range of the operator"""
        self.domain = domain.cloneSpace()
        self.range = range.cloneSpace()

    # TODO not useful
    def getDomain(self):
        """Function to return operator domain"""
        return self.domain

    # TODO not useful
    def getRange(self):
        """Function to return operator range"""
        return self.range

    def checkDomainRange(self, x, y):
        """Function to check model and data vector sizes"""
        if not self.domain.checkSame(x):
            raise ValueError("Provided x vector does not match operator domain")
        if not self.range.checkSame(y):
            raise ValueError("Provided y vector does not match operator range")

    def powerMethod(self, verbose=False, tol=1e-8, niter=None, eval_min=False, return_vec=False):
        """
        Function to estimate maximum eigenvalue of the operator:

        :param return_vec: boolean - Return the estimated eigenvectors [False]
        :param niter: int - Maximum number of operator applications [None]
            if not provided, the function will continue until the tolerance is reached)
        :param eval_min: boolean - Compute the minimum eigenvalue [False]
        :param verbose: boolean - Print information to screen as the method is being run [False]
        :param tol: float - Tolerance on the change of the estimated eigenvalues [1e-6]
        """
        # Cloning input and output vectors
        if verbose:
            print('Running power method to estimate maximum eigenvalue (operator L2 norm)')
        x = self.domain.clone()
        # Checking if matrix is square
        square = False
        try:
            if self.domain.checkSame(self.range):
                square = True
        except RuntimeError:
            pass
        if not square:
            if verbose:
                print("Note: operator is not square, the eigenvalue is associated to A'A not A!")
            d_temp = self.range.clone()
        y = self.domain.clone()
        # randomize the input vector
        x.rand()
        x.scale(1.0 / x.norm())  # Normalizing the initial vector
        y.zero()
        iiter = 0
        eigen_old = 0.0  # Previous estimated eigenvalue
        # Estimating maximum eigenvalue
        if verbose:
            print("Starting iterative process for maximum eigenvalue")
        # Starting the power iteration loop
        while True:
            # Applying adjoint if forward not square
            if square:
                self.forward(False, x, y)  # y = A x
            else:
                self.forward(False, x, d_temp)  # d = A x
                self.adjoint(False, y, d_temp)  # y = A' d = A' A x

            # Estimating eigenvalue (Rayleigh quotient)
            eigen = x.dot(y)  # eigen_i = x' A x / (x'x = 1.0)
            # x = y
            x.copy(y)
            # Normalization of the operator
            x.scale(1.0 / x.norm())
            # Stopping criteria (first number of iterations and then tolerance)
            iiter += 1
            if verbose:
                print("Estimated maximum eigenvalue at iter %s: %s" % (iiter, eigen))
            if niter is not None:
                if iiter >= niter:
                    if verbose:
                        print("Maximum number of iteration reached! Stopping iterative process!")
                    break
            # Checking change on the eigenvalue estimated value
            if abs(eigen - eigen_old) < abs(tol * eigen_old):
                if verbose:
                    print("Tolerance value reached! Stopping iterative process!")
                break
            # eigen_(i-1) = eigen_i
            eigen_old = eigen
        if eval_min:
            x_max = x.clone()  # Cloning "maximum" eigenvector
            eigen_max = deepcopy(eigen)
            # Re-initialize variables
            x.rand()
            x.scale(1.0 / x.norm())  # Normalizing the initial vector
            y.zero()
            iiter = 0
            eigen = 0.0  # Current estimated eigenvalue
            eigen_old = 0.0  # Previous estimated eigenvalue
            # Estimating the minimum eigenvalue
            # Shifting all eigenvalues by maximum one (i.e., A_min = A-muI)
            if verbose:
                print("Starting iterative process for minimum eigenvalue")
            while True:
                # Applying adjoint if forward not square
                if not square:
                    self.forward(False, x, d_temp)  # d = A x
                    self.adjoint(False, y, d_temp)  # y = A' d = A' A x
                else:
                    self.forward(False, x, y)  # y = A x
                # y = Ax - mu*Ix
                y.scaleAdd(x, 1.0, -eigen_max)
                # Estimating eigenvalue (Rayleigh quotient)
                eigen = x.dot(y)  # eigen_i = x' A_min x / (x'x = 1.0)
                # x = y
                x.copy(y)
                # Normalization of the operator
                x.scale(1.0 / x.norm())
                # Stopping criteria (first number of iterations and then tolerance)
                iiter += 1
                if verbose:
                    print("Estimated minimum eigenvalue at iter %s: %s"
                          % (iiter, eigen + eigen_max))
                if niter is not None:
                    if iiter >= niter:
                        if verbose:
                            print("Maximum number of iteration reached! Stopping iterative process!")
                        break
                # Checking change on the eigenvalue estimated value
                if abs(eigen - eigen_old) < abs(tol * eigen_old):
                    if verbose:
                        print("Tolerance value reached! Stopping iterative process!")
                    break
                # eigen_(i-1) = eigen_i
                eigen_old = eigen
            x_min = x.clone()  # Cloning "minimum" eigenvector
            eigen_min = deepcopy(eigen + eigen_max)
            eigen = [eigen_max, eigen_min]
            x = [x_max, x_min]
        return eigen, x if return_vec else eigen

    def dotTest(self, verbose=False, tol=1e-4):
        """
           Function to perform dot-product test:
           :param verbose: boolean; Flag to print information to screen as the method is being run [False]
           :param tol: float; The function throws a Warning if the relative error is greater than maxError [1e-4]
        """
        if verbose:
            print("Dot-product test of forward and adjoint operators")
            print('-' * 49)
        # Allocating temporary vectors for dot-product test
        d1 = self.domain.clone()
        d2 = self.domain.clone()
        r1 = self.range.clone()
        r2 = self.range.clone()

        # Randomize the input vectors
        d1.rand()
        r1.rand()

        # Applying forward and adjoint operators with add=False
        if verbose:
            print("Applying forward operator add=False")
        start = time.time()
        self.forward(False, d1, r2)
        end = time.time()
        if verbose:
            print("	Runs in: %s seconds" % (end - start))
            print("Applying adjoint operator add=False")
        start = time.time()
        self.adjoint(False, d2, r1)
        end = time.time()
        if verbose:
            print("	Runs in: %s seconds" % (end - start))

        # Computing dot products
        dt1 = d1.dot(d2)
        dt2 = r1.dot(r2)

        # Dot-product testing
        if verbose:
            print("Dot products add=False: domain=%s range=%s " % (dt1, dt2))
            print("Absolute error: %s" % (abs(dt1 - dt2)))
            print("Relative error: %s \n" % (abs((dt1 - dt2) / dt2)))
        if abs((dt1 - dt2) / dt1) > tol:
            # Deleting temporary vectors
            del d1, d2, r1, r2
            raise Warning("Dot products failure add=False; relative error greater than tolerance of %s" % tol)

        # Applying forward and adjoint operators with add=True
        if verbose:
            print("\nApplying forward operator add=True")
        start = time.time()
        self.forward(True, d1, r2)
        end = time.time()
        if verbose:
            print("	Runs in: %s seconds" % (end - start))
            print("Applying adjoint operator add=True")
        start = time.time()
        self.adjoint(True, d2, r1)
        end = time.time()
        if verbose:
            print("	Runs in: %s seconds" % (end - start))

        # Computing dot products
        dt1 = d1.dot(d2)
        dt2 = r1.dot(r2)

        if verbose:
            print("Dot products add=True: domain=%s range=%s " % (dt1, dt2))
            print("Absolute error: %s" % (abs(dt1 - dt2)))
            print("Relative error: %s \n" % (abs((dt1 - dt2) / dt2)))
        if abs((dt1 - dt2) / dt1) > tol:
            # Deleting temporary vectors
            del d1, d2, r1, r2
            raise Warning("Dot products failure add=True; relative error greater than tolerance of %s" % tol)

        if verbose:
            print("-" * 49)

        # Deleting temporary vectors
        del d1, d2, r1, r2
        return

    def forward(self, add, model, data):
        """Forward operator"""
        raise NotImplementedError("Forward must be defined")

    def adjoint(self, add, model, data):
        """Adjoint operator"""
        raise NotImplementedError("Adjoint must be defined")

    def hermitian(self):
        """Instantiate the Hermitian operator"""
        return _CustomOperator(domain=self.range, range=self.domain,
                               forward_function=self.adjoint, adjoint_function=self.forward)

    H = property(hermitian)
    T = H  # TODO delete property

    def eigs(self, neigs=None, niter=10):
        """
        Most significant eigenvalues of linear Operator A.
        If A is rectangular, return the square root of eigs of A.H*A
        :param neigs: int, number of eigenvalues to compute
        :param niter: number of iterations for eigenvalue estimation
        :return: list of Operator eigenvalues
        """

        import scipy.sparse.linalg as ssl
        from functools import partial
        Op = ssl.LinearOperator(np.float, (self.range.getNdarray().size, self.domain.getNdarray().size))

        # TODO how to transform the (model, data) paradigm to y = A*x ?
        def forward2matvec(self, x):
            data = self.range.clone().zero()
            self.forward(False, x, data)
            return data.getNdArray()

        def adjoint2rmatvec(self, x):
            model = self.domain.clone().zero()
            self.adjoint(False, model, x)
            return model.getNdArray()

        Op.matvec = partial(forward2matvec, self)
        Op.rmatvec = partial(adjoint2rmatvec, self)
        is_squared = self.domain == self.range

        eigenvalues = ssl.eigsh(Op if is_squared else Op.H * Op, k=neigs, maxiter=niter)[0]
        return eigenvalues if is_squared else np.sqrt(eigenvalues)


################################
# OPERATIONS BETWEEN OPERATORS #
################################
class _CustomOperator(Operator):
    """Linear operator defined in terms of user-specified operations."""

    def __init__(self, domain, range, forward_function, adjoint_function):
        super(_CustomOperator, self).__init__(domain, range)
        self.forward_function = forward_function
        self.adjoint_function = adjoint_function

    def forward(self, add, model, data):
        return self.forward_function(add, model, data)

    def adjoint(self, add, model, data):
        return self.adjoint_function(add, model, data)


class _sumOperator(Operator):
    """
    Sum of two operators
        C = A + B
        C.H = A.H + B.H
    """

    def __init__(self, A, B):
        """Sum operator constructor"""
        if not isinstance(A, Operator) or not isinstance(B, Operator):
            raise TypeError('Both operands have to be a Operator')
        if not A.range.checkSame(B.range) or not A.domain.checkSame(B.domain):
            raise ValueError('Cannot add operators: shape mismatch')

        super(_sumOperator, self).__init__(A.domain, A.range)
        self.args = (A, B)

    def forward(self, add, model, data):
        self.checkDomainRange(model, data)
        self.args[0].forward(add, model, data)
        self.args[1].forward(True, model, data)

    def adjoint(self, add, model, data):
        self.checkDomainRange(model, data)
        self.args[0].adjoint(add, model, data)
        self.args[1].adjoint(True, model, data)


class _prodOperator(Operator):
    """
    Multiplication of two operators
    C = A * B
    C.H = B.H * A.H
    """

    def __init__(self, A, B):
        if not isinstance(A, Operator) or not isinstance(B, Operator):
            raise TypeError('Both operands have to be a Operator')
        if not A.range.checkSame(B.range) or not A.domain.checkSame(B.domain):
            raise ValueError('Cannot add operators: shape mismatch')

        super(_prodOperator, self).__init__(B.domain, A.range)
        self.args = (A, B)

    # TODO currently allocating-deallocating temp. Is it better to pre-allocate?
    def forward(self, add, model, data):
        self.checkDomainRange(model, data)
        temp = data.clone()
        self.args[1].forward(False, model, temp)
        self.args[0].forward(add, temp, data)
        del temp

    def adjoint(self, add, model, data):
        self.checkDomainRange(model, data)
        temp = model.clone()
        self.args[0].adjoint(False, temp, data)
        self.args[1].adjoint(add, model, temp)
        del temp


class _scaledOperator(Operator):
    """
    Scalar matrix multiplication
    """

    def __init__(self, A, const):
        if not isinstance(A, Operator):
            raise TypeError('Operator expected as A')
        if not type(const) in [int, float]:
            raise ValueError('scalar expected as const')
        super(_scaledOperator, self).__init__(A.domain, A.range)
        self.const = const
        self.op = A

    def forward(self, add, model, data):
        self.op.forward(add, model, data)
        data.scale(self.const)

    def adjoint(self, add, model, data):
        self.op.adjoint(add, model, data)
        model.scale(np.conj(self.const))


class Vstack(Operator):
    """
    Vertical stack of operators
        y1 = | A | x
        y2   | B |
    """

    def __init__(self, *args):
        """Constructor for the stacked operator"""

        self.ops = []
        for idx, op in enumerate(args):
            if type(op) is Vstack:
                self.ops += op.ops
            elif isinstance(op, Operator):
                self.ops.append(op)
            else:
                raise TypeError('Argument must be either Operator or Vstack')

        # check range
        self.n = len(self.ops)
        op_range = []
        for idx in range(self.n - 1):
            if not self.ops[idx].domain.checkSame(self.ops[idx + 1].domain):
                raise ValueError('Domain incompatibility between Op %d and Op %d' % (idx, idx + 1))
            op_range += [op.range]

        super(Vstack, self).__init__(domain=op.domain, range=superVector(op_range))

    def forward(self, add, model, data):
        """Forward operator Cm"""
        self.checkDomainRange(model, data)
        for idx in range(self.n):
            self.ops[idx].forward(add, model, data.vecs[idx])

    def adjoint(self, add, model, data):
        """Adjoint operator C'r = A'r1 + B'r2"""
        self.checkDomainRange(model, data)
        self.ops[0].adjoint(add, model, data.vecs[0])
        for idx in range(1, self.n):
            self.ops[idx].adjoint(True, model, data.vecs[idx])


class Hstack(Operator):
    """
    Horizontal stack of operators
        y = [A  B]  x1
                    x2
    """

    def __init__(self, *args):
        """Constructor for the stacked operator"""

        self.ops = []
        for op in args:
            if type(op) is Hstack:
                self.ops += op.ops
            elif isinstance(op, Operator):
                self.ops.append(op)
            else:
                raise TypeError('Argument must be either Operator or Hstack')

        # check domain
        self.n = len(self.ops)
        domain = []
        for idx in range(self.n - 1):
            if not self.ops[idx].range.checkSame(self.ops[idx + 1].range):
                raise ValueError('Range incompatibility between Op %d and Op %d' % (idx, idx + 1))
            domain += [op.domain]
        super(Hstack, self).__init__(domain=superVector(domain), range=op.range)

    def forward(self, add, model, data):
        self.checkDomainRange(model, data)
        for idx in range(self.n):
            self.ops[idx].forward(True, model.vecs[idx], data)

    def adjoint(self, add, model, data):
        self.checkDomainRange(model, data)
        self.ops[0].adjoint(add, model.vecs[0], data)
        for idx in range(1, self.n):
            self.ops[idx].adjoint(add, model.vecs[idx], data)


# TODO delete
Transpose = Operator.H
# class Transpose(Operator):
#     """
#     Class of transposition of provided operator
#     """
#
#     def __init__(self, op):
#         self.setDomainRange(op.range, op.domain)
#         self.op = op
#         return
#
#     def forward(self, add, model, data):
#         self.checkDomainRange(model, data)
#         self.op.adjoint(add, data, model)
#         return
#
#     def adjoint(self, add, model, data):
#         self.checkDomainRange(model, data)
#         self.op.forward(add, data, model)
#         return


class scalingOp(Operator):
    """scalar multiplication operator"""

    def __init__(self, domain, scalar):
        super(scalingOp, self).__init__(domain, domain)
        if not type(scalar) in [int, float]:
            raise ValueError('Scalar has to be (indeed) a scalar')
        self.scalar = scalar

    def forward(self, add, model, data):
        self.checkDomainRange(model, data)
        data.scaleAdd(model, 1. if add else 0., self.scalar)

    def adjoint(self, add, model, data):
        self.checkDomainRange(model, data)
        model.scaleAdd(data, 1. if add else 0., self.scalar)


class ZeroOp(Operator):
    """Zero matrix operator; useful for Jacobian matrices that are zeros"""

    def __init__(self, domain, range):
        super(ZeroOp, self).__init__(domain, range)

    def forward(self, add, model, data):
        self.checkDomainRange(model, data)
        if not add:
            data.zero()

    def adjoint(self, add, model, data):
        self.checkDomainRange(model, data)
        if not add:
            model.zero()


class IdentityOp(Operator):
    """Identity operator"""

    def __init__(self, domain):
        super(IdentityOp, self).__init__(domain, domain)

    def forward(self, add, model, data):
        self.checkDomainRange(model, data)
        if add:
            data.scaleAdd(model)
        else:
            data.copy(model)

    def adjoint(self, add, model, data):
        self.checkDomainRange(model, data)
        if add:
            model.scaleAdd(data)
        else:
            model.copy(data)


sumOperator = _sumOperator
# class sumOperator(Operator):
#     """
#     Sum of two operators
#         op = op1 + ... + opN
#     """
#
#     def __init__(self, ops):
#         """Sum operator constructor"""
#         if not np.all(isinstance(op, Operator) for op in ops):
#             raise TypeError('All operands must be a Operator')
#
#         if not len(set([op.range for op in ops])) == 1 or not len(
#                 set([op.domain for op in ops])) == 1:
#             raise ValueError('Cannot add operators: shape mismatch')
#
#         super(sumOperator, self).__init__()
#         self.ops = ops
#         self.setDomainRange(ops[0].domain, ops[0].range)
#
#     def forward(self, add, model, data):
#         self.checkDomainRange(model, data)
#         for idx in range(len(self.ops) - 1, 0, -1):
#             self.ops[idx].forward(add, model, data)
#
#     def adjoint(self, add, model, data):
#         self.checkDomainRange(model, data)
#         for idx in range(len(self.ops) - 1, 0, -1):
#             self.ops[idx].adjoint(add, model, data)


ChainOperator = _prodOperator
# class ChainOperator(Operator):
#     """
#     Product of operators
#         op = opN * ... * op1
#     """
#
#     def __init__(self, ops):
#         """Constructor of a Chain of operators (in order of application)"""
#         if not np.all(isinstance(op, Operator) for op in ops):
#             raise TypeError('All operands must be a Operator')
#         for idx in range(len(ops) - 1):
#             if not ops[idx].range.checkSame(ops[idx + 1].domain):
#                 raise ValueError('Op %d range differs from Op %d domain' % (idx, idx + 1))
#         super(ChainOperator, self).__init__()
#         self.setDomainRange(ops[0].domain, ops[-1].range)
#         self.ops = ops
#         self.temp = self.domain.clone().zero()
#
#     def forward(self, add, model, data):
#         """Forward operator BAm"""
#         self.checkDomainRange(model, data)
#
#         # first operator
#         self.ops[0].forward(False, model, self.temp)
#         # operators in the middle
#         for idx in range(1, len(self.ops) - 1):
#             self.ops[idx].forward(False, self.temp, self.temp)
#         # last operator
#         self.ops[-1].forward(add, self.temp, data)
#
#     def adjoint(self, add, model, data):
#         """Adjoint operator A'B'd"""
#         self.checkDomainRange(model, data)
#         # last operator
#         self.ops[-1].adjoint(False, self.temp, data)
#         # operators in the middle
#         for idx in range(len(self.ops) - 1, 0, -1):
#             self.ops[idx].adjoint(False, self.temp, self.temp)
#         # first operator
#         self.ops[0].adjoint(add, model, self.temp)


#######################
# NONLINEAR OPERATORS #
#######################

# Dummy function to use Non-linear operator class for Linear ones
def dummy_set_background(dummy_arg):
    """
    Dummy function to use Non-linear operator class for Linear ones (it takes one argument and does nothing)
    """
    return


class NLOperator(Operator):
    """
    Non-linear operator class
    """

    def __init__(self, nl_op, lin_op, set_background_func=dummy_set_background):
        """
           Constructor for non-linear operator class:
           nl_op				= [no default] - operator class;
                                Non-linear operator class where only the forward is overwritten
           lin_op				= [no default] - operator class;
                                Linear Jacobian operator class where only the forward is
                                overwritten (if not necessary, use pyOperator.ZeroOp)
           set_background_func	= [dummy_set_background] - function pointer;
                                Function to set the model vector on which the
                                Jacobian operator is evaluated
        """
        # Setting non-linear and linearized operators
        self.nl_op = nl_op
        self.lin_op = lin_op
        self.set_background = set_background_func
        # Checking if domain of the operators is the same
        if not nl_op.domain.checkSame(lin_op.domain):
            raise ValueError("ERROR! The two provided operators have different domains")
        if not nl_op.range.checkSame(lin_op.range):
            raise ValueError("ERROR! The two provided operators have different ranges")
        super(NLOperator, self).__init__(nl_op.domain, nl_op.range)

    def dotTest(self):
        """
        Raising an exception, dot-product test must be performed directly onto linear operator.
        """
        raise NotImplementedError("ERROR! Perform dot-product test directly on the linear operator.")


class _combNLOperator(NLOperator):
    """
    Combination of non-linear opeartors: f(g(m))
    """

    def __init__(self, A, B):
        """
        Constructor for non-linear operator class
        """
        # Checking if non-linear operators were provided
        if not (isinstance(A, NLOperator) and isinstance(B, NLOperator)):
            raise TypeError("ERROR! Provided operators must be NonLinearOperator instances")
        # Defining f(g(m))
        self.nl_op = _prodOperator(A.nl_op, B.nl_op)
        # Defining F(g(m0))G(m0)
        self.lin_op = _prodOperator(A.lin_op, B.lin_op)
        # Defining internal set_background functions
        self.set_background1 = A.set_background
        self.set_background2 = B.set_background
        # Defining non_linear operator g(m) for Jacobian definition
        self.g_nl_op = A.nl_op
        self.g_range_tmp = A.nl_op.range.clone()
        super(_combNLOperator, self).__init__(self.nl_op, self.lin_op)
        return

    def set_background(self, model):
        """
        Set background function for the chain of Jacobian matrices
        """
        # Setting G(m0)
        self.set_background1(model)
        # Setting F(g(m0))
        self.g_nl_op.forward(False, model, self.g_range_tmp)
        self.set_background2(self.g_range_tmp)
        return


class VstackNLOperator(NLOperator):
    """
    Stack of operators class
            | d1 |   | f(m) |
     h(m) = |    | = |      |
            | d2 |   | g(m) |
    """

    def __init__(self, nl_op1, nl_op2):
        """Constructor for the stacked operator"""
        # Checking if domain of the operators is the same
        if not (isinstance(nl_op1, NLOperator) and isinstance(nl_op2, NLOperator)):
            raise TypeError("Provided operators must be NonLinearOperator instances")
        self.nl_op1 = nl_op1  # f(m)
        self.nl_op2 = nl_op2  # g(m)
        # Defining f(g(m))
        self.nl_op = Vstack(nl_op1.nl_op, nl_op2.nl_op)
        # Defining F(g(m0))G(m0)
        self.lin_op = Vstack(nl_op1.lin_op, nl_op2.lin_op)
        # Defining internal set_background functions
        self.set_background1 = nl_op1.set_background
        self.set_background2 = nl_op2.set_background
        super(VstackNLOperator, self).__init__(self.nl_op, self.lin_op)

    def set_background(self, model):
        """
        Set background function for the stack of Jacobian matrices
        """
        # Setting F(m0)
        self.set_background1(model)
        # Setting G(m0)
        self.set_background2(model)
        return
