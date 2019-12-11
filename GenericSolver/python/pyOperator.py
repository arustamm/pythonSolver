# Module containing the definition of the operator necessary for the solver class
# It takes vector objects from the pyVector class

from __future__ import division, print_function, absolute_import
import warnings
import time
from copy import deepcopy
import numpy as np
from pyVector import vector, superVector

# for __truediv__
from pyLinearSolver import LCGsolver
from pyProblem import ProblemL2Linear
from pyStopper import BasicStopper


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

	def __truediv__(self, other, niter=2000):
		"""x = A / y through CG"""

		if not self.range.checkSame(other):
			raise ValueError('Operator range and data domain mismatch')

		Stop = BasicStopper(niter=niter)
		P = ProblemL2Linear(model=self.domain.cloneSpace(), data=other, op=self)
		Solver = LCGsolver(Stop)
		Solver.setDefaults()
		Solver.run(P, verbose=False)

		return P.model

	# main function for all kinds of multiplication
	def dot(self, other):
		"""Matrix-matrix or matrix-vector or matrix-scalar multiplication."""
		if isinstance(other, Operator):  # A * B
			return _prodOperator(self, other)
		elif type(other) in [int, float]:  # A * c | # TODO I want to call also c * A (but in the latter __mul__ is of scalar)
			return _scaledOperator(self, other)
		elif isinstance(other, vector) or isinstance(other, superVector):  # A * x
			temp = self.range.clone()
			self.forward(False, other, temp)
			return temp
		else:
			raise TypeError('Expected Operator, (super)Vector or scalar, got %r' % other)

	def getDomain(self):
		"""Function to return operator domain"""
		return self.domain

	def getRange(self):
		"""Function to return operator range"""
		return self.range

	def setDomainRange(self,domain,range):
		"""Function to set (cloning space) domain and range of the operator"""
		self.domain = domain.cloneSpace()
		self.range = range.cloneSpace()
		return

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
				print("Estimated maximum eigenvalue at iter %d: %.2e" % (iiter, eigen))
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
					print("Estimated minimum eigenvalue at iter %d: %.2e"
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
		Function to perform dot-product test.
		:param verbose  : boolean; Flag to print information to screen as the method is being run [False]
		:param tol      : float; The function throws a Warning if the relative error is greater than maxError [1e-4]
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
			print(" Runs in: %s seconds" % (end - start))
			print("Applying adjoint operator add=False")
		start = time.time()
		self.adjoint(False, d2, r1)
		end = time.time()
		if verbose:
			print(" Runs in: %s seconds" % (end - start))

		# Computing dot products
		dt1 = d1.dot(d2)
		dt2 = r1.dot(r2)

		# Dot-product testing
		if verbose:
			print("Dot products add=False: domain=%.2e range=%.2e " % (dt1, dt2))
			print("Absolute error: %.2e" % (abs(dt1 - dt2)))
			print("Relative error: %.2e \n" % (abs((dt1 - dt2) / dt2)))
		if abs((dt1 - dt2) / dt1) > tol:
			# Deleting temporary vectors
			del d1, d2, r1, r2
			raise Warning("Dot products failure add=False; relative error greater than tolerance of %.2e" % tol)

		# Applying forward and adjoint operators with add=True
		if verbose:
			print("\nApplying forward operator add=True")
		start = time.time()
		self.forward(True, d1, r2)
		end = time.time()
		if verbose:
			print(" Runs in: %s seconds" % (end - start))
			print("Applying adjoint operator add=True")
		start = time.time()
		self.adjoint(True, d2, r1)
		end = time.time()
		if verbose:
			print(" Runs in: %s seconds" % (end - start))

		# Computing dot products
		dt1 = d1.dot(d2)
		dt2 = r1.dot(r2)

		if verbose:
			print("Dot products add=True: domain=%.2e range=%.2e " % (dt1, dt2))
			print("Absolute error: %.2e" % (abs(dt1 - dt2)))
			print("Relative error: %.2e \n" % (abs((dt1 - dt2) / dt2)))
		if abs((dt1 - dt2) / dt1) > tol:
			# Deleting temporary vectors
			del d1, d2, r1, r2
			raise Warning("Dot products failure add=True; relative error greater than tolerance of %.2e" % tol)

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
	
	def __str__(self):
		return self.args[0].__str__()[:3]+"+"+self.args[0].__str__()[:4]
	
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
		if not A.domain.checkSame(B.range):
			raise ValueError('Cannot multiply operators: shape mismatch')
		super(_prodOperator, self).__init__(B.domain, A.range)
		self.args = (A, B)
		self.temp = B.getRange().clone()
	
	def __str__(self):
		return self.args[0].__str__()[:3]+"*"+self.args[0].__str__()[:4]
	
	def forward(self, add, model, data):
		self.checkDomainRange(model, data)
		self.args[1].forward(False, model, self.temp)
		self.args[0].forward(add, self.temp, data)

	def adjoint(self, add, model, data):
		self.checkDomainRange(model, data)
		self.args[0].adjoint(False, self.temp, data)
		self.args[1].adjoint(add, model, self.temp)


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
	
	def __str__(self):
		op_name = self.op.__str__().replace(" ","")
		l = len(op_name)
		if l <= 6:
			name = "sc"+op_name+""*(6-l)
		else:
			name = "sc"+op_name[:6]
		return name
		
	def forward(self, add, model, data):
		self.op.forward(add, model.clone().scale(self.const), data)

	def adjoint(self, add, model, data):
		self.op.adjoint(add, model, data.clone().scale(np.conj(self.const)))


class Vstack(Operator):
	"""
	Vertical stack of operators
		y1 = | A | x
		y2   | B |
	"""

	def __init__(self, *args):
		"""Constructor for the stacked operator"""

		self.ops = []
		for _, arg in enumerate(args):
			if arg is None:
				continue
			elif isinstance(arg, Operator):
				self.ops.append(arg)
			elif isinstance(arg, list):
				for op in arg:
					if op is None:
						continue
					elif isinstance(op, Operator):
						self.ops.append(op)
			else:
				raise TypeError('Argument must be either Operator or Vstack')

		# check range
		self.n = len(self.ops)
		op_range = []
		for idx in range(self.n):
			if idx < self.n - 1:
				if not self.ops[idx].domain.checkSame(self.ops[idx + 1].domain):
					raise ValueError('Domain incompatibility between Op %d and Op %d' % (idx, idx + 1))
			op_range += [self.ops[idx].range]

		super(Vstack, self).__init__(domain=self.ops[0].domain, range=superVector(op_range))
	
	def __str__(self):
		return " VStack "
	
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
		for _, arg in enumerate(args):
			if arg is None:
				continue
			elif isinstance(arg, Operator):
				self.ops.append(arg)
			elif isinstance(arg, list):
				for op in arg:
					if op is None:
						continue
					elif isinstance(op, Operator):
						self.ops.append(op)
			else:
				raise TypeError('Argument must be either Operator or Hstack')

		# check domain
		self.n = len(self.ops)
		domain = []
		for idx in range(self.n):
			if idx < self.n - 1:
				if not self.ops[idx].range.checkSame(self.ops[idx + 1].range):
					raise ValueError('Range incompatibility between Op %d and Op %d' % (idx, idx + 1))
			domain += [self.ops[0].domain]
		super(Hstack, self).__init__(domain=superVector(domain), range=self.ops[0].range)
	
	def __str__(self):
		return " HStack "
	
	def forward(self, add, model, data):
		self.checkDomainRange(model, data)
		self.ops[0].forward(add, model.vecs[0], data)
		for idx in range(1, self.n):
			self.ops[idx].forward(True, model.vecs[idx], data)

	def adjoint(self, add, model, data):
		self.checkDomainRange(model, data)
		self.ops[0].adjoint(add, model.vecs[0], data)
		for idx in range(1, self.n):
			self.ops[idx].adjoint(add, model.vecs[idx], data)


class scalingOp(Operator):
	"""scalar multiplication operator"""

	def __init__(self, domain, scalar):
		super(scalingOp, self).__init__(domain, domain)
		if not np.isscalar(scalar):
			raise ValueError('scalar has to be (indeed) a scalar variable')
		self.scalar = scalar
	
	def __str__(self):
		return "Scaling "
	
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
	
	def __str__(self):
		return "  Zero  "
	
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
	
	def __str__(self):
		return "Identity"
	
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


class DiagonalOp(Operator):
	"""Diagonal operator for performing element-wise multiplication"""

	def __init__(self, diag):
		# if not isinstance(diag, vector):
		#     raise TypeError('diag has to be a vector')
		super(DiagonalOp, self).__init__(diag, diag)
		self.diag = diag
		
	def __str__(self):
		return "Diagonal"

	def forward(self, add, model, data):
		self.checkDomainRange(model, data)
		data.scaleAdd(model, 1. if add else 0.)
		data.multiply(self.diag)

	def adjoint(self, add, model, data):
		self.checkDomainRange(model, data)
		model.scaleAdd(data, 1. if add else 0.)
		model.multiply(self.diag)


# for backward compatibility
Transpose = Operator.H
sumOperator = _sumOperator
stackOperator = Vstack


def ChainOperator(A,B):
	"""
	 	Chain of two operators
				d = B A m
	"""
	return _prodOperator(B,A)

#######################
# NONLINEAR OPERATORS #
#######################

# Dummy function to use Non-linear operator class for Linear ones


def dummy_set_background(dummy_arg):
	"""
	Dummy function to use Non-linear operator class for Linear ones (it takes one argument and does nothing)
	"""
	return


class NonLinearOperator(Operator):
	"""
	Non-linear operator class
	"""

	def __init__(self, nl_op, lin_op=None, set_background_func=dummy_set_background):
		"""
		   Constructor for non-linear operator class:
		   nl_op                = [no default] - operator class;
								Non-linear operator class where only the forward is overwritten
		   lin_op               = [no default] - operator class;
								Linear Jacobian operator class where only the forward is
								overwritten (if not necessary, use pyOperator.ZeroOp)
		   set_background_func  = [dummy_set_background] - function pointer;
								Function to set the model vector on which the
								Jacobian operator is evaluated
		"""
		# Setting non-linear and linearized operators
		self.nl_op = nl_op
		self.lin_op = lin_op if lin_op != None else nl_op
		self.set_background = set_background_func
		# Checking if domain of the operators is the same
		if not self.nl_op.domain.checkSame(self.lin_op.domain):
			raise ValueError("ERROR! The two provided operators have different domains")
		if not self.nl_op.range.checkSame(self.lin_op.range):
			raise ValueError("ERROR! The two provided operators have different ranges")
		super(NonLinearOperator, self).__init__(self.nl_op.domain, self.nl_op.range)
	
	def dotTest(self):
		"""
		Raising an exception, dot-product test must be performed directly onto linear operator.
		"""
		raise NotImplementedError("ERROR! Perform dot-product test directly on the linear operator.")


class _combNonLinearOperator(NonLinearOperator):
	"""
	Combination of non-linear opeartors: f(g(m))
	"""

	def __init__(self, A, B):
		"""
		Constructor for non-linear operator class
		"""
		# Checking if non-linear operators were provided
		if not (isinstance(A, NonLinearOperator) and isinstance(B, NonLinearOperator)):
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
		super(_combNonLinearOperator, self).__init__(self.nl_op, self.lin_op)

	def set_background(self, model):
		"""
		Set background function for the chain of Jacobian matrices
		"""
		# Setting G(m0)
		self.set_background1(model)
		# Setting F(g(m0))
		self.g_nl_op.forward(False, model, self.g_range_tmp)
		self.set_background2(self.g_range_tmp)


class VstackNonLinearOperator(NonLinearOperator):
	"""
	Stack of operators class
			| d1 |   | f(m) |
	 h(m) = |    | = |      |
			| d2 |   | g(m) |
	"""

	def __init__(self, nl_op1, nl_op2):
		"""Constructor for the stacked operator"""
		# Checking if domain of the operators is the same
		if not (isinstance(nl_op1, NonLinearOperator) and isinstance(nl_op2, NonLinearOperator)):
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
		super(VstackNonLinearOperator, self).__init__(self.nl_op, self.lin_op)
	
	def __str__(self):
		return "NLVstack"
	
	def set_background(self, model):
		"""
		Set background function for the stack of Jacobian matrices
		"""
		# Setting F(m0)
		self.set_background1(model)
		# Setting G(m0)
		self.set_background2(model)


def main():
	from sys import path
	path.insert(0, '.')
	import numpy as np
	import pyVector

	# First test on scaling a vector
	x = pyVector.vectorIC(np.empty((100, 200))).set(1)
	y = x.clone()
	S = scalingOp(x, 10)
	S.forward(False, x, y)

	# Test add operator
	Z = ZeroOp(x, x)
	I = IdentityOp(x)
	sumOp = I + Z
	sumOp.forward(False, x, y)
	if x.isDifferent(y):
		print('sumOp not working')

	# Test prod operator
	I2 = I * 2
	I2.forward(False, x, y)
	z = x.clone()
	z * 2
	y.isDifferent(z)
	if y.isDifferent(z):
		print('prod not working')

	prod = I * Z
	y = prod * x  # I*Z*x
	z = x.clone().zero()
	y.isDifferent(z)
	if y.isDifferent(z):
		print('prod not working')

	# Test combinations
	C = S * S + I
	C.forward(False, x, y)

	C.adjoint(False, z, x)
	C.H.forward(False, x, z)

	# Test inversion x = A / y
	x = pyVector.vectorIC(np.empty((100, 200))).set(1)
	y = pyVector.vectorIC(np.empty((100, 200))).set(10)
	A = scalingOp(x, 10)
	# y_hat = y.clone().zero()
	# A.forward(False, x, y_hat)
	# y.isDifferent(y_hat)
	x_hat = A / y
	if x.isDifferent(x_hat):
		print('inversion not working')

	# test superVector and operator stack
	V = Vstack(I, I*2)
	y = V.range.clone().set(1.)
	x = V.domain.clone().zero()
	V.adjoint(False, x, y)
	x = pyVector.vectorIC(np.ones((100, 200)))
	x2 = x.clone()
	x2 * 2
	y = pyVector.superVector(x.clone(), x2.clone())
	y_hat = y.clone()
	V.forward(False, x, y_hat)
	if y.isDifferent(y_hat):
		print('Vstack not working')
	x_inv = V / y
	if x_inv.isDifferent(x):
		print('Vstack inversion not working')

	H = Hstack(I, I*2)
	x = H.domain.clone().set(1.)    # x = 1, 1
	y = H.range.clone().zero()
	H.forward(False, x, y)          # y = 3
	x_hat = x.clone()
	H.adjoint(False, x_hat, y)      # x_hat = 3, 6
	x_inv = H / y  # TODO not working but probably it is correct:
				   #  we have a number of solutions that is twice the number of equations!
				   #  add a dimensionality check to the inversion (or better, a "onto" attribute to the operator
	if x.isDifferent(x_inv):
		print('Hstack not working')
	#
	# H.dotTest(True)

	# test inversion on superVector
	x = pyVector.vectorIC(np.empty((100, 200)))
	xx = pyVector.superVector(x.clone()).set(1)
	yy = xx.clone().set(10)
	S = scalingOp(xx, 10)
	xx_inv = S / yy


if __name__ == '__main__':
	main()
