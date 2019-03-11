#Module containing the definition of the operator necessary for the solver class
#It takes vector objects from the pyVector class
import pyVector as Vec
import time


class Operator:
	"""Abstract python operator class"""

	#Default class methods/functions
	def __init__(self):
		"""Generic class for operator"""
		return

	def __del__(self):
		"""Default destructor"""
		return

	def setDomainRange(self,domain,range):
		"""Function to set (cloning space) domain and range of the operator"""
		self.domain = domain.cloneSpace()
		self.range = range.cloneSpace()
		return

	def getDomain(self):
		"""Function to return operator domain"""
		return self.domain

	def getRange(self):
		"""Function to return operator range"""
		return self.range

	def checkDomainRange(self,model,data):
		"""Function to check model and data vector sizes"""
		if not self.domain.checkSame(model):
			raise ValueError("Provided model vector does not match operator domain")
		if not self.range.checkSame(data):
			raise ValueError("Provided data vector does not match operator range")
		return

	def powerMethod(self,verbose=False,tol=1e-8,n_iter=None,square=False,return_vec=False):
		"""
		   Function to estimate maximum eigenvalue of the operator
		   verbose    = [False] - boolean; Flag to print information to screen as the method is being run
		   tol    	  = [1e-6] - float; Tolerance on the change of the estimated eigenvalues
		   n_iter  	  = [None] - int; Maximum number of operator applications (if not provided, the function will continue until the tolerance is reached)
		   square  	  = [False] - boolean; If True, only the forward will be applied (i.e., operator is a square matrix).
		   return_vec = [False] - boolean; If True, the function will return the estimated eigenvector as well
		"""
		#Cloning input and output vectors
		if(verbose): print("Running power method to estimate maximum eigenvalue (operator L2 norm)")
		x = self.domain.clone()
		if(not square):
			if(verbose): print("Note: operator is not square, the eigenvalue is associated to A'A not A!")
			d_temp = self.range.clone()
		y = self.domain.clone()
		#randomize the input vector
		x.rand()
		x.scale(1.0/x.norm()) #Normalizing the initial vector
		y.zero()
		iter = 0
		eigen  = 0.0 #Current estimated eigenvalue
		eigen_old = 0.0 #Previous estimated eigenvalue
		#Starting the power iteration loop
		if(verbose): print("Starting iterative process")
		while True:
			#Applying adjoint if forward not square
			if(not square):
				self.forward(False,x,d_temp) #d = A x
				self.adjoint(False,y,d_temp) #y = A' d = A' A x
			else:
				self.forward(False,x,y)		 #y = A x

			#Estimating eigenvalue (Rayleigh quotient)
			eigen = x.dot(y)				 #eigen_i = x' A x / (x'x = 1.0)
			#x = y
			x.copy(y)
			#Normalization of the operator
			x.scale(1.0/x.norm())
			#Stopping criteria (first number of iterations and then tolerance)
			iter += 1
			if(verbose): print("	Estimated maximum eigenvalue at iter %s: %s"%(iter,eigen))
			if(n_iter != None):
				if(iter >= n_iter):
					if(verbose): print("Maximum number of iteration reached! Stopping iterative process!")
					break
			#Checking change on the eigenvalue estimated value
			if(abs(eigen-eigen_old) <  abs(tol * eigen_old)):
				if(verbose): print("Tolerance value reached! Stopping iterative process!")
				break
			#eigen_(i-1) = eigen_i
			eigen_old = eigen
		if(return_vec): return eigen,x
		return eigen

	def dotTest(self,verb=False,maxError=.0001):
		"""Function to perform dot-product test
		   If passing the dot-product test, the function does not throw a Warning
		   Default relative error 10^-4
		"""
		if(verb): print("Dot-product test of forward and adjoint operators")
		if(verb): print("-------------------------------------------------")
		#Allocating temporary vectors for dot-product test
		d1=self.domain.clone()
		d2=self.domain.clone()
		r1=self.range.clone()
		r2=self.range.clone()

		#Randomize the input vectors
		d1.rand()
		r1.rand()

		#Applying forward and adjoint operators with add=False
		if(verb): print("Applying forward operator add=False")
		start = time.time()
		self.forward(False,d1,r2)
		end = time.time()
		if(verb): print("	Runs in: %s seconds"%(end-start))
		if(verb): print("Applying adjoint operator add=False")
		start = time.time()
		self.adjoint(False,d2,r1)
		end = time.time()
		if(verb): print("	Runs in: %s seconds"%(end-start))

		#Computing dot products
		dt1=d1.dot(d2)
		dt2=r1.dot(r2)

		#Dot-product testing
		if(verb): print("Dot products add=False: domain=%s range=%s "%(dt1,dt2))
		if(verb): print("Absolute error: %s"%(abs(dt1-dt2)))
		if(verb): print("Relative error: %s \n"%(abs((dt1-dt2)/dt2)))
		if (abs((dt1-dt2)/dt1) > maxError):
			#Deleting temporary vectors
			del d1,d2,r1,r2
			raise Warning("Dot products failure add=False; relative error greater than tolerance of %s"%(maxError))

		#Applying forward and adjoint operators with add=True
		if(verb): print("\nApplying forward operator add=True")
		start = time.time()
		self.forward(True,d1,r2)
		end = time.time()
		if(verb): print("	Runs in: %s seconds"%(end-start))
		if(verb): print("Applying adjoint operator add=True")
		start = time.time()
		self.adjoint(True,d2,r1)
		end = time.time()
		if(verb): print("	Runs in: %s seconds"%(end-start))

		#Computing dot products
		dt1=d1.dot(d2)
		dt2=r1.dot(r2)

		if(verb): print("Dot products add=True: domain=%s range=%s "%(dt1,dt2))
		if(verb): print("Absolute error: %s"%(abs(dt1-dt2)))
		if(verb): print("Relative error: %s \n"%(abs((dt1-dt2)/dt2)))
		if(abs((dt1-dt2)/dt1) > maxError):
			#Deleting temporary vectors
			del d1,d2,r1,r2
			raise Warning("Dot products failure add=True; relative error greater than tolerance of %s"%(maxError))

		if(verb): print("-------------------------------------------------")

		#Deleting temporary vectors
		del d1,d2,r1,r2
		return

	#Class methods/functions to be overridden
	def forward(self,add,model,data):
		"""Forward operator"""
		raise NotImplementedError("Forward must be overwritten")
		return

	def adjoint(self,add,model,data):
		"""Adjoint operator"""
		raise NotImplementedError("Adjoint must be overwritten")
		return

class Transpose(Operator):
	"""
	   Class of transposition of provided operator
	"""
	def __init__(self,op):
		self.setDomainRange(op.range,op.domain)
		self.op=op
		return

	def forward(self,add,model,data):
		self.checkDomainRange(model,data)
		self.op.adjoint(add,data,model)
		return

	def adjoint(self,add,model,data):
		self.checkDomainRange(model,data)
		self.op.forward(add,data,model)
		return


class scalingOp(Operator):
	"""Simple operator for testing Operator class"""
	def __init__(self,domain,scalar):
		assert(isinstance(scalar,float))
		self.setDomainRange(domain,domain)
		self.scalar=scalar
		return

	def forward(self,add,model,data):
		self.checkDomainRange(model,data)
		sc=0.
		if add: sc=1.
		data.scaleAdd(model,sc,self.scalar)
		return

	def adjoint(self,add,model,data):
		self.checkDomainRange(model,data)
		sc=0.
		if add: sc=1.
		model.scaleAdd(data,sc,self.scalar)
		return

class ZeroOp(Operator):
	"""Zero matrix operator; useful for Jacobian matrices that are zeros"""

	def __init__(self,domain,range):
		self.setDomainRange(domain,range)
		return

	def forward(self,add,model,data):
		self.checkDomainRange(model,data)
		if (not add):
			data.zero()
		return

	def adjoint(self,add,model,data):
		self.checkDomainRange(model,data)
		if (not add):
			model.zero()
		return


class IdentityOp(Operator):
	"""Identity operator"""

	def __init__(self,domain):
		self.setDomainRange(domain,domain)
		return

	def forward(self,add,model,data):
		self.checkDomainRange(model,data)
		if add:
			data.scaleAdd(model)
		else:
			data.copy(model)
		return

	def adjoint(self,add,model,data):
		self.checkDomainRange(model,data)
		if add:
			model.scaleAdd(data)
		else:
			model.copy(data)
		return

class ChainOperator(Operator):
	"""
			Chain of two operator class
				d = B A m
	"""
	def __init__(self,op1,op2):
		"""Constructor of a Chain of operators"""
		self.setDomainRange(op1.domain,op2.range)
		#Checking if domain of the operators is the same
		if(not op1.range.checkSame(op2.domain)):
			raise ValueError("ERROR! Domain and range of the two operators are not compatible")
		self.tmp_vec = op1.range.clone()
		self.op1 = op1
		self.op2 = op2
		return

	def forward(self,add,model,data):
		"""Forward operator BAm"""
		self.checkDomainRange(model,data)
		self.op1.forward(False,model,self.tmp_vec)
		self.op2.forward(add,self.tmp_vec,data)
		return

	def adjoint(self,add,model,data):
		"""Adjoint operator A'B'd"""
		self.checkDomainRange(model,data)
		self.op2.adjoint(False,self.tmp_vec,data)
		self.op1.adjoint(add,model,self.tmp_vec)
		return

class stackOperator(Operator):
	"""
			  Stack of operators class
					| d1 |   | A |
			   Cm = |    | = |   | m
					| d2 |   | B |
	"""

	def __init__(self,op1,op2):
		"""Constructor for the stacked operator"""
		#Checking if domain of the operators is the same
		if(not op1.domain.checkSame(op2.domain)):
			raise ValueError("ERROR! The two provided operators have different domains")
		self.setDomainRange(op1.domain,Vec.superVector(op1.range,op2.range))
		self.op1=op1 #A
		self.op2=op2 #B
		return

	def forward(self,add,model,data):
		"""Forward operator Cm"""
		self.checkDomainRange(model,data)
		# d1 = Am
		self.op1.forward(add,model,data.vec1)
		# d2 = Bm
		self.op2.forward(add,model,data.vec2)
		return

	def adjoint(self,add,model,data):
		"""Adjoint operator C'r = A'r1 + B'r2"""
		self.checkDomainRange(model,data)
		# m = A'd1
		self.op1.adjoint(add,model,data.vec1)
		# m += B'd2
		self.op2.adjoint(True,model,data.vec2)
		return

#Dummy function to use Non-linear operator class for Linear ones
def dummy_set_background(dummy_arg):
	"""
	   Dummy function to use Non-linear operator class for Linear ones (it takes one argument and does nothing)
	"""
	return

class NonLinearOperator(Operator):
	"""
		Non-linear operator class
	"""

	def __init__(self,nl_op,lin_op,set_background_func=dummy_set_background):
		"""
		   Constructor for non-linear operator class:
		   nl_op				= [no default] - operator class; Non-linear operator class where only the forward is overwritten
		   lin_op				= [no default] - operator class; Linear Jacobian operator class where only the forward is overwritten (if not necessary, use pyOperator.ZeroOp)
		   set_background_func	= [dummy_set_background] - function pointer; Function to set the model vector on which the Jacobian operator is evaluated
		"""
		#Setting non-linear and linearized operators
		self.nl_op = nl_op
		self.lin_op = lin_op
		self.set_background = set_background_func
		#Checking if domain of the operators is the same
		if(not nl_op.domain.checkSame(lin_op.domain)):
			raise ValueError("ERROR! The two provided operators have different domains")
		if(not nl_op.range.checkSame(lin_op.range)):
			raise ValueError("ERROR! The two provided operators have different ranges")
		self.setDomainRange(nl_op.domain,nl_op.range)
		return

	def dotTest(self):
		"""
		   Raising an exception, dot-product test must be performed directly onto linear operator.
		"""
		raise NotImplementedError("ERROR! Perform dot-product test directly onto linear operator.")
		return

class CombNonlinearOp(NonLinearOperator):
	"""
	   Combination of non-linear opeartors: f(g(m))
	"""
	def __init__(self,nl_op1,nl_op2):
		"""
		   Constructor for non-linear operator class
		"""
		#Checking if non-linear operators were provided
		if(not (isinstance(nl_op1,NonLinearOperator) and isinstance(nl_op2,NonLinearOperator))):
			raise TypeError("ERROR! Provided operators must be NonLinearOperator instances")
		#Defining f(g(m))
		self.nl_op = ChainOperator(nl_op1.nl_op,nl_op2.nl_op)
		#Defining F(g(m0))G(m0)
		self.lin_op = ChainOperator(nl_op1.lin_op,nl_op2.lin_op)
		#Defining internal set_background functions
		self.set_background1 = nl_op1.set_background
		self.set_background2 = nl_op2.set_background
		#Defining non_linear operator g(m) for Jacobian definition
		self.g_nl_op = nl_op1.nl_op
		self.g_range_tmp = nl_op1.nl_op.range.clone()
		return

	def set_background(self,model):
		"""
		   Set background function for the chain of Jacobian matrices
		"""
		#Setting G(m0)
		self.set_background1(model)
		#Setting F(g(m0))
		self.g_nl_op.forward(False,model,self.g_range_tmp)
		self.set_background2(self.g_range_tmp)
		return

class stackNonOperator(NonLinearOperator):
	"""
			Stack of operators class
					| d1 |   | f(m) |
			 h(m) = |    | = |      |
					| d2 |   | g(m) |
	"""

	def __init__(self,nl_op1,nl_op2):
		"""Constructor for the stacked operator"""
		#Checking if domain of the operators is the same
		if(not (isinstance(nl_op1,NonLinearOperator) and isinstance(nl_op2,NonLinearOperator))):
			raise TypeError("ERROR! Provided operators must be NonLinearOperator instances")
		self.nl_op1=nl_op1 #f(m)
		self.nl_op2=nl_op2 #g(m)
		#Defining f(g(m))
		self.nl_op = stackOperator(nl_op1.nl_op,nl_op2.nl_op)
		#Defining F(g(m0))G(m0)
		self.lin_op = stackOperator(nl_op1.lin_op,nl_op2.lin_op)
		#Defining internal set_background functions
		self.set_background1 = nl_op1.set_background
		self.set_background2 = nl_op2.set_background
		return

	def set_background(self,model):
		"""
		   Set background function for the stack of Jacobian matrices
		"""
		#Setting F(m0)
		self.set_background1(model)
		#Setting G(m0)
		self.set_background2(model)
		return
