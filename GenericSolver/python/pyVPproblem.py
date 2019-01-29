#Module containing the definition of inverse problems where Variable-Projection method is used (Golub and Pereyra, 1973)
import pyProblem as pyProb
import pyOperator as pyOp
import pyVector as pyVec


class VpOperator(pyOp.Operator):
	"""
		Operator of the form: h(m_nl)m_lin, for Variable-projection method
	"""

	def __init__(self,h_nl,h_lin,set_nl,set_lin):
		"""
			Constructor for an operator with a linear and non-linear model component
			h_nl    = [no default] - non-linear operator class; Non-linear operator class
			h_lin   = [no default] - operator class; Linear operator class
			set_nl  = [no default] - class function pointer; Class function to set non-linear part within h_lin
			set_lin = [no default] - class function pointer; Class function to set linear part within h_nl
		"""
		if(not isinstance(h_nl,pyOp.NonLinearOperator)):
			raise TypeError("ERROR! Not provided a non-linear operator class for h_nl")
		self.h_nl=h_nl
		self.h_lin=h_lin
		#Checking the range spaces
		if(not h_nl.nl_op.range.checkSame(h_lin.range)):
			raise ValueError("ERROR! The two provided operators have different ranges")
		self.set_nl=set_nl
		self.set_lin=set_lin
		return

	def dotTest(self,verb=False,maxError=.0001):
		"""
		   Raising an exception, dot-product test must be performed directly onto linear operator and the Jacobian of h(m_nl).
		"""
		raise NotImplementedError("ERROR! Perform dot-product test directly onto linear operator and Jacobian of h(m_nl).")
		return



class ProblemL2VpReg(pyProb.Problem):
	"""
       Non-linear inverse problem in which part of the model parameters define a quadratic function
       The non-linear component is solved using the variable-projection method (Golub and Pereyra, 1973)
       Problem form: phi(m) = 1/2*|g(m_nl) + h(m_nl)m_lin - d|_2 + epsilon^2/2*|g'(m_nl) + h'(m_nl)m_lin - d'|_2
    """

	def __init__(self,model_nl,model_lin,h_op,data,lin_solver,g_op=None,g_op_reg=None,h_op_reg=None,data_reg=None,epsilon=None):
		"""
			Constructor for solving a inverse problem using the variable-projection method
			Required arguments:
			model_nl    = [no default] - vector class; Initial non-linear model component of the objective function
			model_lin   = [no default] - vector class; Initial quadritic (Linear) model component of the objective function
			h_op   		= [no default] - Vp operator class; Variable projection operator
			data   		= [no default] - vector class; Data vector
			lin_solver	= [no default] - solver class; Linear solver to invert for linear component of the model
			Optional arguments:
			g_op   		= [None] - non-linear operator class; Fully non-linear additional operator
			g_op_reg   	= [None] - non-linear operator class; Fully non-linear additional operator for regularization term
			h_op_reg	= [None] - Vp operator class; Variable projection operator for regularization term
			data_reg   	= [None] - vector class; Data vector for regularization term
			epsilon 	= [None] - float; Regularization term weight (must be provided if a regularization is needed)
		"""
		if(not isinstance(h_op,VpOperator)):
			raise TypeError("ERROR! Not provided an operator class for the variable projection problem")
		#Setting internal vector
		self.model=model_nl.clone()
		self.dmodel=model_nl.clone()
		self.dmodel.zero()
		#Copying the pointer to data vector
		self.data=data
		#Setting non-linear/linear operator
		self.h_op=h_op
		#Setting non-linear operator (if any)
		self.g_op=g_op
		#Verifying if a regularization is requested
		self.epsilon=epsilon
		#Residual vector
		if(self.epsilon != None):
			#Setting non-linear regularization operator
			self.g_op_reg=g_op_reg
			#Setting non-linear/linear operator
			self.h_op_reg=h_op_reg
			#Setting data term in regularization
			self.data_reg=data_reg
			#Creating regularization residual vector
			res_reg = None
			if(self.g_op_reg != None):
				res_reg = self.g_op_reg.nl_op.range.clone()
			elif(self.h_op_reg != None):
				res_reg = self.h_op_reg.h_lin.range.clone()
			elif(self.data_reg != None):
				res_reg = self.data_reg.clone()
			#Checking if a residual vector for the regularization term was created
			if(res_reg == None):
				raise ValueError("ERROR! If epsilon was provided, then a regularization term must be provided")
			self.res = pyVec.superVector(data.clone(),res_reg)
		else:
			self.res=data.clone()
		#Zeroing out the residual vector
		self.res.zero()
		#Dresidual vector
		self.dres=self.res.clone()
		#Gradient vector
		self.grad=self.dmodel.clone()
		#Setting default variables
		self.setDefaults()
		self.linear=False
		#Linear component of the inverted model
		self.model_lin = model_lin
        return

	def __del__(self):
		"""Default destructor"""
		return

	def resf(self,model):
		"""Method to return residual vector"""

		return self.res
	#
	# def gradf(self,model,res):
	# 	"""Method to return gradient vector g = F'r = F'(f(m) - d)"""
	# 	#Setting model point on which the F is evaluated
	# 	self.op.set_background(model)
	# 	#Computing F'r = g
	# 	self.op.lin_op.adjoint(False,self.grad,res)
	# 	return self.grad
	#
	# def dresf(self,model,dmodel):
	# 	"""Method to return residual vector dres = Fdm"""
	# 	#Setting model point on which the F is evaluated
	# 	self.op.set_background(model)
	# 	#Computing Fdm = dres
	# 	self.op.lin_op.forward(False,dmodel,self.dres)
	# 	return self.dres
	#
	def objf(self,res):
		"""Method to return objective function value 1/2*|g(m_nl) + h(m_nl)m_lin - d|_2 + epsilon^2/2*|g'(m_nl) + h'(m_nl)m_lin - d'|_2"""
		obj=0.5*res.dot(res)
		return obj
