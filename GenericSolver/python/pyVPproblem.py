#Module containing the definition of inverse problems where Variable-Projection method is used (Golub and Pereyra, 1973)
import pyProblem as pyProb
import pyOperator as pyOp
import pyVector as pyVec


class VpOperator(pyOp.Operator):
	"""
		Operator of the form: h(m_nl)m_lin, for Variable-projection method
	"""

	def __init__(self,h_nl,h_lin,set_nl,set_lin_jac,set_lin=None):
		"""
			Constructor for an operator with a linear and non-linear model component
			Required arguments:
			h_nl    	= [no default] - non-linear operator class; Non-linear operator class
			h_lin   	= [no default] - operator class; Linear operator class
			set_nl  	= [no default] - class function pointer; Class function to set non-linear part within h_lin
			set_lin_jac = [no default] - class function pointer; Class function to set linear part within the Jacobian h_nl
			#Optional arguments:
			set_lin 	= [None] - class function pointer; Class function to set linear part within h_nl (not used during an inversion if ProblemL2VpReg is used)
		"""
		if(not isinstance(h_nl,pyOp.NonLinearOperator)):
			raise TypeError("ERROR! Not provided a non-linear operator class for h_nl")
		self.h_nl=h_nl
		self.h_lin=h_lin
		#Checking the range spaces
		if(not h_nl.nl_op.range.checkSame(h_lin.range)):
			raise ValueError("ERROR! The two provided operators have different ranges")
		self.set_nl=set_nl #Function to set the non-linear component of the h(m_nl)
		self.set_lin_jac=set_lin_jac #Function to set the non-linear component of the Jacobian H(m_nl;m_lin)
		self.set_lin=set_lin #Function to set the non-linear component h(m_nl)m_lin
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
			model_lin   = [no default] - vector class; Initial quadritic (Linear) model component of the objective function (will be zeroed out)
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
		if(not isinstance(h_op,VpOperator)):
			raise TypeError("ERROR! Provide a VpOperator operator class for h_op")
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
				if(not isinstance(h_op_reg,VpOperator)):
					raise TypeError("ERROR! Provide a VpOperator operator class for h_op_reg")
				res_reg = self.h_op_reg.h_lin.range.clone()
			elif(self.data_reg != None):
				res_reg = self.data_reg.clone()
			#Checking if a residual vector for the regularization term was created
			if(res_reg == None):
				raise ValueError("ERROR! If epsilon is provided, then a regularization term must be provided")
			self.res = pyVec.superVector(data.clone(),res_reg)
		else:
			self.res=data.clone()
		#Instantiating linear inversion problem
		if(self.h_op_reg != None):
			self.vp_linear_prob = pyProb.ProblemL2LinearReg(self.model_lin,self.data,self.h_op.h_lin,self.epsilon,reg_op=self.h_op_reg.h_lin,prior_model=self.data_reg)
		else:
			self.vp_linear_prob = pyProb.ProblemL2Linear(self.model_lin,self.data,self.h_op.h_lin)
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
		self.model_lin = model_lin.clone()
		self.model_lin.zero()
		#Linear solver for inverting quadratic component
		self.lin_solver=lin_solver
        return

	def __del__(self):
		"""Default destructor"""
		return

	def resf(self,model):
		"""Method to return residual vector"""
		#Zero-out residual vector
		self.res.zero()
		###########################################
		#Applying full non-linear modeling operator
		res = self.res
		if(self.epsilon != None): res = self.res.vec1
		#Computing non-linear part g(m) (if any)
		if(self.g_op != None): self.g_op.nl_op(False,model,res)
		#Computing non-linear part g_reg(m) (if any)
		if(self.g_op_reg != None): self.g_op_reg.nl_op(False,model,self.res.vec2)

		##################################
		#Setting data for linear inversion
		# data term = data - [g(m) if any]
		res.scaleAdd(self.data,-1.0,1.0)
		#Setting data within first term
		self.vp_linear_prob.data=res

		# regularization data term = [g_reg(m) - data_reg if any]
		if(self.data_reg != None):
			self.res.vec2(self.data_reg,1.0,-1.0)
		#Data term for linear regularization term
		if("epsilon" in dir(self.vp_linear_prob)):
			self.res.vec2.scale(-1.0)
			self.vp_linear_prob.prior_model=self.res.vec2

		##################################
		#Running linear inversion
		#Getting fevals for saving linear inversion results
		fevals = self.get_fevals()
		#Setting initial linear inversion model
		self.model_lin.zero()
		self.vp_linear_prob.set_model(self.model_lin)
		#Setting non-linear component of the model
		self.h_op.set_nl(model)
		if(self.h_op_reg != None):
			self.h_op_reg.set_nl(model)
		#Resetting inversion problem variables
		self.vp_linear_prob.setDefaults()
		self.lin_solver.run(self.vp_linear_prob)
		#Copying inverted linear optimal model
		self.model_lin.copy(self.vp_linear_prob.get_model())

		##################################
		#Obtaining the residuals
		if((self.epsilon != None) and not("epsilon" in dir(self.vp_linear_prob))):
			#Regularization contains a non-linear operator only
			self.res.vec1.copy(self.vp_linear_prob.get_res())
			self.res.vec2.scale(self.epsilon)
		else:
			self.res.copy(self.vp_linear_prob.get_res())
		return self.res

	def gradf(self,model,res):
		"""
		   Method to return gradient vector
		   grad= [G(m)' + H(m_nl;m_lin)'] r_d + epsilon * [G'(m_nl)' + H'(m_nl;m_lin)'] r_m
		"""
		#Zero-out gradient vector
		self.grad.zero()
		#Setting the optimal linear model component and background of the Jacobian matrices
		self.h_op.set_lin_jac(self.model_lin) #H(_,m_lin_opt)
		self.h_op.h_nl.set_background(model) #H(m_nl,m_lin_opt)
		if(self.h_op_reg != None):
			self.h_op_reg.set_lin_jac(self.model_lin) #H'(_,m_lin_opt)
			self.h_op_reg.h_nl.set_background(model) #H'(m_nl,m_lin_opt)
		if(self.g_op != None): self.g_op.set_background(model) #G(m_nl)
		if(self.g_op_reg != None): self.g_op_reg.set_background(model) #G'(m_nl)
		#Computing contribuition from the regularization term (if any)
		if(self.epsilon != None):
			# G'(m_nl)' r_m
			if(self.g_op_reg != None): self.g_op_reg.lin_op.adjoint(False,self.grad,res.vec2)
			# H'(m_nl,m_lin_opt)' r_m
			if(self.h_op_reg != None): self.h_op_reg.h_nl.lin_op.adjoint(True,self.grad,res.vec2)
			# epsilon * [G'(m_nl)' + H'(m_nl,m_lin_opt)'] r_m
			self.grad.scale(self.epsilon)
		res = self.res
		if(self.epsilon != None): res = self.res.vec1
		# G(m_nl)' r_d
		if(self.g_op != None): self.g_op.lin_op.adjoint(True,self.grad,res)
		# H(m_nl,m_lin_opt)' r_d
		self.h_op.h_nl.lin_op.adjoint(True,self.grad,res)
		return self.grad

	def dresf(self,model,dmodel):
		"""Method to return residual vector dres (Not currently supported)"""
		raise NotImplementedError("ERROR! dresf is not currently supported! Provide an initial step-length value different than zero.")
		return self.dres

	def objf(self,res):
		"""Method to return objective function value 1/2*|g(m_nl) + h(m_nl)m_lin - d|_2 + epsilon^2/2*|g'(m_nl) + h'(m_nl)m_lin - d'|_2"""
		obj=0.5*res.dot(res)
		return obj
