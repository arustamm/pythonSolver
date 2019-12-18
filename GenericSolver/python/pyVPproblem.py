#Module containing the definition of inverse problems where the Variable-Projection method is used (Golub and Pereyra, 1973)
import pyProblem as pyProb
import pyOperator as pyOp
import pyVector as pyVec
from math import isnan


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
			set_lin_jac = [no default] - class function pointer; Class function to set linear part within the Jacobian h_nl (if not necessary, use pyOperator.dummy_set_background)
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

	def __init__(self,model_nl,lin_model,h_op,data,lin_solver,g_op=None,g_op_reg=None,h_op_reg=None,data_reg=None,epsilon=None,minBound=None,maxBound=None,boundProj=None,prec=None):
		"""
			Constructor for solving a inverse problem using the variable-projection method
			Required arguments:
			model_nl    = [no default] - vector class; Initial non-linear model component of the objective function
			lin_model   = [no default] - vector class; Initial quadritic (Linear) model component of the objective function (will be zeroed out)
			h_op   		= [no default] - Vp operator class; Variable projection operator
			data   		= [no default] - vector class; Data vector
			lin_solver	= [no default] - solver class; Linear solver to invert for linear component of the model
			Optional arguments:
			g_op   		= [None] - non-linear operator class; Fully non-linear additional operator
			g_op_reg   	= [None] - non-linear operator class; Fully non-linear additional operator for regularization term
			h_op_reg	= [None] - Vp operator class; Variable projection operator for regularization term
			data_reg   	= [None] - vector class; Data vector for regularization term
			epsilon 	= [None] - float; Regularization term weight (must be provided if a regularization is needed)
			minBound	= [None] - vector class; Minimum value bounds
 		    maxBound	= [None] - vector class; Maximum value bounds
 		    boundProj	= [None] - Bounds class; Class with a function "apply(input_vec)" to project input_vec onto some convex set
			prec       	= [None] - linear operator class; Preconditioning matrix for VP problem
			####################################################################################################################################
			Note that to save the results of the linear inversion the user has to specify the saving parameters within the setDefaults of the
			linear solver. The results can only be saved on files. To the prefix specified within the lin_solver f_eval_# will be added.
		"""
		if(not isinstance(h_op,VpOperator)):
			raise TypeError("ERROR! Not provided an operator class for the variable projection problem")
		#Setting the bounds (if any)
		super(ProblemL2VpReg,self).__init__(minBound,maxBound,boundProj)
		#Setting internal vector
		self.model=model_nl.clone()
		self.dmodel=model_nl.clone()
		self.dmodel.zero()
		#Linear component of the inverted model
		self.lin_model = lin_model.clone()
		self.lin_model.zero()
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
		#Setting non-linear regularization operator
		self.g_op_reg=g_op_reg
		#Setting non-linear/linear operator
		self.h_op_reg=h_op_reg
		#Setting data term in regularization
		self.data_reg=data_reg
		if(self.h_op_reg != None and self.epsilon == None):
			raise ValueError("ERROR! Epsilon value must be provided if a regularization term is requested.")
		#Residual vector
		if(self.epsilon != None):
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
			#Objective function terms (useful to analyze each term)
			self.obj_terms=[None,None]
		else:
			self.res=data.clone()
		#Instantiating linear inversion problem
		if(self.h_op_reg != None):
			self.vp_linear_prob = pyProb.ProblemL2LinearReg(self.lin_model,self.data,self.h_op.h_lin,self.epsilon,reg_op=self.h_op_reg.h_lin,prior_model=self.data_reg,prec=prec)
		else:
			self.vp_linear_prob = pyProb.ProblemL2Linear(self.lin_model,self.data,self.h_op.h_lin,prec=prec)
		#Zeroing out the residual vector
		self.res.zero()
		#Dresidual vector
		self.dres=self.res.clone()
		#Gradient vector
		self.grad=self.dmodel.clone()
		#Setting default variables
		self.setDefaults()
		self.linear=False
		#Linear solver for inverting quadratic component
		self.lin_solver=lin_solver
		self.lin_solver.flush_memory = True
		self.lin_solver_prefix = self.lin_solver.prefix
		self.vp_linear_prob.linear=True
		return

	def __del__(self):
		"""Default destructor"""
		return

	def estimate_epsilon(self,verbose=False,logger=None):
		"""Method returning epsilon that balances the two terms of the objective function"""
		if(self.epsilon == None):
			raise ValueError("ERROR! Problem is not regularized, cannot evaluate epsilon value!")
		if(self.g_op_reg != None and self.h_op_reg == None):
			#Problem is non-linearly regularized
			msg="Epsilon Scale evaluation"
			if(verbose): print(msg)
			if(logger): logger.addToLog("REGULARIZED PROBLEM log file\n"+msg)
			#Keeping the initial model vector
			prblm_mdl = self.get_model()
			#Keeping user-predefined epsilon if any
			epsilon = self.epsilon
			#Setting epsilon to one to evaluate the scale
			self.epsilon=1.0
			prblm_res = self.get_res(prblm_mdl)	#Compute residual arising from the gradient
			#Balancing the two terms of the objective function
			res_data_norm=prblm_res.vecs[0].norm()
			res_model_norm=prblm_res.vecs[1].norm()
			if (isnan(res_model_norm) or isnan(res_data_norm)):
				raise ValueError("ERROR! Obtained NaN: Residual-data-side-norm = %s, Residual-model-side-norm = %s"%(res_data_norm,res_model_norm))
			if(res_model_norm == 0.0):
				msg = "Model residual component norm is zero, cannot find epsilon scale! Provide a different initial model"
				if(logger): logger.addToLog(msg)
				raise ValueError(msg)
			#Resetting user-predefined epsilon if any
			self.epsilon = epsilon
			epsilon_balance = res_data_norm/res_model_norm
			#Resetting problem
			self.setDefaults()
			msg = "	Epsilon balancing the the two objective function terms is: %s"%(epsilon_balance)
			if(verbose): print(msg)
			if(logger): logger.addToLog(msg+"\nREGULARIZED PROBLEM end log file")
		elif(self.h_op_reg != None):
			#Setting non-linear component of the model
			self.h_op.set_nl(self.model)
			self.h_op_reg.set_nl(self.model)
			#Problem is linearly regularized (fixing non-linear part and evaluating the epsilon on the linear component)
			epsilon_balance = self.vp_linear_prob.estimate_epsilon(verbose,logger)
		return epsilon_balance

	def resf(self,model):
		"""Method to return residual vector"""
		#Zero-out residual vector
		self.res.zero()
		###########################################
		#Applying full non-linear modeling operator
		res = self.res
		if(self.epsilon != None): res = self.res.vecs[0]
		#Computing non-linear part g(m) (if any)
		if(self.g_op != None): self.g_op.nl_op.forward(False,model,res)
		#Computing non-linear part g_reg(m) (if any)
		if(self.g_op_reg != None): self.g_op_reg.nl_op.forward(False,model,self.res.vecs[1])

		##################################
		#Setting data for linear inversion
		# data term = data - [g(m) if any]
		res.scaleAdd(self.data,-1.0,1.0)
		#Setting data within first term
		self.vp_linear_prob.data=res

		# regularization data term = [g_reg(m) - data_reg if any]
		if(self.data_reg != None):
			self.res.vecs[1].scaleAdd(self.data_reg,1.0,-1.0)
		#Data term for linear regularization term
		if("epsilon" in dir(self.vp_linear_prob)):
			self.res.vecs[1].scale(-1.0)
			self.vp_linear_prob.prior_model=self.res.vecs[1]

		##################################
		#Running linear inversion
		#Getting fevals for saving linear inversion results
		fevals = self.get_fevals()
		#Setting initial linear inversion model
		self.lin_model.zero()
		self.vp_linear_prob.set_model(self.lin_model)
		#Setting non-linear component of the model
		self.h_op.set_nl(model)
		if(self.h_op_reg != None):
			self.h_op_reg.set_nl(model)
		#Resetting inversion problem variables
		self.vp_linear_prob.setDefaults()
		#Saving linear inversion results if requested
		if(self.lin_solver_prefix != None):
			self.lin_solver.setPrefix(self.lin_solver_prefix + "_feval%s"%(fevals))

		#Printing non-linear inversion information
		if(self.lin_solver.logger != None):
			#Writing linear inversion log information if requested (i.e., a logger is present in the solver)
			msg  = "NON_LINEAR INVERSION INFO:\n	objective function evaluation\n"
			msg += "#########################################################################################\n"
			self.lin_solver.logger.addToLog(msg+"Linear inversion for non-linear function evaluation # %s"%(fevals))
		self.lin_solver.run(self.vp_linear_prob,verbose=False)
		if(self.lin_solver.logger != None): self.lin_solver.logger.addToLog("#########################################################################################\n")
		#Copying inverted linear optimal model
		self.lin_model.copy(self.vp_linear_prob.get_model())
		#Flushing internal saved results of the linear inversion
		self.lin_solver.flush_results()

		##################################
		#Obtaining the residuals
		if (self.epsilon != None) and not("epsilon" in dir(self.vp_linear_prob)):
			#Regularization contains a non-linear operator only
			self.res.vecs[0].copy(self.vp_linear_prob.get_res(self.lin_model))
			self.res.vecs[1].scale(self.epsilon)
		else:
			self.res.copy(self.vp_linear_prob.get_res(self.lin_model))
		return self.res

	def gradf(self,model,res):
		"""
		   Method to return gradient vector
		   grad= [G(m)' + H(m_nl;m_lin)'] r_d + epsilon * [G'(m_nl)' + H'(m_nl;m_lin)'] r_m
		"""
		#Zero-out gradient vector
		self.grad.zero()
		#Setting the optimal linear model component and background of the Jacobian matrices
		self.h_op.set_lin_jac(self.lin_model) #H(_,m_lin_opt)
		self.h_op.h_nl.set_background(model) #H(m_nl,m_lin_opt)
		if(self.h_op_reg != None):
			self.h_op_reg.set_lin_jac(self.lin_model) #H'(_,m_lin_opt)
			self.h_op_reg.h_nl.set_background(model) #H'(m_nl,m_lin_opt)
		if(self.g_op != None): self.g_op.set_background(model) #G(m_nl)
		if(self.g_op_reg != None): self.g_op_reg.set_background(model) #G'(m_nl)
		#Computing contribuition from the regularization term (if any)
		if(self.epsilon != None):
			# G'(m_nl)' r_m
			if(self.g_op_reg != None): self.g_op_reg.lin_op.adjoint(False,self.grad,res.vecs[1])
			# H'(m_nl,m_lin_opt)' r_m
			if(self.h_op_reg != None): self.h_op_reg.h_nl.lin_op.adjoint(True,self.grad,res.vecs[1])
			# epsilon * [G'(m_nl)' + H'(m_nl,m_lin_opt)'] r_m
			self.grad.scale(self.epsilon)
		res = self.res
		if(self.epsilon != None): res = self.res.vecs[0]
		# G(m_nl)' r_d
		if(self.g_op != None): self.g_op.lin_op.adjoint(True,self.grad,res)
		# H(m_nl,m_lin_opt)' r_d
		self.h_op.h_nl.lin_op.adjoint(True,self.grad,res)
		if(self.lin_solver.logger != None): self.lin_solver.logger.addToLog("NON_LINEAR INVERSION INFO:\n	Gradient has been evaluated, current objective function value: %s;\n 	Stepping!"%(self.get_obj(model)))
		return self.grad

	def dresf(self,model,dmodel):
		"""Method to return residual vector dres (Not currently supported)"""
		raise NotImplementedError("ERROR! dresf is not currently supported! Provide an initial step-length value different than zero.")
		return self.dres

	def objf(self,res):
		"""Method to return objective function value 1/2*|g(m_nl) + h(m_nl)m_lin - d|_2 + epsilon^2/2*|g'(m_nl) + h'(m_nl)m_lin - d'|_2"""
		if("obj_terms" in dir(self)):
			#data term
			val = res.vecs[0].norm()
			self.obj_terms[0]=0.5*val*val
			#model term
			val = res.vecs[1].norm()
			self.obj_terms[1]=0.5*val*val
			obj=self.obj_terms[0]+self.obj_terms[1]
		else:
			val = res.norm()
			obj=0.5*val*val
		return obj
