#Module containing the definition of an abstract inverse problem (phi(m) = |f(m)-d|_2)
import pyVector as Vec
import pyOperator as pyOp
from math import isnan

class Problem:
	"""Problem parent object"""

	#Default class methods/functions
	def __init__(self):
		"""Default class constructor for Problem"""
		return

	def __del__(self):
		"""Default destructor"""
		return

	def setDefaults(self):
		"""Default common variables for any inverse problem"""
		self.linear=False #By default all problem are non-linear
		self.obj_updated=False
		self.res_updated=False
		self.grad_updated=False
		self.dres_updated=False
		self.fevals=0
		self.counter=0
		return

	def set_model(self,model):
		"""Setting internal model vector"""
		if(model.isDifferent(self.model)):
			self.model.copy(model)
			self.obj_updated  = False
			self.res_updated  = False
			self.grad_updated = False
			self.dres_updated = False
		return

	def get_model(self):
		"""Accessor for model vector"""
		return self.model

	def get_dmodel(self):
		"""Accessor for model vector"""
		return self.dmodel

	def get_rnorm(self):
		"""Accessor for residual vector norm"""
		return self.res.norm()

	def get_gnorm(self):
		"""Accessor for gradient vector norm"""
		return self.grad.norm()

	def get_obj(self,model):
		"""Accessor for objective function"""
		self.set_model(model)
		if(not self.obj_updated):
			self.res = self.get_res(self.model)
			self.obj = self.objf(self.res)
			self.obj_updated=True
		return self.obj

	def get_res(self,model):
		"""Accessor for residual vector"""
		self.set_model(model)
		if(not self.res_updated):
			self.res = self.resf(self.model)
			self.fevals += 1
			self.res_updated=True
		return self.res

	def get_grad(self,model):
		"""Accessor for gradient vector"""
		self.set_model(model)
		if not self.grad_updated:
			self.res  = self.get_res(self.model)
			self.grad = self.gradf(self.model,self.res)
			self.fevals += 1
			if(not self.linear): self.fevals += 1 #Non-linear problem Jacobian assumed to be twice the computational cost of f(m)
			self.grad_updated=True
		return self.grad

	def get_dres(self,model,dmodel):
		"""Accessor for dresidual vector (i.e., application of the Jacobian to Dmodel vector)"""
		self.set_model(model)
		if(not self.dres_updated  or dmodel.isDifferent(self.dmodel)):
			self.dmodel.copy(dmodel)
			self.dres = self.dresf(self.model,self.dmodel)
			self.fevals += 1
			if(not self.linear): self.fevals += 1 #Non-linear problem Jacobian assumed to be twice the computational cost of f(m)
			self.dres_updated=True
		return self.dres

	def get_fevals(self):
		"""Accessor for number of objective function evalutions"""
		return self.fevals

	def objf(self,res):
		"""Dummy objf running method, must be overridden in the derived class"""
		raise NotImplementedError("Implement objf for problem in the derived class!")
		return

	def resf(self,model):
		"""Dummy resf running method, must be overridden in the derived class"""
		raise NotImplementedError("Implement resf for problem in the derived class!")
		return

	def dresf(self,model,dmodel):
		"""Dummy dresf running method, must be overridden in the derived class"""
		raise NotImplementedError("Implement dresf for problem in the derived class!")
		return

	def gradf(self,model,residual):
		"""Dummy gradf running method, must be overridden in the derived class"""
		raise NotImplementedError("Implement gradf for problem in the derived class!")
		return

class ProblemL2Linear(Problem):
	"""Linear inverse problem of the form 1/2*|Lm-d|_2"""

	def __init__(self,model,data,op):
		"""Constructor of linear problem"""
		#Setting internal vector
		self.model=model.clone()
		self.dmodel=model.clone()
		self.dmodel.zero()
		#Gradient vector
		self.grad=self.dmodel.clone()
		#Copying the pointer to data vector
		self.data=data
		#Residual vector
		self.res=data.clone()
		self.res.zero()
		#Dresidual vector
		self.dres=self.res.clone()
		#Setting linear operator
		self.op=op
		#Setting default variables
		self.setDefaults()
		self.linear=True
		return

	def __del__(self):
		"""Default destructor"""
		return

	def set_residual(self,residual):
		"""Setting internal residual vector"""
		#Useful for linear inversion (to avoid residual computation)
		if(self.res.isDifferent(residual)): self.res.copy(residual)
		self.res_updated=True
		return

	def resf(self,model):
		"""Method to return residual vector r = Lm - d"""
		#Computing Lm
		if(model.norm(2)!=0.0):
			self.op.forward(False,model,self.res)
		else:
			self.res.zero()
		#Computing Lm - d
		self.res.scaleAdd(self.data,1.,-1.)
		return self.res

	def gradf(self,model,res):
		"""Method to return gradient vector g = L'r = L'(Lm - d)"""
		#Computing L'r = g
		self.op.adjoint(False,self.grad,res)
		return self.grad

	def dresf(self,model,dmodel):
		"""Method to return residual vector dres = Ldm"""
		#Computing Ldm = dres
		self.op.forward(False,dmodel,self.dres)
		return self.dres

	def objf(self,res):
		"""Method to return objective function value 1/2|Lm-d|_2"""
		obj=0.5*res.dot(res)
		return obj


class ProblemLinearSymmetric(Problem):
	"""Linear inverse problem of the form 1/2m'Am - m'b"""

	def __init__(self,model,data,op):
		"""Constructor of linear problem"""
		#Checking range and domain are the same
		if(not model.checkSame(data)):
			raise ValueError("ERROR! Data and model vector live in different spaces!")
		#Setting internal vector
		self.model=model.clone()
		self.dmodel=model.clone()
		self.dmodel.zero()
		#Copying the pointer to data vector
		self.data=data
		#Residual vector
		self.res=data.clone()
		self.res.zero()
		#Gradient vector is equal to the residual vector
		self.grad=self.res
		#Dresidual vector
		self.dres=self.res.clone()
		#Setting linear operator
		self.op=op
		#Setting default variables
		self.setDefaults()
		self.linear=True
		return

	def __del__(self):
		"""Default destructor"""
		return

	def set_residual(self,residual):
		"""Setting internal residual vector"""
		#Useful for linear inversion (to avoid residual computation)
		if(self.res.isDifferent(residual)): self.res.copy(residual)
		self.res_updated=True
		return

	def resf(self,model):
		"""Method to return residual vector r = Am - b"""
		#Computing Lm
		if(model.norm(2)!=0.0):
			self.op.forward(False,model,self.res)
		else:
			self.res.zero()
		#Computing Lm - d
		self.res.scaleAdd(self.data,1.,-1.)
		return self.res

	def gradf(self,model,res):
		"""Method to return gradient vector equal to residual one"""
		#Assigning g = r
		self.grad=self.res
		return self.grad

	def dresf(self,model,dmodel):
		"""Method to return residual vector dres = Adm"""
		#Computing Ldm = dres
		self.op.forward(False,dmodel,self.dres)
		return self.dres

	def objf(self,res):
		"""Method to return objective function value 1/2m'Am - m'b"""
		obj=0.5*(self.model.dot(res)-self.model.dot(self.data))
		return obj

class ProblemL2LinearReg(Problem):
	"""Linear inverse problem regularized of the form 1/2*|Lm-d|_2 + epsilon^2/2*|Am|_2"""

	def __init__(self,model,data,op,epsilon,reg_op=None):
		"""Constructor of linear problem"""
		#Setting internal vector
		self.model=model.clone()
		self.dmodel=model.clone()
		self.dmodel.zero()
		#Gradient vector
		self.grad=self.dmodel.clone()
		#Copying the pointer to data vector
		self.data=data
		#Setting linear operators
		#Assuming identity operator if regularization operator was not provided
		if(reg_op == None): reg_op = pyOp.IdentityOp(self.model)
		self.op=pyOp.stackOperator(op,reg_op,model,Vec.superVector(self.data,reg_op.range)) #Modeling operator
		self.epsilon=epsilon #Regularization weight
		#Residual vector (data and model residual vectors)
		self.res=self.op.range.clone()
		self.res.zero()
		#Dresidual vector
		self.dres=self.res.clone()
		#Setting default variables
		self.setDefaults()
		self.linear=True
		return

	def __del__(self):
		"""Default destructor"""
		return

	def estimate_epsilon(self,verbose=True,logger=None):
		"""Method returning epsilon that balances the first gradient in the 'extended-data' space"""
		msg="Epsilon Scale evaluation"
		if(verbose): print(msg)
		if(logger): self.logger.addToLog("REGULARIZED PROBLEM log file\n"+msg)
		#Keeping the initial model vector
		prblm_mdl = self.get_model()
		mdl_tmp = prblm_mdl.clone()
		#Keeping user-predefined epsilon if any
		epsilon = self.epsilon
		#Setting epsilon to one to evaluate the scale
		self.epsilon=1.0
		prblm_grad = self.get_grad(self.model)  #Compute first gradient
		prblm_res = self.get_res(prblm_grad)	#Compute residual arising from the gradient
		#Balancing the first gradient in the 'extended-data' space
		res_data_norm=prblm_res.vec1.norm()
		res_model_norm=prblm_res.vec2.norm()
		if (isnan(res_model_norm) or isnan(res_data_norm)):
			raise ValueError("ERROR! Obtained NaN: Residual-data-side-norm = %s, Residual-model-side-norm = %s"%(res_data_norm,res_model_norm))
		if(res_model_norm==0.0):
			raise ValueError("Model residual component norm is zero, cannot find epsilon scale")
		#Resetting user-predefined epsilon if any
		self.epsilon = epsilon
		#Resetting problem initial model vector
		self.set_model(mdl_tmp)
		del mdl_tmp
		epsilon_balance = res_data_norm/res_model_norm
		#Resetting feval
		self.fevals = 0
		msg = "	Epsilon balancing the data-space gradients is: %s"%(epsilon_balance)
		if(verbose): print(msg)
		if(logger): self.logger.addToLog(msg+"\nREGULARIZED PROBLEM end log file")
		return epsilon_balance

	def set_residual(self,residual):
		"""Setting internal residual vector"""
		#Useful for linear inversion (to avoid residual computation)
		if(self.res.isDifferent(residual)): self.res.copy(residual)
		self.res_updated=True
		return

	def resf(self,model):
		"""Method to return residual vector r = [r_d; r_m]: r_d = Lm - d; r_m = Am """
		if(model.norm(2)!=0.0):
			self.op.forward(False,model,self.res)
		else:
			self.res.zero()
		#Computing r_d = Lm - d
		self.res.vec1.scaleAdd(self.data,1.,-1.)
		#Scaling by epsilon epsilon*r_m
		self.res.vec2.scale(self.epsilon)
		return self.res

	def gradf(self,model,res):
		"""Method to return gradient vector g = L'r_d + epsilon*A'r_m"""
		#Scaling by epsilon the model residual vector
		res.vec2.scale(self.epsilon)
		#g = L'r_d + A'(epsilon*r_m)
		self.op.adjoint(False,self.grad,res)
		return self.grad

	def dresf(self,model,dmodel):
		"""Method to return residual vector dres = Ldm"""
		#Computing Ldm = dres_d
		self.op.forward(False,dmodel,self.dres)
		#Scaling by epsilon
		self.dres.vec2.scale(self.epsilon)
		return self.dres

	def objf(self,res):
		"""Method to return objective function value 1/2|Lm-d|_2 + epsilon^2/2*|Am|_2"""
		obj=0.5*res.dot(res)
		return obj



class ProblemL2NonLinear(Problem):
	"""Non-linear inverse problem of the form 1/2*|f(m)-d|_2"""

	def __init__(self,model,data,op):
		"""Constructor of linear problem"""
		#Setting internal vector
		self.model=model.clone()
		self.dmodel=model.clone()
		self.dmodel.zero()
		#Gradient vector
		self.grad=self.dmodel.clone()
		#Copying the pointer to data vector
		self.data=data
		#Residual vector
		self.res=data.clone()
		self.res.zero()
		#Dresidual vector
		self.dres=self.res.clone()
		#Setting non-linear and linearized operators
		self.op=op
		#Setting default variables
		self.setDefaults()
		self.linear=True
		return

	def __del__(self):
		"""Default destructor"""
		return

	def set_residual(self,residual):
		"""Setting internal residual vector"""
		#Useful for linear inversion (to avoid residual computation)
		if(self.res.isDifferent(residual)): self.res.copy(residual)
		self.res_updated=True
		return

	def resf(self,model):
		"""Method to return residual vector r = f(m) - d"""
		#Computing Lm
		if(model.norm(2)!=0.0):
			self.op.nl_op.forward(False,model,self.res)
		else:
			self.res.zero()
		#Computing f(m) - d
		self.res.scaleAdd(self.data,1.,-1.)
		return self.res

	def gradf(self,model,res):
		"""Method to return gradient vector g = F'r = F'(f(m) - d)"""
		#Setting model point on which the F is evaluated
		self.op.lin_op.set_background(model)
		#Computing F'r = g
		self.op.lin_op.adjoint(False,self.grad,res)
		return self.grad

	def dresf(self,model,dmodel):
		"""Method to return residual vector dres = Fdm"""
		#Setting model point on which the F is evaluated
		self.op.lin_op.set_background(model)
		#Computing Fdm = dres
		self.op.lin_op.forward(False,dmodel,self.dres)
		return self.dres

	def objf(self,res):
		"""Method to return objective function value 1/2|f(m)-d|_2"""
		obj=0.5*res.dot(res)
		return obj
