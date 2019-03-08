#Module containing containing Iterative Shrikage-Thresholding Algorithm (ISTA) for linear L1-regularized problems
from math import isnan
import numpy as np
import pySolver
from pyProblem import ProblemL1Lasso

def soft_thresh(x, l):
	"""
	   Soft-thresholding function:
	   x    = [no default] - numpy array; input values
	   l    = [no default] - float; soft threshold
	   return:
	   y    = - numpy array; output clipped values
	"""
	return np.sign(x) * np.maximum(np.abs(x) - l, 0.)

class ISTAsolver(pySolver.Solver):
	"""Iterative Shrikage-Thresholding Algorithm (ISTA) solver to solve: convex problem 1/2*| y - Am |_2 + lambda*| m |_1"""

	def __init__(self,stoppr,fast=False,logger=None):
		"""
		   Constructor for ISTA Solver:
		   stoppr   = [no default] - stopper class; Stopper object to terminate inversion
		   fast    	= [False] - boolean; Flag if the apply the fast-ISTA (FISTA)
		   logger   = [None] - logger class; Logger object to write inversion log file
		"""
		#Calling parent construction
		super(ISTAsolver,self).__init__()
		#Defining stopper object
		self.stoppr=stoppr
		#Logger object to write on log file
		self.logger=logger
		#Overwriting logger of the Stopper object
		self.stoppr.logger=self.logger
		#Setting the fast flag
		self.fast = fast
		return

	def __del__(self):
		"""Default destructor"""
		return


	def run(self,prblm,verbose=False,restart=False):
		"""Running ISTA solver"""
		#Checking if the provided problem is L1-LASSO
		if(not isinstance(prblm,ProblemL1Lasso)):
			raise TypeError("ERROR! Provided inverse problem not ProblemL1Lasso!")
		#Checking if the regularization weight was set
		if(prblm.lambda_value == None):
			raise ValueError("ERROR! Regularization weight (lambda_value) is not set!")
		if(not restart):
			if(self.fast):
				msg="FAST ITERATIVE SHRINKAGE-THRESHOLDING ALGORITHM log file\n"
			else:
				msg="ITERATIVE SHRINKAGE-THRESHOLDING ALGORITHM log file\n"
			#Printing restart folder
			msg+="Restart folder: %s\n"%(self.restart.restart_folder)
			msg+="Regularization weight: %s\n"%(prblm.lambda_value)
			if(verbose): print(msg.replace("log file",""))
			if(self.logger): self.logger.addToLog(msg)

			#Setting internal vectors (model, search direction, and previous gradient vectors)
			prblm_mdl = prblm.get_model()
			ista_mdl  = prblm_mdl.clone()
			#Other parameters in case FISTA is requested
			if(self.fast):
				t = 1.0
				fista_mdl = prblm_mdl.clone()

			#Other internal variables
			iter = 0

		else:
			#Retrieving parameters and vectors to restart the solver
			msg="Restarting previous solver run from: %s"%(self.restart.restart_folder)
			if(verbose): print(msg)
			if(self.logger): self.logger.addToLog(msg)
			self.restart.read_restart()
			#Retrieving inversion parameters
			iter = self.restart.retrieve_parameter("iter")
			initial_obj_value=self.restart.retrieve_parameter("obj_initial") #Retrieving initial objective function value
			ista_mdl = self.restart.retrieve_vector("ista_mdl")
			#Other parameters in case FISTA is requested
			if(self.fast):
				t = self.restart.retrieve_parameter("t")
				fista_mdl = self.restart.retrieve_vector("fista_mdl")

		#Common variables unrelated to restart
		success = True
		ista_mdl0 = ista_mdl.clone() #Previous model in case stepping procedure fails

		#Inversion loop
		while True:
			obj0=prblm.get_obj(ista_mdl) 		#Compute objective function value
			prblm_grad=prblm.get_grad(ista_mdl) #Compute the gradient g = - A' [y - Ax]
			if(iter == 0):
				#Saving initial objective function value
				initial_obj_value = obj0
				self.restart.save_parameter("obj_initial",initial_obj_value)
				msg = "iter = %s obj = %s residual norm = %s gradient norm= %s feval = %s"%(iter,obj0,prblm.get_rnorm(),prblm.get_gnorm(),prblm.get_fevals())
				#Writing on log file
				if(verbose): print(msg)
				if(self.logger): self.logger.addToLog(msg)
				#Check if either objective function value or gradient norm is NaN
				if(isnan(obj0) or isnan(prblm_grad.norm())): raise ValueError("ERROR! Either gradient norm or objective function value NaN!")
			if(prblm.get_gnorm() == 0.):
				print("Gradient vanishes identically")
				break

			#Saving results
			self.save_results(iter,prblm,force_save=False)

			ista_mdl0.copy(ista_mdl) #Saving model before updating it
			if(self.fast):
				#Running FISTA
				fista_mdl.scaleAdd(prblm_grad,1.0,-1.0/prblm.op_norm)
				#########################################
				#SOFT-THRESHOLDING STEP
				ista_mdl.copy(fista_mdl)
				modl_arr = ista_mdl.getNdArray()
				modl_arr[:] = soft_thresh(modl_arr,prblm.lambda_value/prblm.op_norm)
				#########################################
				#Projecting model onto the bounds (if any)
				if("bounds" in dir(prblm)): prblm.bounds.apply(ista_mdl)
				t0 = t
				t = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
				#z = x
				fista_mdl.copy(ista_mdl)
				#z = x + ((t0 - 1.) / t) * (x - xold)
				scale = (t0 - 1.0) / t
				fista_mdl.scaleAdd(ista_mdl0,1.0+scale,-scale)
			else:
				#Running ISTA
				ista_mdl.scaleAdd(prblm_grad,1.0,-1.0/prblm.op_norm) #Update model x = x + scale_precond * A' [y - Ax]
				#########################################
				#SOFT-THRESHOLDING STEP
				modl_arr = ista_mdl.getNdArray()
				modl_arr[:] = soft_thresh(modl_arr,prblm.lambda_value/prblm.op_norm)
				#########################################
				#Projecting model onto the bounds (if any)
				if("bounds" in dir(prblm)): prblm.bounds.apply(ista_mdl)

			obj1=prblm.get_obj(ista_mdl)
			if(obj1 >= obj0):
				msg = "Objective function didn't reduce, will terminate solver: obj_new=%s obj_current=%s"%(obj1,obj0)
				if(verbose): print(msg)
				#Writing on log file
				if(self.logger): self.logger.addToLog(msg)
				#Copying back to the previous solution
				ista_mdl.copy(ista_mdl0)
				break

			#Saving current model in case of restart and other parameters
			self.restart.save_parameter("iter",iter)
			self.restart.save_vector("ista_mdl",ista_mdl)
			if(self.fast):
				self.restart.save_parameter("t",t)
				self.restart.save_vector("fista_mdl",fista_mdl)

			#iteration info
			iter = iter + 1
			msg = "iter = %s obj = %s residual norm = %s gradient norm= %s feval = %s"%(iter,obj1,prblm.get_rnorm(),prblm_grad.norm(),prblm.get_fevals())
			if(verbose): print(msg)
			#Writing on log file
			if(self.logger): self.logger.addToLog("\n"+msg)
			#Check if either objective function value or gradient norm is NaN
			if(isnan(obj1) or isnan(prblm_grad.norm())): raise ValueError("ERROR! Either gradient norm or objective function value NaN!")
			if (self.stoppr.run(prblm,iter,initial_obj_value,verbose)): break

		#Writing last inverted model
		self.save_results(iter,prblm,force_save=True,force_write=True)
		if(self.fast):
			if(self.logger): self.logger.addToLog("FAST ITERATIVE SHRINKAGE-THRESHOLDING ALGORITHM log file end")
		else:
			if(self.logger): self.logger.addToLog("ITERATIVE SHRINKAGE-THRESHOLDING ALGORITHM log file end")
		#Clear restart object
		self.restart.clear_restart()

		return
