#Module containing containing Iterative Soft-Thresholding with Cooling solver for linear L1-regularized problems
from math import isnan
import numpy as np
import pySolver

def soft_thresh(x, l):
	"""
	   Soft-thresholding function:
	   x    = [no default] - numpy array; input values
	   l    = [no default] - float; soft threshold
	   return:
	   y    = - numpy array; output clipped values
	"""
	return np.sign(x) * np.maximum(np.abs(x) - l, 0.)

class ISTCsolver(pySolver.Solver):
	"""ISTC solver to solve: linear problem 1/2*| y - Am |_2 + lambda*| m |_1"""

	def __init__(self,stoppr,inner_it,cooling_start,cooling_end,steepest=False,logger=None):
		"""
		   Constructor for ISTC Solver
		"""
		#Defining stopper object
		self.stoppr=stoppr
		#Logger object to write on log file
		self.logger=logger
		#Overwriting logger of the Stopper object
		self.stoppr.logger=self.logger
		#Setting defaults for saving results
		self.setDefaults()

		#ISTC parameters
		self.inner_it=inner_it #number of inner iterations, the outer iterations are taken care by the stopper
		# cooling_start and cooling_end are numbers between 0 and 1 such that cooling_start <= cooling_end
		if(not (0<=cooling_start<=1) or not (0<=cooling_end<=1) or (cooling_end<cooling_start)): raise ValueError("ERROR! cooling_start and cooling_end must be within [0,1] interval and cooling_start <= cooling_end")
		self.cooling_start=cooling_start # start of cooling continuation as fraction of size of sorted array |A'y|
		self.cooling_end=cooling_end # end of cooling continuation as fraction of size of sorted array |A'y|
		return

	def __del__(self):
		"""Default destructor"""
		return


	def run(self,prblm,verbose=False,restart=False):
		"""Running ISTC solver"""
		if(not restart):
			msg="ITERATIVE SOFT-THRESHOLDING WITH COOLING SOLVER log file\n"
			#Printing restart folder
			msg+="Restart folder: %s\n"%(self.restart.restart_folder)
			if(verbose): print(msg.replace("log file",""))
			if(self.logger): self.logger.addToLog(msg)

			#Setting internal vectors (model, search direction, and previous gradient vectors)
			prblm_mdl = prblm.get_model()
			istc_mdl  = prblm_mdl.clone()

			#Inversion always starts from m = 0 (I need to understand if it is possible to start from m different than 0)
			istc_mdl.zero()	# modl = 0
			#Other internal variables
			iter = 0
			#Computing cooling parameters
			prblm_grad=prblm.get_grad(istc_mdl)
			grad_arr = prblm_grad.getNdArray()
			grad_arr = np.abs(grad_arr.flatten())					# |A'y| and removing zero elements
			grad_arr = modl_arr[np.nonzero(grad_arr)]
			if(grad_arr.size == 0):
				raise ValueError("ERROR! -- A'y is returning a null vector (i.e., y in the Null space of A')")
			#Sorting the elements in descending order
			grad_arr.sort()
			grad_arr = np.flip(grad_arr,0)
			#Setting fraction of points sampled by the outer loop (linear sampling)
			samples=np.round(np.linspace(cooling_start,cooling_end,outer_it)*modl_arr.size)
			#Lambda values to be used during inversion for each outer loop iteration
			lambda_values=grad_arr[samples]
			#Saving the lambda values to avoid recomputation if restart is used
			self.restart.save_parameter("lambda_values",lambda_values)
		else:
			#Retrieving parameters and vectors to restart the solver
			msg="Restarting previous solver run from: %s"%(self.restart.restart_folder)
			if(verbose): print(msg)
			if(self.logger): self.logger.addToLog(msg)
			self.restart.read_restart()
			#Retrieving lambda values and other parameters
			lambda_values = self.restart.retrieve_parameter("lambda_values")
			iter = self.restart.retrieve_parameter("iter")
			initial_obj_value=self.restart.retrieve_parameter("obj_initial") #Retrieving initial objective function value
			istc_mdl = self.restart.retrieve_vector("istc_mdl")

		#Common variables unrelated to restart
		success = True
		istc_mdl0 = istc_mdl.clone() #Previous model in case stepping procedure fails
		istc_mdl_save = istc_mdl0 	 #used also to save results
		scale_precond = 0.99 * np.sqrt(2) / np.sqrt(prblm.op_norm); #scaling factor applied to operator A for preconditioning

		#Outer iteration loop
		while True:
			#Setting lambda value for a given outer loop iteration
			prblm.set_lambda(lambda_values[iter])
			msg = "Outer_iter = %s lambda_value = %s"%(iter,lambda_values[iter])
			if(verbose): print(msg)
			if(self.logger): self.logger.addToLog(msg)
			if(not restart):
				inner_iter = 0
			else:
				inner_iter = self.restart.retrieve_parameter("inner_iter",inner_iter)
				restart = False
			if(iter == 0):
				#Applying preconditioning
				istc_mdl.scale(scale_precond)
				obj=prblm.get_obj(istc_mdl) 		#Compute objective function value
				#Saving initial objective function value
				initial_obj_value = obj
				self.restart.save_parameter("obj_initial",initial_obj_value)
			while(inner_iter < self.inner_it):
				obj=prblm.get_obj(istc_mdl) 		#Compute objective function value
				prblm_grad=prblm.get_grad(istc_mdl) #Compute the gradient g = - A' [y - Ax]
				if(inner_iter == 0):
					msg = "	Inner_iter = %s obj = %s residual norm = %s gradient norm= %s feval = %s"%(inner_iter,obj0,prblm.get_rnorm(),prblm.get_gnorm(),prblm.get_fevals())
					#Writing on log file
					if(verbose): print(msg)
					if(self.logger): self.logger.addToLog(msg)
					#Check if either objective function value or gradient norm is NaN
					if(isnan(obj0) or isnan(prblm_grad.norm())): raise ValueError("ERROR! Either gradient norm or objective function value NaN!")
				if(prblm.get_gnorm() == 0.):
					print("Gradient vanishes identically")
					break

				#Removing preconditioning scaling factor from inverted model
				istc_mdl_save.copy(istc_mdl)
				istc_mdl_save.scale(scale_precond)
				#Saving results
				self.save_results(iter,prblm,istc_mdl_save,force_save=False)

				#Stepping for internal iteration model update
				istc_mdl0.copy(istc_mdl) #Saving model before updating it
				istc_mdl.scaleAdd(prblm_grad,1.0,-scale_precond) #Update model x = x + scale_precond * A' [y - Ax]
				#########################################
				#SOFT-THRESHOLDING STEP
				modl_arr = istc_mdl.getNdArray()
				modl_arr[:] = soft_thresh(modl_arr,prblm.lambda_value)
				#########################################
				#Projecting model onto the bounds (if any)
				if("bounds" in dir(prblm)): prblm.bounds.apply(istc_mdl)

				istc_mdl.scale(scale_precond)
				obj1=prblm.get_obj(istc_mdl)
				if(obj1 >= obj0):
					msg = "Objective function didn't reduce, will terminate solver: obj_new=%s obj_current=%s"%(obj1,obj0)
					if(verbose): print(msg)
					#Writing on log file
					if(self.logger): self.logger.addToLog(msg)
					#Copying back to the previous solution
					istc_mdl.set_model(istc_mdl0)
					break

				#Saving current model in case of restart and other parameters
				self.restart.save_parameter("iter",iter)
				self.restart.save_parameter("inner_iter",inner_iter)
				self.restart.save_vector("istc_mdl",istc_mdl)

				#iteration info
				msg = "Inner_iter = %s obj = %s residual norm = %s gradient norm= %s feval = %s"%(inner_iter,obj1,prblm_res.norm(),prblm_grad.norm(),prblm.get_fevals())
				if(verbose): print(msg)
				#Writing on log file
				if(self.logger): self.logger.addToLog("\n"+msg)
				#Check if either objective function value or gradient norm is NaN
				if(isnan(obj1) or isnan(prblm_grad.norm())): raise ValueError("ERROR! Either gradient norm or objective function value NaN!")
				inner_iter += 1
			iter = iter + 1
			if (self.stoppr.run(prblm,iter,initial_obj_value,verbose)): break

		#Removing preconditioning scaling factor from inverted model
		istc_mdl_save.copy(istc_mdl)
		istc_mdl_save.scale(scale_precond)
		#Writing last inverted model
		self.save_results(iter,prblm,istc_mdl_save,force_save=True,force_write=True)
		if(self.logger): self.logger.addToLog("ITERATIVE SOFT-THRESHOLDING WITH COOLING SOLVER log file end")
		#Clear restart object
		self.restart.clear_restart()

		return
