#Module containing containing Iterative Soft-Thresholding with Cooling solver for linear L1-regularized problems
from math import isnan
import pySolver


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
			prblm_mdl=prblm.get_model()
			istc_mdl = prblm_mdl.clone()

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
			#Retrieving lambda values
			lambda_values = self.restart.retrieve_parameter("lambda_values")
			iter = self.restart.retrieve_parameter("iter")
			initial_obj_value=self.restart.retrieve_parameter("obj_initial") #Retrieving initial objective function value

		#Common variables unrelated to restart
		success = True
		#Outer iteration loop
		while True:
			#Setting lambda value for a given outer loop iteration
			prblm.set_lambda(lambda_values[iter-1])
			msg = "Outer_iter = %s lambda_value = %s"%(iter,lambda_values[iter])
			if(verbose): print(msg)
			if(self.logger): self.logger.addToLog(msg)
			inner_iter = 0
			while(inner_iter < self.inner_it):
				obj=prblm.get_obj(modl) 			#Compute objective function value
				prblm_res=prblm.get_res(modl) 		#Compute residuals res = y - Ax
				prblm_grad=prblm.get_grad(modl) 	#Compute the gradient g = - A' [y - Ax]
				if(inner_iter == 0):
					msg = "	Inner_iter = %s obj = %s residual norm = %s gradient norm= %s feval = %s"%(inner_iter-1,obj0,prblm.get_rnorm(),prblm.get_gnorm(),prblm.get_fevals())
					if(verbose): print info
					#Writing on log file
					if(verbose): print(msg)
					if(self.logger): self.logger.addToLog(msg)
					#Check if either objective function value or gradient norm is NaN
					if(isnan(obj0) or isnan(prblm_grad.norm())): raise ValueError("ERROR! Either gradient norm or objective function value NaN!")
				if(prblm.get_gnorm() == 0.):
					print "Gradient vanishes identically"
					break
				#Outputing files if problem was not restarted
				if(not prblm.restart.restarting):
					prblm.output(modl)
					#Removing preconditioning scaling factor from saved inverted model
					sep.Scale(prblm.inverted_model,prblm.scale_precond)
				prblm.restart.restarting=False

				#Stepping for internal iteration model update
				sep.Cp(modl,modl0) #Saving model before updating it
				sep.Sum(modl,grad,1.0,-1.0) #Update model x = x + A' [y - Ax]
				#########################################
				#SOFT-THRESHOLDING STEP
				modl_arr,modl_axes = sep.sep_read_file(modl)
				modl_arr = np.sign(modl_arr)*np.clip(np.abs(modl_arr)-prblm.lambda_value,0.,None)
				if(sep_write_file(modl,modl_axes,modl_arr)): raise IOError("ERROR! Problem writing soft-thresholded model vector to file")
				#########################################
				#Apply hard bounds if defined
				clipped=self.stpr.clipping(modl,log_file)

				obj=prblm.get_obj(modl)
				obj1=sep.Get_value(obj)
				if(obj1 >= obj0):
					info = "Objective function didn't reduce, will terminate solver: obj_new=%s obj_current=%s"%(obj1,obj0)
					if(verbose): print info
					#Writing on log file
					solv.write_log_file(log_file,info=info)
					#Stepping back to the previous solution
					sep.Cp(modl0,modl)
					break

				#Saving current model in case of restart
				prblm.restart.write_file(modl,"model_restart_ISTC.H")
				#iteration info
				info = "Inner_iter = %s obj = %s residual norm = %s gradient norm= %s feval = %s"%(inner_iter-1,obj0,prblm.get_rnorm(),prblm.get_gnorm(),prblm.get_fevals())
				if(verbose): print info
				#Writing on log file
				solv.write_log_file(log_file,info="\n"+info)
				solv.write_log_file(prblm.restart.log_file,info="\n"+info)
				#Check if either objective function value or gradient norm is NaN
				if(math.isnan(obj1) or math.isnan(prblm.get_gnorm())): raise ValueError("Error! Either gradient norm or objective function value NaN!")
				inner_iter += 1
			iter = iter + 1
			if (self.stoppr.run(prblm,iter,initial_obj_value,verbose)): break

		#Writing last inverted model
		self.save_results(iter,prblm,force_save=True,force_write=True)
		if(self.logger): self.logger.addToLog("ITERATIVE SOFT-THRESHOLDING WITH COOLING SOLVER log file end")
		#Clear restart object
		self.restart.clear_restart()

		return
