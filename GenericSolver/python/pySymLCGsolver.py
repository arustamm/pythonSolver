#Module containing Linear-Conjugate Gradient and Steepest-descent (for symmetric systems) Solver class
#It solves 1/2m'Am -m'b
from math import isnan
import pySolver
from pyProblem import ProblemLinearSymmetric

class SymLCGsolver(pySolver.Solver):
	"""Linear-Conjugate Gradient Solver (for symmetric systems) parent object"""

	#Default class methods/functions
	def __init__(self,stoppr,steepest=False,logger=None):
		"""Constructor for LCG Solver for symmetric systems"""
		#Defining stopper object
		self.stoppr=stoppr
		#Whether to run steepest descent or not
		self.steepest=steepest
		#Logger object to write on log file
		self.logger=logger
		#Overwriting logger of the Stopper object
		self.stoppr.logger=self.logger
		#Setting defaults for saving results
		self.setDefaults()
		return

	def __del__(self):
		"""Default destructor"""
		return

	def run(self,prblm,verbose=True,restart=False):
		"""Running LCG solver for symmetric systems"""
		#Checking if we are solving a linear square problem
		if(not isinstance(prblm,ProblemLinearSymmetric)):
			raise TypeError("ERROR! Provided problem object not a linear symmetric problem")
		if(not restart):
			if(self.steepest):
				msg="LINEAR STEEPEST-DESCENT SOLVER FOR SYMMETRIC MATRIX"
			else:
				msg="LINEAR CONJUGATE GRADIENT SOLVER FOR SYMMETRIC MATRIX"
			if(verbose): print(msg)
			if(self.logger): self.logger.addToLog(msg+" log file")
			#Printing restart folder
			msg="Restart folder: %s\n"%(self.restart.restart_folder)
			if(verbose): print(msg)
			if(self.logger): self.logger.addToLog(msg)

			#Setting internal vectors (model and search direction vectors)
			prblm_mdl=prblm.get_model()
			cg_mdl = prblm_mdl.clone()
			cg_dmodl = prblm_mdl.clone()
			cg_dmodl.zero()

			#Other internal variables
			iter = 0
			beta = 0.0
		else:
			#Retrieving parameters and vectors to restart the solver
			msg="Restarting previous solver run from: %s"%(self.restart.restart_folder)
			if(verbose): print(msg)
			if(self.logger): self.logger.addToLog(msg)
			self.restart.read_restart()
			iter = self.restart.retrieve_parameter("iter")
			beta = self.restart.retrieve_parameter("beta")
			obj_old = self.restart.retrieve_parameter("obj_old")
			cg_mdl    = self.restart.retrieve_vector("cg_mdl")
			cg_dmodl  = self.restart.retrieve_vector("cg_dmodl")
			#Setting the model and residuals to avoid residual double computation
			prblm.set_model(cg_mdl)
			#Setting residual vector to avoid its unnecessary computation
			prblm.set_residual(self.restart.retrieve_vector("prblm_res"))

		#Common variables unrelated to restart
		success = True
		data_norm = prblm.data.norm()

		#Iteration loop
		while True:
			#Computing objective function
			prblm_res=prblm.get_res(cg_mdl) 	#Compute residuals
			obj0 = prblm.get_obj(cg_mdl)    	#Compute objective function value
			if(iter==0):
				msg = "iter = %s obj = %s residual norm = %s feval = %s"%(iter,obj0,prblm_res.norm(),prblm.get_fevals())
				if(verbose): print(msg)
				msg += "\nrelative data matching (i.e., 1-|Am-b|/|b|): %s"%((1.0-prblm.get_rnorm()/data_norm)*100.0)+"%"
				#Writing on log file
				if(self.logger): self.logger.addToLog(msg)
				#Check if either objective function value or gradient norm is NaN
				if(isnan(obj0)): raise ValueError("Error! Objective function value NaN!")

			#Saving results
			self.save_results(iter,prblm,force_save=False)

			#dmodl = beta * dmodl - res
			cg_dmodl.scaleAdd(prblm_res,beta,-1.0) 				#update search direction
			prblm_ddmodl=prblm.get_dres(cg_mdl,cg_dmodl)	#Project gradient in the data space

			dot_dmodl_ddmodl=cg_dmodl.dot(prblm_ddmodl)
			dot_res=prblm_res.dot(prblm_res)
			if(dot_res == 0.):
				msg = "Residual/Gradient vector vanishes identically"
				if(verbose): print(msg)
				#Writing on log file
				if(self.logger): self.logger.addToLog(msg)
				break
			elif(dot_dmodl_ddmodl==0.0):
				success = False
				msg="Residual/Gradient vector orthogonal to span of linear operator, will terminate solver"
				#Writing on log file
				if(self.logger): self.logger.addToLog(msg)

			if(not success):
				msg="Stepper couldn't find a proper step size, will terminate solver"
				if(verbose): print(msg)
				if(self.logger): self.logger.addToLog(msg)
				break

			alpha = dot_res/dot_dmodl_ddmodl
			if(self.logger): self.logger.addToLog("Alpha step length: %s"%(alpha))
			#res  = res + alpha * dres =  res + alpha * A * dmodl
			prblm_res.scaleAdd(prblm_ddmodl,sc2=alpha) #update residuals
			#modl = modl + alpha * dmodl
			cg_mdl.scaleAdd(cg_dmodl,sc2=alpha) #Update model

			# clipped=self.stpr.clipping(modl,log_file)

			#Increasing iteration counter
			iter = iter + 1
			#Setting the model and residuals to avoid residual twice computation
			prblm.set_model(cg_mdl)

			#Setting residual vector to avoid its unnecessary computation
			prblm.set_residual(prblm_res)

			#Computing new objective function value
			obj1=prblm.get_obj(cg_mdl)
			#New residual norm
			dot_res_new=prblm_res.dot(prblm_res)
			if(not self.steepest):
				beta = dot_res_new/dot_res
			#Checking monotonic behavior of objective function
			if(iter==1):
				obj_old = obj0 #Saving objective function at iter-1
			else:
				#If not monotonically changing stop the inversion
				if not((obj_old < obj0 < obj1) or (obj_old > obj0 > obj1)):
					msg = "Objective function variation not monotonic, will terminate solver: obj_old=%s obj_cur=%s obj_new=%s"%(obj_old,obj0,obj1)
					if(verbose): print(msg)
					#Writing on log file
					if(self.logger): self.logger.addToLog(msg)
					#Stepping back to the previous solution
					cg_mdl.scaleAdd(cg_dmodl,1.0,-1.0)
					prblm.set_model(cg_mdl)
					break
				obj_old = obj0 #Saving objective function at iter-1

			#Saving current model and previous search direction in case of restart
			self.restart.save_parameter("iter",iter)
			self.restart.save_parameter("beta",beta)
			self.restart.save_parameter("obj_old",obj_old)
			self.restart.save_vector("cg_mdl",cg_mdl)
			self.restart.save_vector("cg_dmodl",cg_dmodl)
			#Saving data space vectors
			self.restart.save_vector("prblm_res",prblm_res)

			#iteration info
			msg = "iter = %s obj = %s residual norm = %s feval = %s"%(iter,obj1,prblm_res.norm(),prblm.get_fevals())
			if(verbose): print(msg)
			msg += "\nrelative data matching (i.e., 1-|Am-b|/|b|): %s"%((1.0-prblm.get_rnorm()/data_norm)*100.0)+"%"
			#Writing on log file
			if(self.logger): self.logger.addToLog(msg)
			#Check if either objective function value or gradient norm is NaN
			if(isnan(obj1)): raise ValueError("Error! Objective function value NaN!")
			if (self.stoppr.run(prblm,iter,verbose=verbose)): break

		#Writing last inverted model
		self.save_results(iter,prblm,force_save=True,force_write=True)
		if(self.steepest):
			if(self.logger): self.logger.addToLog("LINEAR STEEPEST-DESCENT SOLVER FOR SYMMETRIC MATRIX log file end")
		else:
			if(self.logger): self.logger.addToLog("LINEAR CONJUGATE GRADIENT SOLVER FOR SYMMETRIC MATRIX log file end")
		#Clear restart object
		self.restart.clear_restart()

		return
