#Module containing Linear-Conjugate Gradient Solver class
#It can also handle steppest-descent algorithm
from math import isnan
import pySolver

class LCGsolver(pySolver.Solver):
	"""Linear-Conjugate Gradient and Steepest-Descent Solver parent object"""

	#Default class methods/functions
	def __init__(self,stoppr,steepest=False,logger=None):
		"""
		   Constructor for LCG Solver
		"""
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
		"""Running LCG and steppest-descent solver"""
		if(not restart):
			if(self.steepest):
				msg="LINEAR STEEPEST-DESCENT SOLVER"
			else:
				msg="LINEAR CONJUGATE GRADIENT SOLVER"
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
		else:
			#Retrieving parameters and vectors to restart the solver
			msg="Restarting previous solver run from: %s"%(self.restart.restart_folder)
			if(verbose): print(msg)
			if(self.logger): self.logger.addToLog(msg)
			self.restart.read_restart()
			iter = self.restart.retrieve_parameter("iter")
			initial_obj_value=self.restart.retrieve_parameter("obj_initial") #Retrieving initial objective function value
			cg_mdl = self.restart.retrieve_vector("cg_mdl")
			cg_dmodl = self.restart.retrieve_vector("cg_dmodl")
			cg_dres = self.restart.retrieve_vector("cg_dres")
			#Setting the model and residuals to avoid residual twice computation
			prblm.set_model(cg_mdl)
			#Setting residual vector to avoid its unnecessary computation
			prblm.set_residual(self.restart.retrieve_vector("prblm_res"))

		#Common variables unrelated to restart
		success = True

		#Iteration loop
		while True:
			#Computing objective function
			obj0 = prblm.get_obj(cg_mdl)    	#Compute objective function value
			prblm_res=prblm.get_res(cg_mdl) 	#Compute residuals
			prblm_grad=prblm.get_grad(cg_mdl) 	#Compute the gradient
			if(iter==0):
				initial_obj_value=obj0 			#For relative objective function value
				#Saving objective function value
				self.restart.save_parameter("obj_initial",initial_obj_value)
				msg = "iter = %s obj = %s residual norm = %s gradient norm= %s feval = %s"%(iter,obj0,prblm_res.norm(),prblm_grad.norm(),prblm.get_fevals())
				if(verbose): print(msg)
				#Writing on log file
				if(self.logger): self.logger.addToLog(msg)
				#Check if either objective function value or gradient norm is NaN
				if(isnan(obj0) or isnan(prblm_grad.norm())): raise ValueError("Error! Either gradient norm or objective function value NaN!")
				#Set internal delta residual vector
				cg_dres = prblm_res.clone()
				cg_dres.zero()
			if(prblm_grad.norm() == 0.):
				print("Gradient vanishes identically")
				break
			prblm_gradd=prblm.get_dres(cg_mdl,prblm_grad)	#Project gradient in the data space

			#Saving results
			self.save_results(iter,prblm,force_save=False)

			if(iter==0 or self.steepest):
				#Steepest descent
				beta = 0.0
				dot_gradd=prblm_gradd.dot(prblm_gradd)
				if(dot_gradd==0.0):
					success = False
					msg="Gradient orthogonal to span of linear operator, will terminate solver"
					#Writing on log file
					if(self.logger): self.logger.addToLog(msg)
				else:
					dot_gradd_res=prblm_gradd.dot(prblm_res)
					alpha = - dot_gradd_res/dot_gradd
					msg="Steppest-descent step length: %s"%(alpha)
					#Writing on log file
					if(iter==1):
						msg="First steppest-descent step length: %s"%(alpha)
					if(self.logger): self.logger.addToLog(msg)
			else:
				#Conjugate-gradient coefficients
				dot_gradd=prblm_gradd.dot(prblm_gradd)
				dot_dres=cg_dres.dot(cg_dres)
				dot_gradd_dres=prblm_gradd.dot(cg_dres)
				if((dot_gradd == 0.) or (dot_dres == 0.)):
					success = False
				else:
					determ = dot_gradd * dot_dres - dot_gradd_dres * dot_gradd_dres
					dot_gradd_res=prblm_gradd.dot(prblm_res)
					dot_dres_res=cg_dres.dot(prblm_res)
					alpha = -(dot_dres*dot_gradd_res - dot_gradd_dres*dot_dres_res) /determ
					beta = (dot_gradd_dres*dot_gradd_res - dot_gradd*dot_dres_res) /determ
					#Writing on log file
					if(self.logger): self.logger.addToLog("Conjugate alpha,beta: %s,%s"%(alpha,beta))


			if(not success):
				msg="Stepper couldn't find a proper step size, will terminate solver"
				if(verbose): print(msg)
				if(self.logger): self.logger.addToLog(msg)
				break

			#dmodl = alpha * grad + beta * dmodl
			cg_dmodl.scaleAdd(prblm_grad,beta,alpha) 			#update search direction
			#dres  = alpha * gradd + beta * dres
			cg_dres.scaleAdd(prblm_gradd,beta,alpha)			#update residual step
			#modl = modl + dmodl
			cg_mdl.scaleAdd(cg_dmodl)							#Update model
			#res = res + dres
			prblm_res.scaleAdd(cg_dres)							#Update residuals

			# clipped=self.stpr.clipping(modl,log_file)

			#Increasing iteration counter
			iter = iter + 1
			#Setting the model and residuals to avoid residual twice computation
			prblm.set_model(cg_mdl)

			#Setting residual vector to avoid its unnecessary computation
			prblm.set_residual(prblm_res)

			#Computing new objective function value
			obj1=prblm.get_obj(cg_mdl)
			if(obj1 >= obj0):
				msg = "Objective function didn't reduce, will terminate solver: obj_new=%s obj_current=%s"%(obj1,obj0)
				if(verbose): print(msg)
				#Writing on log file
				if(self.logger): self.logger.addToLog(msg)
				#Stepping back to the previous solution
				cg_mdl.scaleAdd(cg_dmodl,1.0,-1.0)
				prblm.set_model(cg_mdl)
				break

			#Saving current model and previous search direction in case of restart
			self.restart.save_parameter("iter",iter)
			self.restart.save_vector("cg_mdl",cg_mdl)
			self.restart.save_vector("cg_dmodl",cg_dmodl)
			#Saving data space vectors
			self.restart.save_vector("cg_dres",cg_dres)
			self.restart.save_vector("prblm_res",prblm_res)

			#iteration info
			msg = "iter = %s obj = %s residual norm = %s gradient norm= %s feval = %s"%(iter,obj1,prblm_res.norm(),prblm_grad.norm(),prblm.get_fevals())
			if(verbose): print(msg)
			#Writing on log file
			if(self.logger): self.logger.addToLog("\n"+msg)
			#Check if either objective function value or gradient norm is NaN
			if(isnan(obj1) or isnan(prblm_grad.norm())): raise ValueError("Error! Either gradient norm or objective function value NaN!")
			if (self.stoppr.run(prblm,iter,initial_obj_value,verbose)): break

		#Writing last inverted model
		self.save_results(iter,prblm,force_save=True,force_write=True)
		if(self.steepest):
			if(self.logger): self.logger.addToLog("LINEAR STEEPEST-DESCENT SOLVER log file end")
		else:
			if(self.logger): self.logger.addToLog("LINEAR CONJUGATE GRADIENT SOLVER log file end")
		#Clear restart object
		self.restart.clear_restart()

		return
