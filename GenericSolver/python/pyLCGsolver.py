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
		#Calling parent construction
		super(LCGsolver,self).__init__()
		#Defining stopper object
		self.stoppr=stoppr
		#Whether to run steepest descent or not
		self.steepest=steepest
		#Logger object to write on log file
		self.logger=logger
		#Overwriting logger of the Stopper object
		self.stoppr.logger=self.logger
		return

	def __del__(self):
		"""Default destructor"""
		return

	def run(self,prblm,verbose=False,restart=False):
		"""Running LCG and steppest-descent solver"""
		#Resetting stopper before running the inversion
		self.stoppr.reset()
		#Check for preconditioning
		precond = False
		if("prec" in dir(prblm)):
			if(prblm.prec != None): precond = True
		if(not restart):
			msg = ""
			if(precond): msg="PRECONDITIONED "
			if(self.steepest):
				msg+="LINEAR STEEPEST-DESCENT SOLVER"
			else:
				msg+="LINEAR CONJUGATE GRADIENT SOLVER"
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
			if(not precond):
				cg_dres = self.restart.retrieve_vector("cg_dres")
			else:
				dot_grad_prec_grad = self.restart.retrieve_vector("dot_grad_prec_grad")
			#Setting the model and residuals to avoid residual twice computation
			prblm.set_model(cg_mdl)
			prblm_mdl=prblm.get_model()
			#Setting residual vector to avoid its unnecessary computation
			prblm.set_residual(self.restart.retrieve_vector("prblm_res"))

		#Common variables unrelated to restart
		success = True
		if(precond):
			cg_prec_grad = cg_dmodl.clone()
			cg_prec_grad.zero()

		#Iteration loop
		while True:
			#Computing objective function
			obj0 = prblm.get_obj(cg_mdl)    	#Compute objective function value
			prblm_res=prblm.get_res(cg_mdl) 	#Compute residuals
			prblm_grad=prblm.get_grad(cg_mdl) 	#Compute the gradient
			if(iter==0):
				initial_obj_value=obj0 			#For relative objective function value
				#Saving initial objective function value
				self.restart.save_parameter("obj_initial",initial_obj_value)
				msg = "iter = %s obj = %s residual norm = %s gradient norm= %s feval = %s"%(iter,obj0,prblm.get_rnorm(cg_mdl),prblm.get_gnorm(cg_mdl),prblm.get_fevals())
				if(verbose): print(msg)
				#Writing on log file
				if(self.logger): self.logger.addToLog(msg)
				#Check if either objective function value or gradient norm is NaN
				if(isnan(obj0) or isnan(prblm_grad.norm())): raise ValueError("ERROR! Either gradient norm or objective function value NaN!")
				#Set internal delta residual vector
				if(not precond): cg_dres = prblm_res.clone()
				if(not precond): cg_dres.zero()
			if(prblm_grad.norm() == 0.):
				print("Gradient vanishes identically")
				break

			#Saving results
			self.save_results(iter,prblm,force_save=False)

			#Computing alpha and beta coefficients
			if(precond):
				#Applying preconditioning to current gradient
				prblm.prec.forward(False,prblm_grad,cg_prec_grad)
				if(iter==0 or self.steepest):
					#Steepest descent
					beta = 0.0
					dot_grad_prec_grad=prblm_grad.dot(cg_prec_grad)
				else:
					#Conjugate-gradient coefficients for preconditioned CG
					dot_grad_prec_grad_old = dot_grad_prec_grad
					if(dot_grad_prec_grad_old==0.0):
						success = False
						#Writing on log file
						if(self.logger): self.logger.addToLog("Gradient orthogonal to preconditioned one, will terminate solver")
					dot_grad_prec_grad=prblm_grad.dot(cg_prec_grad)
					beta = dot_grad_prec_grad/dot_grad_prec_grad_old
				#Update search direction
				cg_dmodl.scaleAdd(cg_prec_grad,beta,1.0)
				cg_dmodld=prblm.get_dres(cg_mdl,cg_dmodl)		#Project search direction into the data space
				dot_cg_dmodld=cg_dmodld.dot(cg_dmodld)
				if(dot_cg_dmodld==0.0):
					success = False
					#Writing on log file
					if(self.logger): self.logger.addToLog("Search direction orthogonal to span of linear operator, will terminate solver")
				else:
					alpha = - dot_grad_prec_grad/dot_cg_dmodld
					#Writing on log file
					if(beta==0.0):
						msg="Steppest-descent step length: %s"%(alpha)
					else:
						msg="Conjugate alpha,beta: %s,%s"%(alpha,beta)
					if(self.logger): self.logger.addToLog(msg)
			else:
				prblm_gradd=prblm.get_dres(cg_mdl,prblm_grad)	#Project gradient into the data space
				#Computing alpha and beta coefficients
				if(iter==0 or self.steepest):
					#Steepest descent
					beta = 0.0
					dot_gradd=prblm_gradd.dot(prblm_gradd)
					if(dot_gradd==0.0):
						success = False
						#Writing on log file
						if(self.logger): self.logger.addToLog("Gradient orthogonal to span of linear operator, will terminate solver")
					else:
						dot_gradd_res=prblm_gradd.dot(prblm_res)
						alpha = - dot_gradd_res/dot_gradd
						msg="Steppest-descent step length: %s"%(alpha)
						#Writing on log file
						if(iter==0):
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

			if(precond):
				#modl = modl + alpha * dmodl
				cg_mdl.scaleAdd(cg_dmodl,1.0,alpha)					#Update model
			else:
				#dmodl = alpha * grad + beta * dmodl
				cg_dmodl.scaleAdd(prblm_grad,beta,alpha) 			#update search direction
				#modl = modl + dmodl
				cg_mdl.scaleAdd(cg_dmodl)							#Update model

			#Increasing iteration counter
			iter = iter + 1
			#Setting the model
			prblm.set_model(cg_mdl)
			#Projecting model onto the bounds (if any)
			if("bounds" in dir(prblm)): prblm.bounds.apply(cg_mdl)

			if(prblm_mdl.isDifferent(cg_mdl)):
				#Model went out of the bounds
				msg="Model hit provided bounds. Projecting it onto them."
				if(self.logger): self.logger.addToLog(msg)
				#Recomputing m_current = m_new - dmodl
				prblm_mdl.scaleAdd(cg_dmodl,1.0,-1.0)
				#Finding the projected dmodl = m_new_clipped - m_current
				cg_dmodl.copy(cg_mdl)
				cg_dmodl.scaleAdd(prblm_mdl,1.0,-1.0)
				prblm.set_model(cg_mdl)
				if(precond):
					cg_dmodl.scale(1.0/alpha) #Unscaling the search direction
				else:
					#copying previos residuals dres = res_old
					cg_dres.copy(prblm_res)
					#Computing actual change in the residual vector dres = res_new - res_old
					prblm_res=prblm.get_res(cg_mdl) #New residual vector
					cg_dres.scaleAdd(prblm_res,-1.0,1.0)
			else:
				#Setting residual vector to avoid its unnecessary computation (if model was not clipped)
				if(precond):
					#res = res + alpha * dres
					prblm_res.scaleAdd(cg_dmodld,1.0,alpha)			#Update residuals
				else:
					#dres  = alpha * gradd + beta * dres
					cg_dres.scaleAdd(prblm_gradd,beta,alpha)		#Update residual step
					#res = res + dres
					prblm_res.scaleAdd(cg_dres)						#Update residuals
				prblm.set_residual(prblm_res)


			#Computing new objective function value
			obj1=prblm.get_obj(cg_mdl)
			if(obj1 >= obj0):
				msg = "Objective function didn't reduce, will terminate solver: obj_new=%s obj_current=%s"%(obj1,obj0)
				if(verbose): print(msg)
				#Writing on log file
				if(self.logger): self.logger.addToLog(msg)
				#Stepping back to the previous solution
				if(precond):
					cg_mdl.scaleAdd(cg_dmodl,1.0,-alpha)
				else:
					cg_mdl.scaleAdd(cg_dmodl,1.0,-1.0)
				prblm.set_model(cg_mdl)
				break

			#Saving current model and previous search direction in case of restart
			self.restart.save_parameter("iter",iter)
			self.restart.save_vector("cg_mdl",cg_mdl)
			self.restart.save_vector("cg_dmodl",cg_dmodl)
			#Saving data space vectors or scaling if preconditioned
			if(not precond):
				self.restart.save_vector("cg_dres",cg_dres)
			else:
				self.restart.save_parameter("dot_grad_prec_grad",dot_grad_prec_grad)
			self.restart.save_vector("prblm_res",prblm_res)

			#iteration info
			msg = "iter = %s obj = %s residual norm = %s gradient norm= %s feval = %s"%(iter,obj1,prblm.get_rnorm(cg_mdl),prblm.get_gnorm(cg_mdl),prblm.get_fevals())
			if(verbose): print(msg)
			#Writing on log file
			if(self.logger): self.logger.addToLog("\n"+msg)
			#Check if either objective function value or gradient norm is NaN
			if(isnan(obj1) or isnan(prblm_grad.norm())): raise ValueError("ERROR! Either gradient norm or objective function value NaN!")
			if (self.stoppr.run(prblm,iter,initial_obj_value,verbose)): break

		#Writing last inverted model
		self.save_results(iter,prblm,force_save=True,force_write=True)
		msg = ""
		if(precond): msg="PRECONDITIONED "
		if(self.steepest):
			if(self.logger): self.logger.addToLog(msg+"LINEAR STEEPEST-DESCENT SOLVER log file end")
		else:
			if(self.logger): self.logger.addToLog(msg+"LINEAR CONJUGATE GRADIENT SOLVER log file end")
		#Clear restart object
		self.restart.clear_restart()

		return
