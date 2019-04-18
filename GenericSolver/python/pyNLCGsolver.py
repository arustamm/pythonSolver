#Module containing Non-Linear Conjugate-Gradient Solver class
#It can also handle steppest-descent algorithm
import pySolver
import pyStepperParabolic
from math import isnan

def betaFR(grad,grad0,dir,logger):
	"""Fletcher and Reeves method"""
	#betaFR = sum(dprod(g,g))/sum(dprod(g0,g0))
	dot_grad=grad.dot(grad)
	dot_grad0=grad0.dot(grad0)
	if (dot_grad0 == 0.): #Avoid division by zero
		beta = 0.
		if(logger): logger.addToLog("Setting beta to zero since norm of previous gradient is zero!!!")
	else:
		beta = dot_grad/dot_grad0
	return beta

def betaPRP(grad,grad0,dir,logger):
	"""Polak, Ribiere, Polyak method"""
	#betaPRP = sum(dprod(g,g-g0))/sum(dprod(g0,g0))
	tmp1=grad.clone()
	#g-g0
	tmp1.scaleAdd(grad0,1.0,-1.0)
	dot_num=tmp1.dot(grad)
	dot_grad0=grad0.dot(grad0)
	if (dot_grad0 == 0.): #Avoid division by zero
		beta = 0.
		if(logger): logger.addToLog("Setting beta to zero since norm of previous gradient is zero!!!")
	else:
		beta = dot_num/dot_grad0
	return beta

def betaHS(grad,grad0,dir,logger):
	"""Hestenes and Stiefel"""
	#betaHS = sum(dprod(g,g-g0))/sum(dprod(d,g-g0))
	tmp1=grad.clone()
	#g-g0
	tmp1.scaleAdd(grad0,1.0,-1.0)
	dot_num=tmp1.dot(grad)
	dot_denom=tmp1.dot(dir)
	if (dot_denom == 0.): #Avoid division by zero
		beta = 0.
		if(logger): logger.addToLog("Setting beta to zero since norm of denominator is zero!!!")
	else:
		beta = dot_num/dot_denom
	return beta

def betaCD(grad,grad0,dir,logger):
	"""Conjugate Descent"""
	#betaCD = -sum(dprod(g,g))/sum(dprod(d,g0))
	dot_num=grad.dot(grad)
	dot_denom=-grad0.dot(dir)
	if (dot_denom == 0.): #Avoid division by zero
		beta = 0.
		if(logger): logger.addToLog("Setting beta to zero since norm of denominator is zero!!!")
	else:
		beta = dot_num/dot_denom
	return beta

def betaLS(grad,grad0,dir,logger):
	"""Liu and Storey"""
	#betaLS = -sum(dprod(g,g-g0))/sum(dprod(d,g0))
	tmp1=grad.clone()
	#g-g0
	tmp1.scaleAdd(grad0,1.0,-1.0)
	dot_num=tmp1.dot(grad)
	dot_denom=-grad0.dot(dir)
	if (dot_denom == 0.): #Avoid division by zero
		beta = 0.
		if(logger): logger.addToLog("Setting beta to zero since norm of denominator is zero!!!")
	else:
		beta = dot_num/dot_denom
	return beta

def betaDY(grad,grad0,dir,logger):
	"""Dai and Yuan"""
	#betaDY = sum(dprod(g,g))/sum(dprod(d,g-g0))
	tmp1=grad.clone()
	#g-g0
	tmp1.scaleAdd(grad0,1.0,-1.0)
	dot_num=grad.dot(grad)
	dot_denom=tmp1.dot(dir)
	if (dot_denom == 0.): #Avoid division by zero
		beta = 0.
		if(logger): logger.addToLog("Setting beta to zero since norm of denominator is zero!!!")
	else:
		beta = dot_num/dot_denom
	return beta

def betaBAN(grad,grad0,dir,logger):
	"""Bamigbola, Ali and Nwaeze"""
	#betaDY = sum(dprod(g,g-g0))/sum(dprod(g0,g-g0))
	tmp1=grad.clone()
	#g-g0
	tmp1.scaleAdd(grad0,1.0,-1.0)
	dot_num=tmp1.dot(grad)
	dot_denom=tmp1.dot(grad0)
	if (dot_denom == 0.): #Avoid division by zero
		beta = 0.
		if(logger): logger.addToLog("Setting beta to zero since norm of denominator is zero!!!")
	else:
		beta = -dot_num/dot_denom
	return beta

def betaHZ(grad,grad0,dir,logger):
	"""Hager and Zhang"""
	#betaN = sum(dprod(g-g0-2*sum(dprod(g-g0,g-g0))*d/sum(dprod(d,g-g0)),g))/sum(dprod(d,g-g0))
	tmp1=grad.clone()
	#g-g0
	tmp1.scaleAdd(grad0,1.0,-1.0)
	#sum(dprod(g-g0,g-g0))
	dot_diff_g_g0=tmp1.dot(tmp1)
	#sum(dprod(d,g-g0))
	dot_dir_diff_g_g0=tmp1.dot(dir)
	if (dot_dir_diff_g_g0 == 0.): #Avoid division by zero
		beta = 0.
		if(logger): logger.addToLog("Setting beta to zero since norm of denominator is zero!!!")
	else:
		#g-g0-2*sum(dprod(g-g0,g-g0))*d/sum(dprod(d,g-g0))
		tmp1.scaleAdd(dir,1.0,-2.0*dot_diff_g_g0/dot_dir_diff_g_g0)
		#sum(dprod(g-g0-2*sum(dprod(g-g0,g-g0))*d/sum(dprod(d,g-g0)),g))
		dot_num=grad.dot(tmp1)
		#dot_num/sum(dprod(d,g-g0))
		beta = dot_num/dot_dir_diff_g_g0
	return beta

def betaSD(grad,grad0,dir,logger):
	"""Steepest descent"""
	beta = 0.
	return beta

#Beta functions
#grad=new gradient, grad0=old, dir=search direction
#From A SURVEY OF NONLINEAR CONJUGATE GRADIENT METHODS

class NLCGsolver(pySolver.Solver):
	"""Non-Linear Conjugate Gradient and Steepest-Descent Solver object"""

	#Default class methods/functions
	def __init__(self,stoppr,stepper=None,beta_type="FR",logger=None):
		"""
		   Constructor for NLCG Solver

		"""
		#Calling parent construction
		super(NLCGsolver,self).__init__()
		#Defining stopper object
		self.stoppr=stoppr
		#Defining stepper object
		if(stepper != None):
			self.stepper=stepper
		else:
			self.stepper=pyStepperParabolic.ParabolicStep()
		#Beta function to use during the inversion
		self.beta_type=beta_type
		#Logger object to write on log file
		self.logger=logger
		#Overwriting logger of the Stopper object
		self.stoppr.logger=self.logger
		return

	def __del__(self):
		"""Default destructor"""
		return

	def beta_func(self,grad,grad0,dir):
		"""Beta function interface"""
		beta_type = self.beta_type
		if(beta_type == "FR"):
			beta = betaFR(grad,grad0,dir,self.logger)
		elif(beta_type == "PRP"):
			beta = betaPRP(grad,grad0,dir,self.logger)
		elif(beta_type == "HS"):
			beta = betaHS(grad,grad0,dir,self.logger)
		elif(beta_type == "CD"):
			beta = betaCD(grad,grad0,dir,self.logger)
		elif(beta_type == "LS"):
			beta = betaLS(grad,grad0,dir,self.logger)
		elif(beta_type == "DY"):
			beta = betaDY(grad,grad0,dir,self.logger)
		elif(beta_type == "BAN"):
			beta = betaBAN(grad,grad0,dir,self.logger)
		elif(beta_type == "HZ"):
			beta = betaHZ(grad,grad0,dir,self.logger)
		elif(beta_type == "SD"):
			beta = betaSD(grad,grad0,dir,self.logger)
		else:
			raise ValueError("ERROR! Requested Beta function type not existing")
		return beta

	def run(self,prblm,verbose=False,restart=False):
		"""Running NLCG or steppest-descent solver"""
		#Resetting stopper before running the inversion
		self.stoppr.reset()
		if(not restart):
			if(self.beta_type == "SD"):
				msg="NON-LINEAR STEEPEST-DESCENT SOLVER log file\n"
			else:
				msg="NON-LINEAR CONJUGATE-GRADIENT SOLVER log file\n"
				msg+="Conjugate method used: %s \n"%(self.beta_type)
			#Printing restart folder
			msg+="Restart folder: %s\n"%(self.restart.restart_folder)
			if(verbose): print(msg.replace("log file",""))
			if(self.logger): self.logger.addToLog(msg)

			#Setting internal vectors (model, search direction, and previous gradient vectors)
			prblm_mdl=prblm.get_model()
			cg_mdl = prblm_mdl.clone()
			cg_dmodl = prblm_mdl.clone()
			cg_dmodl.zero()
			cg_grad0 = cg_dmodl.clone()

			#Other internal variables
			beta = 0.0
			iter = 0
		else:
			#Retrieving parameters and vectors to restart the solver
			msg="Restarting previous solver run from: %s"%(self.restart.restart_folder)
			if(verbose): print(msg)
			if(self.logger): self.logger.addToLog(msg)
			self.restart.read_restart()
			iter = self.restart.retrieve_parameter("iter")
			self.stepper.alpha = self.restart.retrieve_parameter("alpha")
			initial_obj_value=self.restart.retrieve_parameter("obj_initial") #Retrieving initial objective function value
			cg_mdl = self.restart.retrieve_vector("cg_mdl")
			cg_dmodl = self.restart.retrieve_vector("cg_dmodl")
			cg_grad0 = self.restart.retrieve_vector("cg_grad0")
			#Setting the model and residuals to avoid residual twice computation
			prblm.set_model(cg_mdl)
			prblm_mdl=prblm.get_model()
			#Setting residual vector to avoid its unnecessary computation
			prblm.set_residual(self.restart.retrieve_vector("prblm_res"))

		#Common variables unrelated to restart
		success = True

		while True:
			#Computing objective function
			obj0 = prblm.get_obj(cg_mdl)    	#Compute objective function value
			prblm_res=prblm.get_res(cg_mdl) 	#Compute residuals
			prblm_grad=prblm.get_grad(cg_mdl) 	#Compute the gradient
			if(iter==0):
				initial_obj_value=obj0 			#For relative objective function value
				#Saving objective function value
				self.restart.save_parameter("obj_initial",initial_obj_value)
				msg = "iter = %s obj = %s residual norm = %s gradient norm= %s feval = %s geval = %s"%(iter,obj0,prblm.get_rnorm(cg_mdl),prblm.get_gnorm(cg_mdl),prblm.get_fevals(),prblm.get_gevals())
				if(verbose): print(msg)
				#Writing on log file
				if(self.logger): self.logger.addToLog(msg)
				#Check if either objective function value or gradient norm is NaN
				if(isnan(obj0) or isnan(prblm_grad.norm())): raise ValueError("ERROR! Either gradient norm or objective function value NaN!")
			if(prblm_grad.norm() == 0.):
				print("Gradient vanishes identically")
				break

			#Saving results
			self.save_results(iter,prblm,force_save=False)

			if(iter >= 1):
				beta = self.beta_func(prblm_grad, cg_grad0, cg_dmodl)
				if(beta < 0.):
					if(self.logger): self.logger.addToLog("Beta negative setting to zero: beta value=%s!!!"%(beta))
					beta = 0.
			#Writing on log file
			if(self.beta_type != "SD"):
				if(self.logger): self.logger.addToLog("beta coefficient: %s"%(beta))

			#dmodl = beta*dmodl - grad
			cg_dmodl.scaleAdd(prblm_grad,beta,-1.0)
			#grad0 = grad
			cg_grad0.copy(prblm_grad)
			#Calling line search
			alpha,success=self.stepper.run(prblm,cg_mdl,cg_dmodl,self.logger)
			if(not success):
				msg = "Stepper couldn't find a proper step size, will terminate solver"
				if(verbose): print(msg)
				#Writing on log file
				if(self.logger): self.logger.addToLog(msg)
				break

			#Increasing iteration counter
			iter = iter + 1
			obj1 = prblm.get_obj(cg_mdl)    	#Compute objective function value
			#Redundant test on verifying convergence
			if(obj0<obj1):
				msg = "Objective function at new point greater or equal than previous one: obj_fun_old=%s obj_fun_new=%s\nPotential issue in the stepper or in revaluation of objective function! Solver will stop!"%(obj0,obj1)
				if(self.logger): self.logger.addToLog(msg)
				if(verbose): print(msg)
				cg_mdl.scaleAdd(cg_dmodl,1.0,-alpha)
				prblm.set_model(cg_mdl)
				break


			#Saving current model and previous search direction in case of restart
			self.restart.save_parameter("iter",iter)
			self.restart.save_parameter("alpha",alpha)
			self.restart.save_vector("cg_mdl",cg_mdl)
			self.restart.save_vector("cg_dmodl",cg_dmodl)
			self.restart.save_vector("cg_grad0",cg_grad0)
			#Saving data space vectors
			self.restart.save_vector("prblm_res",prblm_res)
			#iteration info
			msg = "iter = %s obj = %s residual norm = %s gradient norm= %s feval = %s geval = %s"%(iter,obj1,prblm.get_rnorm(cg_mdl),prblm.get_gnorm(cg_mdl),prblm.get_fevals(),prblm.get_gevals())
			if(verbose): print(msg)
			#Writing on log file
			if(self.logger): self.logger.addToLog("\n"+msg)
			#Check if either objective function value or gradient norm is NaN
			if(isnan(obj1) or isnan(prblm_grad.norm())): raise ValueError("ERROR! Either gradient norm or objective function value NaN!")
			if (self.stoppr.run(prblm,iter,initial_obj_value,verbose)): break

		#Writing last inverted model
		self.save_results(iter,prblm,force_save=True,force_write=True)
		if(self.beta_type == "SD"):
			if(self.logger): self.logger.addToLog("NON-LINEAR STEEPEST-DESCENT SOLVER log file end")
		else:
			if(self.logger): self.logger.addToLog("NON-LINEAR CONJUGATE-GRADIENT SOLVER log file end")
		#Clear restart object
		self.restart.clear_restart()

		return
