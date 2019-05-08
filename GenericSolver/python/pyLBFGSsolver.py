#Module containing L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno) solver class
import pySolver
from pyOperator import scalingOp
import pyStepperCvSrch as StepperMT

import numpy as np
from collections import deque
from math import isnan

#Testing if genericIO and SepVector module is present
import imp
try:
	imp.find_module('genericIO')
	import genericIO
	SepVector=genericIO.SepVector
	genIO_found = True
except ImportError:
	genIO_found = False



class LBFGSsolver(pySolver.Solver):
	"""L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno) Solver object"""


	def __init__(self,stoppr,stepper=None,save_alpha=False,m_steps=None,save_est=False,H0=None,prefix=None,logger=None):
		"""
		   Constructor for LBFGS Solver.
		   stoppr     = [no default] - stopper class; Stopper object necessary to terminate the solver
		   stepper    = [CvSrch] - stepper class; Stepper object necessary to perform line-search step
		   save_alpha = [False] - boolean; Use previous step-length value as initial guess. Otherwise, the algorithm starts from an initial guess of 1.0
		   m_steps    = [None] - int; Maximum number of steps to store to estimate the inverse Hessian (by default it runs BFGS method)
		   save_est   = [False] - boolean; Flag to save inverse Hessian estimate vectors (note self.prefix must be different than None)
		   H0         = [None] - operator class; Operator object for the initial estimated Hessian inverse (by default it assumes an identity operator)
		   prefix 	  = [None] - string; Prefix of the the files in which the vectors of the estimate Hessian inverse will be saved
		   logger 	  = [None] - logger class; Logger object to save inversion information at runtime
		"""
		#Calling parent construction
		super(LBFGSsolver,self).__init__()
		#Defining stopper object
		self.stoppr=stoppr
		#Defining stepper object
		if(stepper != None):
			self.stepper=stepper
		else:
			self.stepper=StepperMT.CvSrchStep()
		#Logger object to write on log file
		self.logger=logger
		#Overwriting logger of the Stopper object
		self.stoppr.logger=self.logger
		#LBFGS-specific parameters
		self.save_alpha = save_alpha
		self.H0 = H0
		self.m_steps = m_steps
		self.save_est = save_est
		self.prefix = prefix
		self.tmp_vector = None #A copy of the model vector will be create when the function run is invoked
		return

	def save_hessian_estimate(self,index,iter):
		"Function to save current vector of estimated Hessian inverse"
		#index of the step to save
		if(self.prefix != None and self.save_est):
			step_filename = self.prefix + "step_vector_%s.H"%(iter)
			grad_diff_filename = self.prefix + "grad_diff_vector_%s.H"%(iter)
			if(genIO_found and self.use_SepVector): #Writing using genericIO and SepVector
				genericIO.defaultIO.writeVector(step_filename,self.step_vectors[index])
				genericIO.defaultIO.writeVector(grad_diff_filename,self.grad_diff_vectors[index])
			else:
				self.step_vectors[index].writeVec(step_filename)
				self.grad_diff_vectors[index].writeVec(grad_diff_filename)
		return

	def check_rho(self,denom_dot,step_index,iter):
		'''Function to check scaling factor of Hessian inverse estimate'''
		if (denom_dot == 0.0):
			if(self.m_steps != None):
				self.rho[step_index] = 0.0
			else:
				self.rho.append(0.0)
			msg = "Skipping update to estimated Hessian; y vector orthogonal to s vector at iteration %s"%(iter)
			if(self.logger): self.logger.addToLog(msg)
		elif (denom_dot < 0.0):
			if(self.m_steps != None):
				self.rho[step_index] = 0.0
			else:
				self.rho.append(0.0)
			msg = "Skipping update to estimated Hessian; not positive at iteration %s"%(iter)
			if(self.logger): self.logger.addToLog(msg)
		else:
			if(self.m_steps != None):
				self.rho[step_index] = 1.0/denom_dot
			else:
				self.rho.append(1.0/denom_dot)
			#Saving current update for inverse Hessian estimate (i.e., gradient-difference and model-step vectors)
			self.save_hessian_estimate(step_index,iter)
		return

	#BFGSMultiply function
	def BFGSMultiply(self,dmodl,grad,iter):
		"""Function to apply approximated inverse Hessian"""
		#Array containing dot-products
		if(self.m_steps != None):
			alpha=[0.0]*self.m_steps
			#Handling of limited memory
			if (iter <= self.m_steps):
				initial_point=0
			else:
				initial_point=iter%self.m_steps
			#Right step list
			rloop = deque(range(0,min(iter,self.m_steps)))
			rloop.reverse()
			#Rotate the list
			rloop.rotate(initial_point)
			#Left step list
			lloop = deque(range(0,min(iter,self.m_steps)))
			#Rotate the list
			lloop.rotate(-initial_point)
		else:
			alpha=[0.0]*iter
			rloop = deque(range(0,iter))
			rloop.reverse()
			lloop = deque(range(0,iter))
		# r = -grad
		dmodl.copy(grad)
		dmodl.scale(-1.0)
		#Apply right-hand series of operators
		for ii in rloop:
			#Check positivity, if not true skip the update
			if (self.rho[ii] > 0.0):
				# alpha_i=rho_i*s_i'r
				alpha[ii]=self.rho[ii]*self.step_vectors[ii].dot(dmodl)
				# r=r-alpha_i*y_i
				dmodl.scaleAdd(self.grad_diff_vectors[ii],1.0,-alpha[ii])
		#Comput center (If not provide Identity matrix is assumed)
		# r=H0r
		if(self.H0!=None):
			#Apply a forward of the initial Hessian estimate
			self.H0.forward(False,dmodl,self.tmp_vector)
			dmodl.copy(self.tmp_vector)
		#Apply left-hand series of operators
		for ii in lloop:
			#Check positivity, if not true skip the update
			if (self.rho[ii] > 0.0):
				# beta=rhoiyi'r
				beta=self.rho[ii]*self.grad_diff_vectors[ii].dot(dmodl)
				dmodl.scaleAdd(self.step_vectors[ii],1.0,alpha[ii]-beta)
		return

	def run(self,prblm,verbose=False,restart=False):
		"""Running LBFGS solver"""
		#Resetting stopper before running the inversion
		self.stoppr.reset()
		#Preliminary variables for Hessian inverse estimation
		if(self.m_steps != None):
			self.step_vectors = [None]*self.m_steps #s_i vectors
			self.grad_diff_vectors = [None]*self.m_steps #y_i vectors
			self.rho = [None]*self.m_steps #Scalar term necessary for Hessian inverse estimation
		else:
			self.step_vectors = [] #s_i vectors
			self.grad_diff_vectors = [] #y_i vectors
			self.rho = [] #Scalar term necessary for Hessian inverse estimation

		if(not restart):
			if(self.m_steps != None):
				msg = "Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) algorithm log file\n"
				msg += "Maximum number of steps to be used for Hessian inverse estimation: %s \n"%(self.m_steps)
			else:
				msg = "Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm log file\n"
			#Printing restart folder
			msg+="Restart folder: %s\n"%(self.restart.restart_folder)
			if(verbose): print(msg.replace("log file",""))
			if(self.logger): self.logger.addToLog(msg)

			#Setting internal vectors (model, search direction, and previous gradient vectors)
			prblm_mdl=prblm.get_model()
			bfgs_mdl = prblm_mdl.clone()
			bfgs_dmodl = prblm_mdl.clone()
			bfgs_dmodl.zero()
			bfgs_grad0 = bfgs_dmodl.clone()

			#Other internal variables
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
			bfgs_mdl = self.restart.retrieve_vector("bfgs_mdl")
			bfgs_dmodl = self.restart.retrieve_vector("bfgs_dmodl")
			bfgs_grad0 = self.restart.retrieve_vector("bfgs_grad0")
			#Setting the model and residuals to avoid residual twice computation
			prblm.set_model(bfgs_mdl)
			prblm_mdl=prblm.get_model()
			#Setting residual vector to avoid its unnecessary computation
			prblm.set_residual(self.restart.retrieve_vector("prblm_res"))
			#Retrieving Hessian inverse estimate
			self.rho=self.restart.retrieve_parameter("rho")
			for istep in range(iter):
				if(self.m_steps != None):
					if(istep < self.m_steps):
						self.grad_diff_files[istep] = self.restart.retrieve_vector("grad_diff_vectors%s.H"%(istep))
						self.step_files[istep] = self.restart.retrieve_vector("step_vectors%s.H"%(istep))
				else:
					self.grad_diff_files.append(self.restart.retrieve_vector("grad_diff_vectors%s.H"%(istep)))
					self.step_files.append(self.restart.retrieve_vector("step_vectors%s.H"%(istep)))

		#Common variables unrelated to restart
		success = True
		self.tmp_vector = bfgs_dmodl.clone()
		self.tmp_vector.zero()
		#For saving estimated Hessian vector
		self.use_SepVector = False
		if(genIO_found):
			self.use_SepVector = isinstance(prblm_mdl,SepVector.vector)


		#Inversion loop
		while True:
			#Computing objective function
			obj0 = prblm.get_obj(bfgs_mdl)    	#Compute objective function value
			prblm_res=prblm.get_res(bfgs_mdl) 	#Compute residuals
			prblm_grad=prblm.get_grad(bfgs_mdl) 	#Compute the gradient
			if(iter==0):
				initial_obj_value=obj0 			#For relative objective function value
				#Saving objective function value
				self.restart.save_parameter("obj_initial",initial_obj_value)
				msg = "iter = %s obj = %s residual norm = %s gradient norm= %s feval = %s geval = %s"%(iter,obj0,prblm.get_rnorm(bfgs_mdl),prblm.get_gnorm(bfgs_mdl),prblm.get_fevals(),prblm.get_gevals())
				if(verbose): print(msg)
				#Writing on log file
				if(self.logger): self.logger.addToLog(msg)
				#Check if either objective function value or gradient norm is NaN
				if(isnan(obj0) or isnan(prblm_grad.norm())): raise ValueError("Error! Either gradient norm or objective function value NaN!")
			if(prblm_grad.norm() == 0.):
				print("Gradient vanishes identically")
				break

			#Saving results
			self.save_results(iter,prblm,force_save=False)

			#Applying approximated Hessian inverse
			msg = "Appplying inverse Hessian estimate"
			if(self.m_steps != None):
				msg += "\nCurrent inverse dot-products of BFGS estimation vectors %s"%(self.rho[0:min(self.m_steps,iter)])
			else:
				if(len(self.rho) > 0):
					msg += "\nCurrent inverse dot-products of BFGS estimation vectors %s"%(self.rho)
			if(self.logger): self.logger.addToLog(msg)
			self.BFGSMultiply(bfgs_dmodl,prblm_grad,iter)
			msg = "Done applying inverse Hessian estimate"
			if(self.logger): self.logger.addToLog(msg)

			#grad0 = grad
			bfgs_grad0.copy(prblm_grad)
			#Calling line search
			alpha,success=self.stepper.run(prblm,bfgs_mdl,bfgs_dmodl,self.logger)
			if(not success):
				msg = "Stepper couldn't find a proper step size, will terminate solver"
				if(verbose): print(msg)
				#Writing on log file
				if(self.logger): self.logger.addToLog(msg)
				break

			obj1 = prblm.get_obj(bfgs_mdl)    	#Compute objective function value
			#Redundant test on verifying convergence
			if(obj0<=obj1):
				msg = "Objective function at new point greater or equal than previous one: obj_fun_old=%s obj_fun_new=%s\nPotential issue in the stepper or in revaluation of objective function!"%(obj0,obj1)
				if(self.logger): self.logger.addToLog(msg)
				raise ValueError(msg)

			#Compute new gradient
			prblm_grad=prblm.get_grad(bfgs_mdl)
			#Compute updates for estimated Hessian inverse
			if(self.m_steps != None):
				#LBFGS
				step_index = iter%self.m_steps #Modulo to handle limited memory
				# yn+1=gn+1-gn
				self.grad_diff_vectors[step_index] = bfgs_grad0.clone()
				self.grad_diff_vectors[step_index].scaleAdd(prblm_grad,-1.0,1.0)
				# sn+1=xn+1-xn = alpha * dmodl
				self.step_vectors[step_index]=bfgs_dmodl.clone()
				self.step_vectors[step_index].scale(alpha)
			else:
				#BFGS
				step_index = iter
				# yn+1=gn+1-gn
				self.grad_diff_vectors.append(bfgs_grad0.clone())
				self.grad_diff_vectors[step_index].scaleAdd(prblm_grad,-1.0,1.0)
				# sn+1=xn+1-xn = alpha * dmodl
				self.step_vectors.append(bfgs_dmodl.clone())
				self.step_vectors[step_index].scale(alpha)
			#	rhon+1=1/yn+1'sn+1
			denom_dot=self.grad_diff_vectors[step_index].dot(self.step_vectors[step_index])
			#Checking rho
			self.check_rho(denom_dot,step_index,iter)

			#Making first step-length value Hessian guess if not provided by user
			if(iter == 0 and self.H0 == None):
				self.restart.save_parameter("fist_alpha",alpha)
				self.H0 = scalingOp(bfgs_dmodl,alpha)
				if(self.logger): self.logger.addToLog("	First step-length value used as first Hessian inverse estimate!")
				self.stepper.alpha = 1.0

			#Increasing iteration counter
			iter = iter + 1

			#Using alpha = 1.0 after first iteration
			if(iter != 0 and not self.save_alpha): self.stepper.alpha = 1.0

			#Saving current model and previous search direction in case of restart
			self.restart.save_parameter("iter",iter)
			self.restart.save_parameter("alpha",alpha)
			self.restart.save_vector("bfgs_mdl",bfgs_mdl)
			self.restart.save_vector("bfgs_dmodl",bfgs_dmodl)
			self.restart.save_vector("bfgs_grad0",bfgs_grad0)
			#Saving Inverse Hessian estimate for restart
			self.restart.save_parameter("rho",self.rho)
			self.restart.save_vector("grad_diff_vectors%s.H"%(step_index),self.grad_diff_vectors[step_index])
			self.restart.save_vector("step_vectors%s.H"%(step_index),self.step_vectors[step_index])
			#Saving data space vectors
			self.restart.save_vector("prblm_res",prblm_res)

			#iteration info
			msg = "iter = %s obj = %s residual norm = %s gradient norm= %s feval = %s geval = %s"%(iter,obj1,prblm.get_rnorm(bfgs_mdl),prblm.get_gnorm(bfgs_mdl),prblm.get_fevals(),prblm.get_gevals())
			if(verbose): print(msg)
			#Writing on log file
			if(self.logger): self.logger.addToLog("\n"+msg)
			#Check if either objective function value or gradient norm is NaN
			if(isnan(obj1) or isnan(prblm_grad.norm())): raise ValueError("Error! Either gradient norm or objective function value NaN!")
			if (self.stoppr.run(prblm,iter,initial_obj_value,verbose)): break





		#Writing last inverted model
		self.save_results(iter,prblm,force_save=True,force_write=True)
		if(self.m_steps != None):
			if(self.logger): self.logger.addToLog("Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) algorithm log file end")
		else:
			if(self.logger): self.logger.addToLog("Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm log file end")
		self.restart.clear_restart()
		del self.tmp_vector; self.tmp_vector = None
		return
