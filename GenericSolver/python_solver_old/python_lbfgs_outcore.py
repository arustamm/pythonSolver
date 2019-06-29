#module containing object for L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno) solver
import sys
import operator
import math
import sep_python as sep
import python_solver_outcore as solv
import stepper_sample as simp_stpr
import stopper_basic as simp_stop
import operator_obj as operator
from collections import deque
from numbers import Number
from math import isnan



class lbfgs_solver(solv.solver):

	def __init__(self,m_steps,stoppr=simp_stop.stopper_basic(),H0=None,step_file_prefix=None,diff_grad_prefix=None):
		"""Constructor for l-bfgs solver"""
		#Defining stepper and stopper objects
		self.stoppr=stoppr
		self.files_to_clean=[]
		self.m_steps=m_steps #Steps to use in order to estimate the Hessian inverse
		self.H0=H0 #Initial Hessian inverse estimate (operator object if different than identity!)
		self.tmp_LBFGSmulti_file=sep.tmp_file("tmp_LBFGSmulti.H"); self.files_to_clean.append(self.tmp_LBFGSmulti_file)
		self.rho=[0.0]*self.m_steps #weights for Hessian inverse estimation
		self.removefiles=True
		#If different than None it will save the vectors of the estimated Hessian inverse
		self.step_file_prefix=step_file_prefix
		self.diff_grad_prefix=diff_grad_prefix
		return

	def __del__(self):
		"""Overriding default destructor"""
		import sep_python as sep
		if hasattr(self, 'removefiles'):
			if (self.removefiles):
				sep.Rm(self.files_to_clean)
			else:
				for ifile in self.files_to_clean:
					print "	Temporary file not removed for debugging in L-BFGS: %s"%(ifile)
		return

	def save_hessian_estimate(self,index,iter):
		#index of the step to save
		n_estimate=iter-1
		if (self.step_file_prefix != None):
			sep.Cp(self.step_files[index],self.step_file_prefix+"%s.H"%(n_estimate))
		if (self.diff_grad_prefix != None):
			sep.Cp(self.grad_diff_files[index],self.diff_grad_prefix+"%s.H"%(n_estimate))
		return

	#BFGSMultiply function
	def BFGSMultiply(self,dmodl,grad,iter):
		"""Function to apply approximated inverse Hessian"""
		#Array containing dot-products
		alpha=[0.0]*self.m_steps
		#Handling of limited memory
		if ((iter-1)<=self.m_steps):
			initial_point=0
		else:
			initial_point=(iter-1)%self.m_steps
		# r = -grad
		sep.Sum(dmodl,grad,0.0,-1.0)
		#Apply right-hand series of operators
		rloop = deque(range(0,min(iter-1,self.m_steps)))
		rloop.reverse()
		#Rotate the list
		rloop.rotate(initial_point)
		for ii in rloop:
			#Check positivity, if not true skip the update
			if (self.rho[ii] > 0.0):
				# alphai=rhoisi'r
				alpha[ii]=(self.rho[ii]*sep.Dot_incore(self.step_files[ii],dmodl))
				# r=r-alphaiyi
				sep.Sum(dmodl,self.grad_diff_files[ii],1.0,-alpha[ii])
		#Comput center (If not provide Identity matrix is assumed)
		# r=H0r
		if(self.H0!=None):
			assert isinstance(self.H0,operator.Operator), "Initial provided Hessian inverse estimate not an operator class!"
			#Apply a forward modeling
			self.H0.set_input_output(dmodl,self.tmp_LBFGSmulti_file)
			stat=self.H0.run() #Apply initial Hessian inverse estimate operator
			assert stat==0, "problem running Hessian inverse estimate operator"
			sep.Cp(self.tmp_LBFGSmulti_file,dmodl)
		#Apply left-hand series of operators
		lloop = deque(range(0,min(iter-1,self.m_steps)))
		#Rotate the list
		lloop.rotate(-initial_point)
		for ii in lloop:
			#Check positivity, if not true skip the update
			if (self.rho[ii] > 0.0):
				# beta=rhoiyi'r
				beta=self.rho[ii]*sep.Dot_incore(self.grad_diff_files[ii],dmodl)
				sep.Sum(dmodl,self.step_files[ii],1.0,alpha[ii]-beta)
		stat = 0
		return stat

	def check_rho(self,denom_dot,step_index,log_file=None,iter=None):
		'''Function to check scaling factor of Hessian inverse estimate'''
		if (denom_dot==0):
			self.rho[step_index] = 0.0
			info = "Skipping update to estimated Hessian; y vector orthogonal to s vector at iteration %s"%(iter)
			solv.write_log_file(log_file,info=info)
		elif (denom_dot < 0.0):
			self.rho[step_index] = 0.0
			info = "Skipping update to estimated Hessian; not positive at iteration %s"%(iter)
			solv.write_log_file(log_file,info=info)
		else:
			self.rho[step_index]=1.0/denom_dot
			#Saving current update for inverse Hessian estimate (i.e., gradient-difference and model-step vectors)
			if(iter!=None): self.save_hessian_estimate(step_index,iter)
		return

	def run(self,prblm,stpr=simp_stpr.stepper_sample(),log_file=None):
		"""Running the solver"""
		#Writing first line in log file if present
		if(not prblm.restart.restarting):
			if(self.m_steps < self.stoppr.niter):
				solv.write_log_file(log_file,info="Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) algorithm log file")
				solv.write_log_file(log_file,info="Maximum number of steps to be used for Hessian inverse estimation: %s \n"%(self.m_steps))
			else:
				solv.write_log_file(log_file,info="Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm log file")
			solv.write_log_file(log_file,info="Restarting folder: %s\n"%(prblm.restart.restart_folder))
			if(hasattr(prblm, 'epsilon')):
				info = "Regularized inversion with epsilon value of: %s"%(prblm.epsilon)
				print info
				solv.write_log_file(log_file,info)
				if(prblm.op_reg_fwd == 'identity' and prblm.op_reg_adj == 'identity'):
					info = "Regularization operator used: IDENTIY OPERATOR\n"
				else:
					info = "Regularization operator used: USER DEFINED\n"
				print info
				solv.write_log_file(log_file,info)
			if(hasattr(stpr, 'maxval') and hasattr(stpr, 'minval')):
				info = ""
				if(stpr.maxval != None): 
					info += "Maximum allowed value for inverted model: %s\n"%(stpr.maxval)
				if(stpr.minval != None): 
					info += "Minimum allowed value for inverted model: %s\n"%(stpr.minval)
				if(info != ""):
					print info
					solv.write_log_file(log_file,info)
		
		#Set internal temporary files
		grad0=sep.tmp_file("grad0_lbfgs.H"); self.files_to_clean.append(grad0)
		dmodl=sep.tmp_file("dmodel_lbfgs.H");self.files_to_clean.append(dmodl)
		modl=sep.tmp_file("model_lbfgs.H");	 self.files_to_clean.append(modl)
		iter = 1
		#Set internal previous gradient file
		modl_prblm=prblm.get_model()
		sep.Cp(modl_prblm,grad0)
		sep.Zero(grad0)
		#Set internal search direction file
		sep.Cp(modl_prblm,dmodl)
		sep.Zero(dmodl)
		#Set internal model file
		sep.Cp(modl_prblm,modl)

		self.grad_diff_files=[]
		self.step_files=[]

		#Setting step and gradient difference file lists
		for ifile in range(self.m_steps):
			file=sep.tmp_file("grad_diff_lbfgs_%s.H"%(ifile)); self.files_to_clean.append(file)
			self.grad_diff_files.append(file)
			file=sep.tmp_file("step_lbfgs_%s.H"%(ifile)); self.files_to_clean.append(file)
			self.step_files.append(file)

		if(not prblm.restart.restarting):
			#Obtaining initial objective function value
			obj=prblm.get_obj(modl)
			obj0=sep.Get_value(obj) #For redundant test on stepper
			prblm.initial_obj_value=obj0 #For relative objective function value
			info = "iter = %s obj = %s residual norm = %s gradient norm= %s feval = %s"%(iter-1,obj0,prblm.get_rnorm(),prblm.get_gnorm(),prblm.get_fevals())
			print info
			#Writing on log file
			solv.write_log_file(log_file,info=info)
			#Creating restart folder
			prblm.restart.create_restart_folder()
			if(self.m_steps < self.stoppr.niter):
				solv.write_log_file(prblm.restart.log_file,info="Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) restart log file\n")
			else:
				solv.write_log_file(prblm.restart.log_file,info="Broyden-Fletcher-Goldfarb-Shanno (BFGS) restart log file\n")
			solv.write_log_file(prblm.restart.log_file,info="\n"+info)
			#Check if either objective function value or gradient norm is NaN
			assert not(isnan(obj0) or isnan(prblm.get_gnorm())), "Error! Either gradient norm or objective function value NaN!"
		while True:
			#Restart inversion from previous run
			if(prblm.restart.restarting):
				#Restarting the problem and getting iteration number
				iter=prblm.restart.set_restart(prblm,log_file)+1
				info="Restarting inversion from previous run from: %s"%(prblm.restart.restart_folder)
				solv.write_log_file(log_file,info)
				print info
				#Resetting current model and previous gradient
				prblm.restart.copy_file_from_restart("model_restart_LBFGS.H",modl)
				#Resetting inverse Hessian estimate
				for istep in range(iter-1):
					if(istep < self.m_steps):
						prblm.restart.copy_file_from_restart("grad_diff_restart_LBFGS_step%s.H"%(istep),self.grad_diff_files[istep])
						prblm.restart.copy_file_from_restart("step_files_restart_LBFGS_step%s.H"%(istep),self.step_files[istep])
						denom_dot=sep.Dot_incore(self.grad_diff_files[istep],self.step_files[istep])
						#Recomputing rho for restart
						self.check_rho(denom_dot,istep)
					else:
						break
				#Resetting last step length
				if hasattr(stpr, 'alpha'):
					stpr.alpha=float(prblm.restart.get_info_log("step_length"))
				prblm.initial_obj_value = float(prblm.restart.get_info_log("obj",iter_num=0))
				obj0=float(prblm.restart.get_info_log("obj"))

			#Compute problem gradient
			grad=prblm.get_grad(modl)
			if(prblm.get_gnorm() == 0.):
				print "Gradient vanishes identically"
				break
			#Outputing files if problem was not restarted
			if(not prblm.restart.restarting):
				prblm.output(modl)
			prblm.restart.restarting=False

			#Applying approximated Hessian inverse
			info = "Appplying inverse Hessian estimate"
			solv.write_log_file(log_file,info=info)
			info = "Current inverse dot-products of BFGS estimation vectors %s"%(self.rho[0:min(self.m_steps,iter-1)])
			solv.write_log_file(log_file,info=info)
			stat=self.BFGSMultiply(dmodl,grad,iter)
			info = "Done applying inverse Hessian estimate"
			solv.write_log_file(log_file,info=info)
			if(stat!=0):
				info = "Error applying BFGSMultiply!"
				solv.write_log_file(log_file,info=info)
				print info
				break

			#grad0=grad
			sep.Cp(grad,grad0)
			#Performing line search
			alpha,success=stpr.run(prblm,modl,dmodl,log_file)
			if(not success):
				info = "Stepper couldn't find a proper step size, will terminate solver"
				print info
				solv.write_log_file(log_file,info=info)
				break
			obj1=sep.Get_value(prblm.get_obj(modl))
			#Redundant test on verifying convergence
			if(obj0<obj1):
				info = "Objective function at new point greater or equal than previous one: obj_fun_old=%s obj_fun_new=%s\nPotential issue in the stepper or in revaluation of objective function!"%(obj0,obj1)
				solv.write_log_file(log_file,info=info)
				assert False, info
			#Compute new gradient
			grad=prblm.get_grad(modl)
			#Compute updates for estimated Hessian inverse
			step_index=(iter-1)%self.m_steps #Modulo to handle limited memory
			# 	yn+1=gn+1-gn
			sep.Add(grad,grad0,self.grad_diff_files[step_index],scale2=-1.0)
			# 	sn+1=xn+1-xn = alpha * dmodl
			sep.Cp(dmodl,self.step_files[step_index]); sep.Scale(self.step_files[step_index],alpha)
			#	rhon+1=1/yn+1'sn+1
			denom_dot=sep.Dot_incore(self.grad_diff_files[step_index],self.step_files[step_index])
			#Checking rho
			self.check_rho(denom_dot,step_index,log_file,iter)

			#Saving current model, and gradient in case of restart
			prblm.restart.write_file(modl,"model_restart_LBFGS.H")
			prblm.restart.write_file(self.grad_diff_files[step_index],"grad_diff_restart_LBFGS_step%s.H"%(step_index))
			prblm.restart.write_file(self.step_files[step_index],"step_files_restart_LBFGS_step%s.H"%(step_index))
			#Outputting iteration results
			info = "iter = %s obj = %s residual norm = %s gradient norm= %s feval = %s"%(iter,obj1,prblm.get_rnorm(),prblm.get_gnorm(),prblm.get_fevals())
			print info
			#Writing on log file
			solv.write_log_file(log_file,info="\n"+info)
			solv.write_log_file(prblm.restart.log_file,info="step_length = %s"%(alpha))
			solv.write_log_file(prblm.restart.log_file,info="\n"+info)
			#Check if either objective function value or gradient norm is NaN
			assert not(isnan(obj1) or isnan(prblm.get_gnorm())), "Error! Either gradient norm or objective function value NaN!"
			#Saving previous objective function value
			obj0=obj1
			iter = iter + 1
			#Beware stopper is going to change the gradient/obj/res files
			if (self.stoppr.run(iter,prblm,log_file)): break

		#Writing last inverted model
		prblm.output(modl)
		#Solver has finished, no need for restart folder
		prblm.restart.clean_problem_restart()
		#Resetting inverse Hessian estimate weights
		self.rho=[0.0]*self.m_steps

		#Writing on log file
		solv.write_log_file(log_file,info="Solver will terminate\n")
		if(self.m_steps < self.stoppr.niter):
			solv.write_log_file(log_file,info="Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) algorithm log file end")
		else:
			solv.write_log_file(log_file,info="Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm log file end")
		return
