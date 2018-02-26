#module containing object for truncated Newton solver
import sys, os
import operator
import math
import sep_python as sep
import python_solver_outcore as solv
import stepper_sample as simp_stpr
import stopper_basic as simp_stop
import operator_obj as operator
from math import isnan


class trunc_newton_solver(solv.solver):

	def __init__(self,lin_solver,lin_problem,wrk_dir,niter_max,suffix,niter_min=None,warm_start=False,stoppr=simp_stop.stopper_basic()):
		"""Constructor for truncated Newton solver"""
		#Defining stopper, linear problem solver, and linear problem (Newton system) objects
		self.stoppr=stoppr #Stopper for non-linear problem
		self.lin_solver=lin_solver
		self.lin_problem=lin_problem
		#Setting maximum and minimum number of iterations
		self.niter_max=niter_max
		self.niter_min=niter_max
		if(niter_min!=None):
			assert (niter_min<=niter_max), "Error in truncated Newton solver! niter_min (currently %s) must be smaller or equal than niter_max (currently %s)."%(niter_min,niter_max)
			self.niter_min=niter_min
		#Warm starts requested?
		self.warm_start=warm_start
		#Setting input_m0.H for Hessian evaluation (Redundant test)
		if(self.lin_problem.op_fwd.input_m0==None):
			self.lin_problem.op_fwd.input_m0="input_m0.H"
		if(self.lin_problem.op_adj.input_m0==None):
			self.lin_problem.op_adj.input_m0="input_m0.H"
		self.wrk_dir=wrk_dir 								#Necessary to put truncated Newton step results
		if (self.wrk_dir[-1]!="/"): self.wrk_dir+="/"		#adding slash to directory name if necessary
		self.suffix=suffix									#Suffix for linear inversion files
		self.files_to_clean=[]
		self.removefiles=True
		return

	def __del__(self):
		"""Overriding default destructor"""
		import sep_python as sep
		if hasattr(self, 'removefiles'):
			if (self.removefiles):
				sep.Rm(self.files_to_clean)
			else:
				for ifile in self.files_to_clean:
					print "	Temporary file not removed for debugging in truncated Newton: %s"%(ifile)
		return

	def set_newton_system(self,prblm,initial_dmodl,gradient,iter):
		"""Function to set the Newton linear problem"""
		self.wrk_dir_Newton="Newton_inversion_iter%s/"%(iter)
		sep.RunShellCmd("mkdir -p %s"%(self.wrk_dir+self.wrk_dir_Newton))

		#Resetting the inverse problem note: reset_problem(initial_model,data,inverted_model)
		inverted_dmodl=self.wrk_dir+self.wrk_dir_Newton+"inv_model"+self.suffix+"_tnewton_iter%s"%(iter)+".H"
		self.lin_problem.reset_problem(initial_dmodl,gradient,inverted_dmodl)

		#Obtaining movie file names and changing them as function of iteration
		obj_movie_name=""
		if(prblm.obj_movie!=""):
			obj_movie_name=self.wrk_dir+self.wrk_dir_Newton+"obj"+self.suffix+"_tnewton_iter%s"%(iter)+".H"
		model_movie_name=""
		if(prblm.model_movie!=""):
			model_movie_name=self.wrk_dir+self.wrk_dir_Newton+"model"+self.suffix+"_tnewton_iter%s"%(iter)+".H"
		grad_movie_name=""
		if(prblm.grad_movie!=""):
			grad_movie_name=self.wrk_dir+self.wrk_dir_Newton+"gradient"+self.suffix+"_tnewton_iter%s"%(iter)+".H"
		res_movie_name=""
		if(prblm.res_movie!=""):
			res_movie_name=self.wrk_dir+self.wrk_dir_Newton+"residual"+self.suffix+"_tnewton_iter%s"%(iter)+".H"
		#Resetting movie files
		self.lin_problem.set_movies(obj_movie_name,model_movie_name,grad_movie_name,res_movie_name)
		return

	def run(self,prblm,stpr=simp_stpr.stepper_sample(),log_file=None):
		"""Running the solver"""
		import glob
		#Writing first line in log file if present
		if(not prblm.restart.restarting):
			solv.write_log_file(log_file,info="TRUNCATED NEWTON SOLVER log file")
			solv.write_log_file(log_file,info="Restarting folder: %s\n"%(prblm.restart.restart_folder))
			if(hasattr(prblm, 'epsilon')):#self.op_reg_fwd=self.op_reg_adj='identity'
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
		dmodl=sep.tmp_file("dmodel_trunc_newton.H");	 self.files_to_clean.append(dmodl)
		modl=sep.tmp_file("model_trunc_newton.H");	 	 self.files_to_clean.append(modl)
		hess_app=sep.tmp_file("hess_app_trunc_newton.H");self.files_to_clean.append(hess_app)
		#Set internal search direction file
		modl_prblm=prblm.get_model()
		sep.Cp(modl_prblm,dmodl)
		sep.Zero(dmodl)
		#Set internal model file
		sep.Cp(modl_prblm,modl)
		#Set internal last hessian application for restarting from previous run
		sep.Cp(modl_prblm,hess_app)
		sep.Zero(hess_app)

		#Starting inversion
		iter = 1
		if(not prblm.restart.restarting):
			obj=prblm.get_obj(modl)
			obj0=sep.Get_value(obj) #For redundant test on stepper
			prblm.initial_obj_value=obj0 #For relative objective function value
			info = "iter = %s obj = %s residual norm = %s gradient norm= %s feval = %s"%(iter-1,obj0,prblm.get_rnorm(),prblm.get_gnorm(),prblm.get_fevals())
			print info
			#Writing on log file
			solv.write_log_file(log_file,info=info)
			#Creating restart folder
			prblm.restart.create_restart_folder()
			solv.write_log_file(prblm.restart.log_file,info="TRUNCATED NEWTON SOLVER restart log file\n")
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
				prblm.restart.copy_file_from_restart("model_trunc_newton.H",modl)
				prblm.restart.copy_file_from_restart("dmodel_trunc_newton.H",dmodl)
				prblm.restart.copy_file_from_restart("hess_app_trunc_newton.H",hess_app)
				#If after iteration 1, set number of iterations from internal definition
				if(iter>1):
					self.lin_solver.stoppr.niter=int(prblm.restart.get_info_log("niter_lin"))
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

			#Setting Newton linear problem
			self.lin_solver.stoppr.reset_stopper() 						#Resetting stopper inside linear solver
			self.set_newton_system(prblm,dmodl,grad,iter)				#Resetting linear problem
			#Setting current Hessian operator H(modl) and residuals if any for Hessian
			if(self.lin_problem.op_fwd.input_aux==[]):
				self.lin_problem.op_fwd.set_input_output(input_m0=modl)
			else:
				self.lin_problem.op_fwd.set_input_output(input_m0=modl,input_aux=[prblm.get_res(modl)])
			if(self.lin_problem.op_adj.input_aux==[]):
				self.lin_problem.op_adj.set_input_output(input_m0=modl)
			else:
				self.lin_problem.op_adj.set_input_output(input_m0=modl,input_aux=[prblm.get_res(modl)])
			#Solving iteratively the linear system
			# H(modl)inv_dmodl = grad
			lin_inv_log_file=self.wrk_dir+self.wrk_dir_Newton+"truncated_Newton_inv_iter%s.log"%(iter) #Linear inversion log file
			#Check for restarting of linear problem
			if(prblm.restart.restarting):
				restart_lin_dir=self.lin_problem.restart.get_restart_folder(lin_inv_log_file)
				#checking if model-restart file is present (glob returns list of files matching the string)
				if glob.glob(restart_lin_dir+"model*.H"):
					#Restart folder of linear problem exists
					self.lin_problem.restart.restarting=True

			#Outputing files if problem was not restarted
			if(not prblm.restart.restarting):
				prblm.output(modl)
			prblm.restart.restarting=False

			#Running the linear solver
			self.lin_solver.run(self.lin_problem,lin_inv_log_file,verbose=False)
			#Adding fevals of linear problem to non-linear ones
			prblm.fevals+=2*self.lin_problem.fevals #Hessian twice as expensive as a linerized operator (i.e., L'L)
			# dmodl = - inv_dmodl => H(modl)inv_dmodl = grad
			sep.Sum(dmodl,self.lin_problem.inverted_model,scale1=0.0,scale2=-1.0)

			#Line search with search direction from truncated Newton inversion
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
			info = "iter = %s obj = %s residual norm = %s gradient norm= %s feval = %s"%(iter,obj1,prblm.get_rnorm(),prblm.get_gnorm(),prblm.get_fevals())
			print info
			#Writing on log file
			solv.write_log_file(log_file,info="\n"+info)
			solv.write_log_file(prblm.restart.log_file,info="\n"+info)
			#Check if either objective function value or gradient norm is NaN
			assert not(isnan(obj1) or isnan(prblm.get_gnorm())), "Error! Either gradient norm or objective function value NaN!"
			#Saving previous objective function value
			obj0=obj1

			#Checking whether to reset starting dmodl or not for next inversion
			check_init_num=sep.Dot_incore(dmodl,grad)
			if(check_init_num<0.0 and self.warm_start): 				#Checking if previous step direction is still a descent one (i.e., dmodl'*grad<0)
				residual_lin=self.lin_problem.get_res(dmodl) 			#Getting the residuals for computing Hessian application
				sep.Add(residual_lin,self.lin_problem.data,hess_app)	#H dm = res + d
				check_init_denom=sep.Dot_incore(dmodl,hess_app)     	#dm' H dm
				if(check_init_denom>0.):
					#Start inversion from previous dmodel and running minimum number of iterations
					sep.Scale(dmodl,-check_init_num/check_init_denom)
					self.lin_solver.stoppr.niter=self.niter_min
					info = "Using previous search direction as initial guess of linear Newton system: niter_lin = %s"%(self.lin_solver.stoppr.niter)
					solv.write_log_file(log_file,info=info)
				else:
					#Start inversion from zero dmodel and running maximum number of iterations
					sep.Zero(dmodl)
					self.lin_solver.stoppr.niter=self.niter_max
					info = "Starting linear Newton system from zero model since denominator smaller than zero: niter_lin = %s"%(self.lin_solver.stoppr.niter)
					solv.write_log_file(log_file,info=info)
			else:
				#Start inversion from zero dmodel and running maximum number of iterations
				sep.Zero(dmodl)
				self.lin_solver.stoppr.niter=self.niter_max
				if(not self.warm_start):
					info = "Starting linear Newton system from zero model since warm starts were not requested: niter_lin = %s"%(self.lin_solver.stoppr.niter)
				else:
					info = "Starting linear Newton system from zero model since previous search is not along the same direction of new gradient: niter_lin = %s"%(self.lin_solver.stoppr.niter)
				solv.write_log_file(log_file,info=info)

			#Saving current model, previous search direction and gradient in case of restart
			prblm.restart.write_file(dmodl,"dmodel_trunc_newton.H")
			prblm.restart.write_file(modl,"model_trunc_newton.H")
			prblm.restart.write_file(hess_app,"hess_app_trunc_newton.H")
			#Writing on log file
			solv.write_log_file(prblm.restart.log_file,info="niter_lin = %s"%(self.lin_solver.stoppr.niter)) #Saving number of linear iterations for next run
			solv.write_log_file(prblm.restart.log_file,info="step_length = %s"%(alpha))

			iter = iter + 1
			#Beware stopper is going to change the gradient/obj/res files
			if (self.stoppr.run(iter,prblm,log_file)): break

		#Writing last inverted model
		prblm.output(modl)
		#Solver has finished, no need for restart folder
		prblm.restart.clean_problem_restart()
		#Writing on log file
		solv.write_log_file(log_file,info="Solver will terminate\n")
		solv.write_log_file(log_file,info="TRUNCATED NEWTON SOLVER log file end")
		return
