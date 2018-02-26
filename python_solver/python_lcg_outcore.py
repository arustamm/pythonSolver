#module containing objects for linear conjugate gradient solver
import sys
import operator
import math
import sep_python as sep
import python_solver_outcore as solv
import stopper_basic as simp_stop
from math import isnan

class lcg_solver(solv.solver):

	def __init__(self,stoppr=simp_stop.stopper_basic(),stpr=solv.stepper(),steepest=False):
		"""Constructor for linear conjugate gradient solver"""
		#Defining stepper and stopper objects
		self.stoppr=stoppr
		self.stpr=stpr
		#Whether to run steepest descent or not
		self.steepest=steepest
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
					print "	Temporary file not removed for debugging in LINEAR CG: %s"%(ifile)
		return


	def run(self,prblm,log_file=None,verbose=True):
		"""Running the solver"""
		#Writing first line in log file if present
		if(not prblm.restart.restarting):
			if(self.steepest):
				solv.write_log_file(log_file,info="LINEAR STEEPEST-DESCENT SOLVER log file")
			else:
				solv.write_log_file(log_file,info="LINEAR CONJUGATE GRADIENT SOLVER log file")
			solv.write_log_file(log_file,info="Restarting folder: %s\n"%(prblm.restart.restart_folder))
			if(hasattr(prblm, 'epsilon')):
				info = "Regularized inversion with epsilon value of: %s"%(prblm.epsilon)
				if(verbose): print info
				solv.write_log_file(log_file,info)
				if(prblm.op_reg_fwd == 'identity' and prblm.op_reg_adj == 'identity'):
					info = "Regularization operator used: IDENTIY OPERATOR\n"
				else:
					info = "Regularization operator used: USER DEFINED\n"
				print info
				solv.write_log_file(log_file,info)
			if(hasattr(self.stpr, 'maxval') and hasattr(self.stpr, 'minval')):
				info = ""
				if(self.stpr.maxval != None): 
					info += "Maximum allowed value for inverted model: %s\n"%(self.stpr.maxval)
				if(self.stpr.minval != None): 
					info += "Minimum allowed value for inverted model: %s\n"%(self.stpr.minval)
				if(info != ""):
					print info
					solv.write_log_file(log_file,info)
		#Set internal delta residual and model files
		if(isinstance(prblm.res,list)):
			dres=[]
			for ifile in range(len(prblm.res)):
				tmp_file=sep.tmp_file("dres_lcg%s.H"%(ifile)); self.files_to_clean.append(tmp_file)
				dres.append(tmp_file)
		else:
			dres=sep.tmp_file("dres_lcg.H");self.files_to_clean.append(dres)

		dmodl=sep.tmp_file("dmodel_lcg.H");	self.files_to_clean.append(dmodl)
		modl=sep.tmp_file("model_lcg.H");	self.files_to_clean.append(modl)
		#other temp variables
		iter = 1
		success = True
		#Set internal model file
		modl_prblm=prblm.get_model()
		sep.Cp(modl_prblm,modl)
		#Search direction
		sep.Cp(modl_prblm,dmodl)
		sep.Zero(dmodl)

		if(not prblm.restart.restarting):
			#Creating restart folder
			prblm.restart.create_restart_folder()
			solv.write_log_file(prblm.restart.log_file,info="LINEAR CONJUGATE GRADIENT SOLVER restart log file\n")

		#Iteration loop
		while True:
			#Restart inversion from previous run
			if(prblm.restart.restarting):
				#Restarting the problem and getting iteration number
				iter=prblm.restart.set_restart(prblm,log_file)+1
				info="Restarting inversion from previous run from: %s"%(prblm.restart.restart_folder)
				solv.write_log_file(log_file,info)
				if(verbose): print info
				prblm.restart.copy_file_from_restart("model_restart_LCG.H",modl)
				prblm.restart.copy_file_from_restart("dmodel_restart_LCG.H",dmodl)
				#Recomputing delta residuals
				dres_prblm=prblm.get_dres(modl,dmodl)
				sep.Cp(dres_prblm,dres)
				prblm.initial_obj_value = float(prblm.restart.get_info_log("obj",iter_num=0))

			obj=prblm.get_obj(modl) 		#Compute objective function value and put it in a file
			obj0=sep.Get_value(obj) 		#Read the objective function value in the program
			res=prblm.get_res(modl) 		#Compute residuals
			grad=prblm.get_grad(modl) 		#Compute the gradient
			if(iter==1):
				prblm.initial_obj_value=obj0 #For relative objective function value
				info = "iter = %s obj = %s residual norm = %s gradient norm= %s feval = %s"%(iter-1,obj0,prblm.get_rnorm(),prblm.get_gnorm(),prblm.get_fevals())
				if(verbose): print info
				#Writing on log file
				solv.write_log_file(log_file,info=info)
				solv.write_log_file(prblm.restart.log_file,info="\n"+info)
				#Check if either objective function value or gradient norm is NaN
				assert not(isnan(obj0) or isnan(prblm.get_gnorm())), "Error! Either gradient norm or objective function value NaN!"
				#Set internal delta residual files
				sep.Cp(res,dres)
				sep.Zero(dres)
			if(prblm.get_gnorm() == 0.):
				print "Gradient vanishes identically"
				break
			gradd=prblm.get_dres(modl,grad)	#Project gradient in the data space

			#Outputing files if problem was not restarted
			if(not prblm.restart.restarting):
				prblm.output(modl)
			prblm.restart.restarting=False

			if(iter==1 or self.steepest):
				beta = 0.0 #Steepest descent
				dot_gradd=sep.Dot_incore(gradd,gradd)
				if(dot_gradd==0.0):
					success = False
					info="Gradient orthogonal to span of linear operator, will terminate solver"
					#Writing on log file
					solv.write_log_file(log_file,info)
				else:
					dot_gradd_res=sep.Dot_incore(gradd,res)
					alpha = - dot_gradd_res/dot_gradd
					#Writing on log file
					if(self.steepest):
						solv.write_log_file(log_file,info="Steppest-descent step length: %s"%(alpha))
					else:
						solv.write_log_file(log_file,info="First steppest-descent step length: %s"%(alpha))
			else:
				dot_gradd=sep.Dot_incore(gradd,gradd)
				dot_dres=sep.Dot_incore(dres,dres)
				dot_gradd_dres=sep.Dot_incore(gradd,dres)
				if((dot_gradd==0.)or(dot_dres==0.)):
					success = False
				else:
					determ = dot_gradd * dot_dres - dot_gradd_dres * dot_gradd_dres
					dot_gradd_res=sep.Dot_incore(gradd,res)
					dot_dres_res=sep.Dot_incore(dres,res)
					alpha = -(dot_dres*dot_gradd_res - dot_gradd_dres*dot_dres_res) /determ
					beta = (dot_gradd_dres*dot_gradd_res - dot_gradd*dot_dres_res) /determ
					#Writing on log file
					solv.write_log_file(log_file,info="Conjugate alpha,beta: %s,%s"%(alpha,beta))
			if(not success):
				info = "Stepper couldn't find a proper step size, will terminate solver"
				if(verbose): print info
				solv.write_log_file(log_file,info=info)
				break

			#dmodl = alpha * grad + beta * dmodl
			sep.Sum(dmodl,grad,beta,alpha) #update search direction
			#dres  = alpha * gradd + beta * dres
			sep.Sum(dres,gradd,beta,alpha) #update residual step
			#modl = modl + dmodl
			sep.Sum(modl,dmodl) #Update model
			#res = res + dres
			sep.Sum(res,dres) #Update residuals

			#Hard bounds if defined
			clipped=self.stpr.clipping(modl,log_file)

			#Setting the model and residuals to avoid residual twice computation
			prblm.set_model(modl)
			#Recomputing residuals if model was clipped
			if(not clipped):
				prblm.set_residual(res)
			obj=prblm.get_obj(modl)
			obj1=sep.Get_value(obj)
			if(obj1 >= obj0):
				info = "Objective function didn't reduce, will terminate solver: obj_new=%s obj_current=%s"%(obj1,obj0)
				if(verbose): print info
				#Writing on log file
				solv.write_log_file(log_file,info=info)
				#Stepping back to the previous solution
				sep.Sum(modl,dmodl,1.0,-1.0)
				break

			#Saving current model and previous search direction in case of restart
			prblm.restart.write_file(modl,"model_restart_LCG.H")
			prblm.restart.write_file(dmodl,"dmodel_restart_LCG.H")
			#iteration info
			info = "iter = %s obj = %s residual norm = %s gradient norm= %s feval = %s"%(iter,obj1,prblm.get_rnorm(),prblm.get_gnorm(),prblm.get_fevals())
			if(verbose): print info
			#Writing on log file
			solv.write_log_file(log_file,info="\n"+info)
			solv.write_log_file(prblm.restart.log_file,info="\n"+info)
			#Check if either objective function value or gradient norm is NaN
			assert not(isnan(obj1) or isnan(prblm.get_gnorm())), "Error! Either gradient norm or objective function value NaN!"
			iter = iter + 1
			if (self.stoppr.run(iter,prblm,log_file,verbose)): break


		#Writing last model
		prblm.output(modl)
		#Solver has finished, no need for restart folder
		prblm.restart.clean_problem_restart()
		#Writing on log file
		solv.write_log_file(log_file,info="Solver will terminate\n")
		if(self.steepest):
			solv.write_log_file(log_file,info="LINEAR STEEPEST-DESCENT SOLVER log file end")
		else:
			solv.write_log_file(log_file,info="LINEAR CONJUGATE GRADIENT SOLVER log file end")

		return
