#module containing objects for linear conjugate gradient solver for symmetric matrix (gradient is equal to residual (1/2m'Am -b'm) see Aster et al. (2012))
import sys
import operator
import math
import sep_python as sep
import python_solver_outcore as solv
import stopper_symmetric as simp_stop
from math import isnan

class lcg_symmetric_solver(solv.solver):

	def __init__(self,stoppr=simp_stop.stopper_symmetric(),stpr=solv.stepper()):
		"""Constructor for linear conjugate gradient solver"""
		#Defining stepper and stopper objects
		self.stoppr=stoppr
		self.stpr=stpr
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
					print "	Temporary file not removed for debugging in LINEAR SYMMETRIC SOLVER: %s"%(ifile)
		return


	def run(self,prblm,log_file=None,verbose=True):
		"""Running the solver"""
		#Writing first line in log file if present
		if(not prblm.restart.restarting):
			solv.write_log_file(log_file,info="LINEAR CONJUGATE GRADIENT SOLVER FOR SYMMETRIC MATRIX log file")
			solv.write_log_file(log_file,info="Restarting folder: %s\n"%(prblm.restart.restart_folder))
			if(hasattr(self.stpr, 'maxval') and hasattr(self.stpr, 'minval')):
				info = ""
				if(self.stpr.maxval != None): 
					info += "Maximum allowed value for inverted model: %s\n"%(self.stpr.maxval)
				if(self.stpr.minval != None): 
					info += "Minimum allowed value for inverted model: %s\n"%(self.stpr.minval)
				if(info != ""):
					if(verbose): print info
					solv.write_log_file(log_file,info)
			
		#Set internal model files
		dmodl=sep.tmp_file("dmodel_lcg.H");	self.files_to_clean.append(dmodl)
		modl=sep.tmp_file("model_lcg.H");	self.files_to_clean.append(modl)
		iter = 1
		success = True
		#Set internal model file
		modl_prblm=prblm.get_model()
		sep.Cp(modl_prblm,modl)
		#Search direction
		sep.Cp(modl_prblm,dmodl)
		sep.Zero(dmodl)
		beta = 0.0 #Steepest descent
		#Necessary for relative matching
		if(prblm.data != None):
			data_norm = sep.Norm_incore(prblm.data)

		if(not prblm.restart.restarting):
			#Creating restart folder
			prblm.restart.create_restart_folder()
			solv.write_log_file(prblm.restart.log_file,info="LINEAR CONJUGATE GRADIENT SOLVER FOR SYMMETRIC MATRIX restart log file\n")

		#Iteration loop
		while True:
			#Restart inversion from previous run
			if(prblm.restart.restarting):
				#Restarting the problem and getting iteration number
				iter=prblm.restart.set_restart(prblm,log_file)+1
				info="Restarting inversion from previous run from: %s"%(prblm.restart.restart_folder)
				solv.write_log_file(log_file,info)
				if(verbose): print info
				if(iter>=2):
					prblm.restart.copy_file_from_restart("model_restart_LCG_sym.H",modl)
					prblm.restart.copy_file_from_restart("dmodel_restart_LCG_sym.H",dmodl)
					beta=float(prblm.restart.get_info_log("beta_step"))
					obj_old = float(prblm.restart.get_info_log("obj",iter_num=iter-2))

			res=prblm.get_res(modl) 		#Compute residuals
			obj=prblm.get_obj(modl) 		#Compute objective function value and put it in a file
			obj0=sep.Get_value(obj) 		#Read the objective function value in the program
			if(iter==1):
				info = "iter = %s obj = %s residual norm = %s feval = %s"%(iter-1,obj0,prblm.get_rnorm(),prblm.get_fevals())
				if(verbose): print info
				#Writing on log file
				solv.write_log_file(log_file,info=info)
				solv.write_log_file(prblm.restart.log_file,info)
				#Writing relative data matching
				if(prblm.data != None):
					info = "relative data matching (i.e., 1-|Am-b|/|b|): %s"%((1.0-prblm.get_rnorm()/data_norm)*100.0)+"%"
					solv.write_log_file(log_file,info=info)
				#Check if objective function value is NaN
				assert not(isnan(obj0)), "Error! Objective function value NaN!"
			#dmodl = -1.0 * res + beta * dmodl
			sep.Sum(dmodl,res,beta,-1.0) #update search direction
			ddmodl=prblm.get_dres(modl,dmodl) #Project search direction in the "data space" (same as model space)
			dot_dmodl_ddmodl=sep.Dot_incore(dmodl,ddmodl)

			#Outputing files if problem was not restarted
			if(not prblm.restart.restarting):
				prblm.output(modl)
			prblm.restart.restarting=False

			dot_res=sep.Dot_incore(res,res)
			if(dot_res == 0.):
				print "Residuals/Gradient vanishes identically"
				break
			elif(dot_dmodl_ddmodl==0.0):
				success = False
				info="Residuals/Gradient orthogonal to span of linear operator, will terminate solver"
				#Writing on log file
				solv.write_log_file(log_file,info)
			if(not success):
				info = "Stepper couldn't find a proper step size, will terminate solver"
				if(verbose): print info
				solv.write_log_file(log_file,info=info)
				break
			alpha = dot_res/dot_dmodl_ddmodl
			solv.write_log_file(log_file,info="Alpha step length: %s"%(alpha))
			#res  = res + alpha * dres =  res + alpha * A * dmodl
			sep.Sum(res,ddmodl,1.0,alpha) #update residuals
			#modl = modl + alpha * dmodl
			sep.Sum(modl,dmodl,1.0,alpha) #Update model
			#Hard bounds if defined
			clipped=self.stpr.clipping(modl,log_file)

			#Setting the model and residuals to avoid residual twice computation
			prblm.set_model(modl)
			#Recomputing residuals if model was clipped
			if(not clipped):
				prblm.set_residual(res)
			obj=prblm.get_obj(modl) 		#Compute objective function value and put it in a file
			obj1=sep.Get_value(obj) 		#Read the objective function value in the program
			#Checking monotonic behavior of objective function
			if(iter==1):
				obj_old = obj0 #Saving objective function at iter-1
			else:
				#If not monotonic stop the inversion
				if not((obj_old < obj0 < obj1) or (obj_old > obj0 > obj1)):
					info = "Objective function variation not monotonic, will terminate solver: obj_old=%s obj_cur=%s obj_new=%s"%(obj_old,obj0,obj1)
					if(verbose): print info
					#Writing on log file
					solv.write_log_file(log_file,info=info)
					#Stepping back to the previous solution
					sep.Sum(modl,dmodl,1.0,-1.0)
					break
				obj_old = obj0 #Saving objective function at iter-1

			#Saving current model and previous search direction in case of restart
			prblm.restart.write_file(modl,"model_restart_LCG_sym.H")
			prblm.restart.write_file(dmodl,"dmodel_restart_LCG_sym.H")
			info = "iter = %s obj = %s residual norm = %s feval = %s"%(iter,obj1,prblm.get_rnorm(),prblm.get_fevals())
			if(verbose): print info
			#Writing on log file
			solv.write_log_file(log_file,info="\n"+info)
			solv.write_log_file(prblm.restart.log_file,info="\n"+info)
			#Check if objective function value is NaN
			assert not(isnan(obj1)), "Error! Objective function value NaN!"
			#Writing relative data matching
			if(prblm.data != None):
				info = "relative data matching (i.e., 1-|Am-b|/|b|): %s"%((1.0-prblm.get_rnorm()/data_norm)*100.0)+"%"
				solv.write_log_file(log_file,info=info)

			#New residual norm
			dot_res_new=sep.Dot_incore(res,res)
			beta = dot_res_new/dot_res
			iter = iter + 1
			if (self.stoppr.run(iter,prblm,log_file,verbose)): break
			solv.write_log_file(log_file,info="Beta step length: %s"%(beta))
			solv.write_log_file(prblm.restart.log_file,info="beta_step = %s"%(beta))

		#Writing last model
		prblm.output(modl)
		#Solver has finished, no need for restart folder
		prblm.restart.clean_problem_restart()
		#Writing on log file
		solv.write_log_file(log_file,info="Solver will terminate\n")
		solv.write_log_file(log_file,info="LINEAR CONJUGATE GRADIENT SOLVER FOR SYMMETRIC MATRIX log file end")

		return
