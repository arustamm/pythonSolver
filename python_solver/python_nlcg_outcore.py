#module containing object for non-linear conjugate gradient solver
import sys
import operator
import math
import sep_python as sep
import python_solver_outcore as solv
import stepper_sample as simp_stpr
import stopper_basic as simp_stop
from math import isnan


#Beta functions
#grad=new gradient, grad0=old, dir=search direction
#From A SURVEY OF NONLINEAR CONJUGATE GRADIENT METHODS

def errhandler():
	print "Beta function not recognized"
	return

def betaFR(grad,grad0,dir,log_file=None):
	"""Fletcher and Reeves method"""
	#betaFR = sum(dprod(g,g))/sum(dprod(g0,g0))
	dot_grad=sep.Dot_incore(grad,grad)
	dot_grad0=sep.Dot_incore(grad0,grad0)
	if (dot_grad0 == 0.): #Avoid division by zero
		beta = 0.
		solv.write_log_file(log_file,info="!!!Setting beta to zero since norm of previous gradient is zero!!!")
	else:
		beta = dot_grad/dot_grad0
	return beta

def betaPRP(grad,grad0,dir,log_file=None):
	"""Polak, Ribiere, Polyak method"""
	#betaPRP = sum(dprod(g,g-g0))/sum(dprod(g0,g0))
	tmp1=sep.tmp_file("tmp1betaPRP.H")
	sep.Cp(grad,tmp1)
	#g-g0
	sep.Sum(tmp1,grad0,1.0,-1.0)
	dot_num=sep.Dot_incore(grad,tmp1)
	dot_grad0=sep.Dot_incore(grad0,grad0)
	if (dot_grad0 == 0.): #Avoid division by zero
		beta = 0.
		solv.write_log_file(log_file,info="!!!Setting beta to zero since norm of previous gradient is zero!!!")
	else:
		beta = dot_num/dot_grad0
	sep.Rm(tmp1);
	return beta

def betaHS(grad,grad0,dir,log_file=None):
	"""Hestenes and Stiefel"""
	#betaHS = sum(dprod(g,g-g0))/sum(dprod(d,g-g0))
	tmp1=sep.tmp_file("tmp1betaHS.H")
	sep.Cp(grad,tmp1)
	#g-g0
	sep.Sum(tmp1,grad0,1.0,-1.0)
	dot_num=sep.Dot_incore(grad,tmp1)
	dot_denom=sep.Dot_incore(dir,tmp1)
	if (dot_denom == 0.): #Avoid division by zero
		beta = 0.
		solv.write_log_file(log_file,info="!!!Setting beta to zero since norm of denominator is zero!!!")
	else:
		beta = dot_num/dot_denom
	sep.Rm(tmp1);
	return beta

def betaCD(grad,grad0,dir,log_file=None):
	"""Conjugate Descent"""
	#betaCD = -sum(dprod(g,g))/sum(dprod(d,g0))
	dot_num=sep.Dot_incore(grad,grad)
	dot_denom=-sep.Dot_incore(dir,grad0)
	if (dot_denom == 0.): #Avoid division by zero
		beta = 0.
		solv.write_log_file(log_file,info="!!!Setting beta to zero since norm of denominator is zero!!!")
	else:
		beta = dot_num/dot_denom
	return beta

def betaLS(grad,grad0,dir,log_file=None):
	"""Liu and Storey"""
	#betaLS = -sum(dprod(g,g-g0))/sum(dprod(d,g0))
	tmp1=sep.tmp_file("tmp1betaLS.H")
	sep.Cp(grad,tmp1)
	#g-g0
	sep.Sum(tmp1,grad0,1.0,-1.0)
	dot_num=sep.Dot_incore(grad,tmp1)
	dot_denom=-sep.Dot_incore(dir,grad0)
	if (dot_denom == 0.): #Avoid division by zero
		beta = 0.
		solv.write_log_file(log_file,info="!!!Setting beta to zero since norm of denominator is zero!!!")
	else:
		beta = dot_num/dot_denom
	sep.Rm(tmp1);
	return beta


def betaDY(grad,grad0,dir,log_file=None):
	"""Dai and Yuan"""
	#betaDY = sum(dprod(g,g))/sum(dprod(d,g-g0))
	tmp1=sep.tmp_file("tmp1betaDY.H")
	sep.Cp(grad,tmp1)
	#g-g0
	sep.Sum(tmp1,grad0,1.0,-1.0)
	dot_num=sep.Dot_incore(grad,grad)
	dot_denom=sep.Dot_incore(dir,tmp1)
	if (dot_denom == 0.): #Avoid division by zero
		beta = 0.
		solv.write_log_file(log_file,info="!!!Setting beta to zero since norm of denominator is zero!!!")
	else:
		beta = dot_num/dot_denom
	sep.Rm(tmp1);
	return beta

def betaBAN(grad,grad0,dir,log_file=None):
	"""Bamigbola, Ali and Nwaeze"""
	#betaDY = sum(dprod(g,g-g0))/sum(dprod(g0,g-g0))
	tmp1=sep.tmp_file("tmp1betaBAN.H")
	sep.Cp(grad,tmp1)
	#g-g0
	sep.Sum(tmp1,grad0,1.0,-1.0)
	dot_num=sep.Dot_incore(grad,tmp1)
	dot_denom=sep.Dot_incore(grad0,tmp1)
	if (dot_denom == 0.): #Avoid division by zero
		beta = 0.
		solv.write_log_file(log_file,info="!!!Setting beta to zero since norm of denominator is zero!!!")
	else:
		beta = -dot_num/dot_denom
	sep.Rm(tmp1);
	return beta

def betaHZ(grad,grad0,dir,log_file=None):
	"""Hager and Zhang"""
	#betaN = sum(dprod(g-g0-2*sum(dprod(g-g0,g-g0))*d/sum(dprod(d,g-g0)),g))/sum(dprod(d,g-g0))
	tmp1=sep.tmp_file("tmp1betaHZ.H")
	sep.Cp(grad,tmp1)
	#g-g0
	sep.Sum(tmp1,grad0,1.0,-1.0)
	#sum(dprod(g-g0,g-g0))
	dot_diff_g_g0=sep.Dot_incore(tmp1,tmp1)
	#sum(dprod(d,g-g0))
	dot_dir_diff_g_g0=sep.Dot_incore(dir,tmp1)
	if (dot_dir_diff_g_g0 == 0.): #Avoid division by zero
		beta = 0.
		solv.write_log_file(log_file,info="!!!Setting beta to zero since norm of denominator is zero!!!")
	else:
		#g-g0-2*sum(dprod(g-g0,g-g0))*d/sum(dprod(d,g-g0))
		sep.Sum(tmp1,dir,1.0,-2.0*dot_diff_g_g0/dot_dir_diff_g_g0)
		#sum(dprod(g-g0-2*sum(dprod(g-g0,g-g0))*d/sum(dprod(d,g-g0)),g))
		dot_num=sep.Dot_incore(tmp1,grad)
		#dot_num/sum(dprod(d,g-g0))
		beta = dot_num/dot_dir_diff_g_g0
	sep.Rm(tmp1);
	return beta

def betaST(grad,grad0,dir,log_file=None):
	"""Steepest descent"""
	beta = 0.
	return beta



class nlcg_solver(solv.solver):
	#Dictionary containing beta functions
	betafunction = {
	"FR" : betaFR,
	"PRP": betaPRP,
	"HS" : betaHS,
	"CD" : betaCD,
	"LS" : betaLS,
	"DY" : betaDY,
	"BAN": betaBAN,
	"HZ" : betaHZ,
	"steepest descent" : betaST
	}

	def __init__(self,stoppr=simp_stop.stopper_basic(),beta_type="FR"):
		"""Constructor for non-linear conjugate gradient solver"""
		#Defining stepper and stopper objects
		self.stoppr=stoppr
		self.beta_type=beta_type
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
					print "	Temporary file not removed for debugging in NON-LINEAR CG: %s"%(ifile)
		return


	def run(self,prblm,stpr=simp_stpr.stepper_sample(),log_file=None):
		"""Running the solver"""
		#Writing first line in log file if present
		if(not prblm.restart.restarting):
			if(self.beta_type=="steepest descent"):
				solv.write_log_file(log_file,info="STEEPEST-DESCENT SOLVER log file\n")
			else:
				solv.write_log_file(log_file,info="NON-LINEAR CONJUGATE GRADIENT SOLVER log file")
				if(not prblm.restart.restarting): solv.write_log_file(log_file,info="Conjugate method used: %s \n"%(self.beta_type))
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
		
		#Defining beta function computation
		beta_fun=nlcg_solver.betafunction.get(self.beta_type,errhandler)
		#Set internal temporary files
		grad0=sep.tmp_file("grad0_nlcg.H");	self.files_to_clean.append(grad0)
		dmodl=sep.tmp_file("dmodel_nlcg.H");self.files_to_clean.append(dmodl)
		modl=sep.tmp_file("model_nlcg.H");	self.files_to_clean.append(modl)
		beta = 0.0
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
			solv.write_log_file(prblm.restart.log_file,info="NON-LINEAR CONJUGATE GRADIENT SOLVER restart log file\n")
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
				prblm.restart.copy_file_from_restart("model_restart_NLCG.H",modl)
				prblm.restart.copy_file_from_restart("dmodel_restart_NLCG.H",dmodl)
				prblm.restart.copy_file_from_restart("grad0_restart_NLCG.H",grad0)
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

			if(iter > 1):
				beta = beta_fun(grad, grad0, dmodl, log_file)
				if(beta < 0.):
					solv.write_log_file(log_file,"!!!Beta negative setting to zero: beta value=%s!!!"%(beta))
					beta = 0.
			#Writing on log file
			if(self.beta_type!="steepest descent"):
				solv.write_log_file(log_file,info="beta coefficient: %s"%(beta))
			#dmodl = beta*dmodl - grad
			sep.Sum(dmodl,grad,beta,-1.0)
			#grad0=grad
			sep.Cp(grad,grad0)
			#Calling line search (for some reason cannot be put in the solver_obj)
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
			#Saving current model, previous search direction and gradient in case of restart
			prblm.restart.write_file(modl,"model_restart_NLCG.H")
			prblm.restart.write_file(dmodl,"dmodel_restart_NLCG.H")
			prblm.restart.write_file(grad0,"grad0_restart_NLCG.H")
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
		#Writing on log file
		solv.write_log_file(log_file,info="Solver will terminate\n")
		if(self.beta_type=="steepest descent"):
			solv.write_log_file(log_file,info="STEEPEST-DESCENT SOLVER log file end")
		else:
			solv.write_log_file(log_file,info="NON-LINEAR CONJUGATE GRADIENT SOLVER log file end")
		return
