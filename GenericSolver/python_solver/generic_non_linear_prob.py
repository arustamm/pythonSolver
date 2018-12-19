#!/usr/bin/env python
#Generic non-linear problem solver with non-linear CG (L2-norm problem: 1/2*|f(m)-d|^2)
import python_nlcg_outcore as nlcg
import python_lbfgs_outcore as lbfgs
import python_trunc_newton_outcore as truncnewton
import python_solver_outcore as solv
import python_lcg_outcore as lcg
import python_lcg_symmetric_outcore as lcg_sym
import sep_python as sep # import sep functions
import operator_obj as op
import stopper_basic as stop
import stopper_symmetric as stop_sym
import stepper_parab as parab_step
import stepper_sample as sample_step
import stepper_linear as linear_step
import non_linear_reg_prob as non_linear_reg_prob_obj
from generic_linear_prob import lin_prob
import copy
import sys


class nl_prob(solv.problem):
	"""General non-linear problem object"""
	def set_prob(self,fwd_nl_cmd_file,fwd_cmd_file,adj_cmd_file):
		#In this problem we assume that the template series of commands takes one input file (input.H) and spits out one output file (output.H)
		self.op_nl_fwd=op.Operator("nonlinear operator forward",fwd_nl_cmd_file,"input.H","output.H")
		self.op_fwd=op.Operator("operator forward",fwd_cmd_file,"input.H","output.H","input_m0.H")
		self.op_adj=op.Operator("operator adjoint",adj_cmd_file,"input.H","output.H","input_m0.H")
		return

	# define function that computes objective function value
	def objf(self,res):
		"""0.5 norm squared of residual"""
		sep.Dot(res,res,self.obj)
		cmd="Solver_ops file1=%s scale1_r=0.5 op=scale"%(self.obj)
		sep.RunShellCmd(cmd)
		obj=self.obj
		return obj

	# define function that computes residuals
	def resf(self,model):
		"""f(m) - d"""
		self.op_nl_fwd.set_input_output(model,self.res)
		stat=self.op_nl_fwd.run() #Apply fwd non-linear modeling operator
		if(stat!=0): assert False, "problem running forward operator to compute the residual"
		sep.Sum(self.res,self.data,1.0,-1.0)
		res=self.res
		return res

	def dresf(self,model,dmodel):
		"""F dm = dr"""
		#Apply a forward modeling
		self.op_fwd.set_input_output(dmodel,self.dres,model)
		stat=self.op_fwd.run() #Apply linearized fwd modeling operator
		if(stat!=0): assert False, "problem running linearized forward operator"
		dres = self.dres
		return dres

	def gradf(self,model,residual):
		"""F'r = g"""
		#Apply an adjoint modeling
		self.op_adj.set_input_output(residual,self.grad,model)
		stat=self.op_adj.run() #Apply linearized adj modeling operator
		if(stat!=0): assert False, "problem running adjoint operator"
		grad=self.grad
		return grad

################################################################################
################################ MAIN ##########################################
################################################################################

if __name__ == '__main__':
################################################################################
###################### 	  GENERIC SOLVER FOR NON-LINEAR PROBLEM  ###############
################################################################################
	#Parsing command line
	if(len(sys.argv) == 1):
		print "NAME"
		print "	generic_non_linear_prob - Solves a L2-norm non-linear problem \n"
		print "SYNOPSIS"
		print "	generic_non_linear_prob.py fwd_nl_cmd_file=fwd_cmds_nl.txt fwd_cmd_file=fwd_cmds.txt adj_cmd_file=adj_cmds.txt data=data.H init_model=init_model.H inv_model=inv_model.H suffix=problem1 iteration_movies=obj,model,gradient,residual niter=10 dotprod=0 \n"
		print "DESCRIPTION"
		print "	Out-of-core CG for solving L2-norm residual difference for non-linear problem (1/2*|f(m)-d|^2) with different non-linear solvers"
		print "INPUT PARAMETERS"
		print "	fwd_nl_cmd_file= - char"
		print "		[no default]: Text file containing commands to be run to apply non-linear forward\n"
		print "	fwd_cmd_file= - char"
		print "		[no default]: Text file containing commands to be run to apply linearized forward\n"
		print "	adj_cmd_file= - char"
		print "		[no default]: Text file containing commands to be run to apply linearized adjoint\n"
		print "	data= - char"
		print "		[no default]: File containing observed data\n"
		print "	init_model= - char"
		print "		[no default]: File containing initial model (must be provided even if zero)\n"
		print "	inv_model= - char"
		print "		[no default]: Name of the inverted model (NOTE: it is overwritten at each iteration)"
		print "		              If only a name is provided, the file will be written in wrk_dir\n"
		print "	suffix= - char"
		print "		[no default]: suffix to append to the iteration movies if requested (e.g., suffix=_prob1 => obj_prob1.H)\n"
		print "	iteration_movies= - char"
		print "		[none]: which movie to produce during inversion (e.g., iteration_movies=obj,model,gradient,residual)"
		print "		        NOTE: for regularized problems the solver creates: obj_suffix.H=total obj_suffix1.H=data-component obj_suffix2.H=model-component objective-function values"
		print "		              and res_suffix1.H=data residuals res_suffix2.H=model residuals\n"
		print "	log_file= - char"
		print "		[inversion_log.txt]: Name of log file of inversion run (Note: erases existing file)\n"
		print "	wrk_dir= - char"
		print "		[./]: Name of directory in which inversion files are placed (if not existing it will be created)\n"
		print "	Stopping Criteria:"
		print "		niter= - int"
		print "			[no default]: Number of iterations to run\n"
		print "		maxfevals= - int"
		print "			[none]: Maximum number of function evaluations\n"
		print "		maxhours= - int"
		print "			[none]: Maxium total running time in hours\n"
		print "		tolr= - float"
		print "			[1.0e-18]: Tolerance on residual norm\n"
		print "		tolg= - float"
		print "			[1.0e-18]: Tolerance on gradient norm\n"
		print "		tolobj= - float"
		print "			[none]: Tolerance on objective function value (Not relative value compared to initial one)\n"
		print "		tolobjrel= - float"
		print "			[none]: Tolerance on relative objective function value (Must range between 0 and 1)\n"
		print "	restart= - char"
		print "		[no]: Restart the inversion from previous run (restarting files' folder written in log_file, erased if solver normally ends)\n"
		print "	debug= - char"
		print "		[no]: Whether to print all command screen outputs during inversion and writes them in debug_log.txt in wrk_dir (see sep_python.py function RunShellCmd)\n"
		print "	dotprod= - int"
		print "		[no]: Whether to run dot-product test on linearized operators"
		print "		     NOTE: WON'T RUN INVERSION, ONLY DOT-PRODUCT TEST\n"

		print "INPUT PARAMETERS FOR REGULARIZED PROBLEM (default not regularized inversion)"
		print "	For solving 1/2*{|f(m)-d|^2 + epsilon^2|A(m-m_ref)|^2}\n"
		print "	fwd_cmd_reg_file= - char"
		print "		[identity]: Text file containing commands to be run to apply forward regularization operator\n"
		print "	adj_cmd_reg_file= - char"
		print "		[identity]: Text file containing commands to be run to apply adjoint regularization operator\n"
		print "	epsilon= - float"
		print "		[0.0]: Positive weighting factor on regularization term (if 0.0 not regularized inversion)\n"
		print "	epsilon_scale= - char"
		print "		[no]: Provide the user with an estimated epsilon scale to balance the two objective functions"
		print "		      NOTE: WON'T RUN INVERSION\n"
		print "	ref_model= - char"
		print "		[None]: Reference model file, if not specified, set to zero\n"

		print "INPUT PARAMETERS FOR SOLVER"
		print "	solver= - char"
		print "		['nlcg']: solver to use during the inversion (available: nlcg = non-linear conjugate-gradient method, lbfgs = limited-memory BFGS method, "
		print "                                                   steepest-descent = steepest-descent method, tnewton = truncated Newton)\n"
		print "	Limited-memory Broyden-Fletcher-Goldfarb-Shanno algorithm parameters: \n"
		print "	msteps= - int"
		print "		[niter]: Maximum number of steps to store to estimate the inverse Hessian (by default it runs BFGS method)"
		print "	H0_cmd_file= - char"
		print "		[identity]: Text file containing commands to be run to apply initial estimated Hessian inverse"
		print "		            If not provided an identity operator is assumed"
		print "	save_estimate= - char"
		print "		[no]: Whether to save vector of estimated Hessian inverse"
		print "		      It will save the vectors in wrk_dir/hessian_vectors"
		print "		      It requires the user to provide the parameter suffix if requested\n"
		print "	Non-linear conjugate-gradient method parameters: \n"
		print "	conj_method= - char"
		print "		[FR]: Conjugate gradient method to use (available: FR,PRP,HS,CD,LS,DY,HZ,BAN), see A SURVEY OF NONLINEAR CONJUGATE GRADIENT METHODS (Hager and Zhang, 2005)\n"
		print "	Truncated Newton method parameters: \n"
		print "	H_cmd_file= - char"
		print "		[Gauss-Newton]: Text file containing commands to be run to apply Hessian operator (!!!NOTE must contain input_m0.H indicating on which model point the Hessian is evaluated (i.e., H(m0)!!!)"
		print "		                If not provided a Gauss-Newton Hessian operator will be constructed using linearized operators and used during the inversion"
		print "		                In case of full Newton that depends on residuals specify the them by input_res.H in the command file"
		print "		                !Note this operator must be symmetric!"
		print "	Linear-system inversion parameters:"
		print "	!!!The solver will create an inversion folder for each iteration of the non-linear problem and place inversion results in it!!!"
		print "	!!!Inversion movie files are created as function of the requested movies of the non-linear problem                          !!!"
		print "		linear_solver= - char"
		print "			[symmetric-lcg] Linear solver to run during the Hessian iterative inversion (available: symmetric-lcg = linear conjugate-gradient for symmetric systems, lcg = linear conjugate-gradient, steepest-descent = steepest-descent method)"
		print "		warm_starts= - char"
		print "			[no]: If requested, the linear Hessian inversion is started from the previous inverted search direction"
		print "		niter_max_lin= - int"
		print "			[no default] Maximum number of iterations of the linear problem to run when starting with a zero initial model guess"
		print "		niter_min_lin= - int"
		print "			[niter_max] Number of iterations of the linear problem to run when linear inversion starts from previous linear inversion run"
		print "		other available parameters: maxval_lin,minval_lin,maxfevals_lin,maxhours_lin,tolr_lin,tolg_lin,tolobj_lin,tolobjrel_lin (SEE GENERIC LINEAR SOLVER FOR DETAILS)\n"

		print "INPUT PARAMETERS FOR STEPPER"
		print "	stepper= - char"
		print "		['parabolic']: stepper procedure to use (available: parabolic,sampler,linear)\n"
		print "	Common parameters: \n"
		print "	maxval= - float"
		print "		[None]: Hard bound for maximum values of the inverted model during inversion"
		print "	minval= - float"
		print "		[None]: Hard bound for minimum values of the inverted model during inversion\n"

		print "	Parabolic stepper parameters (fitting parabola with points: m_current, m1, m2): "
		print "	It tries to find a c_optimal*alpha*dm that minimizes the objective function starting from m_current \n"
		print "	c1= - float"
		print "		[1.0]: Scaling factor of first search point (i.e., m1 = c1*alpha*dm + m_current)"
		print "	c2= - float"
		print "		[2.0]: Scaling factor of first search point (i.e., m2 = c2*alpha*dm + m_current)"
		print "	ntry= - int"
		print "		[10]: Number of trials for finding the step length"
		print "	alpha= - float"
		print "		[0.]: Initial step-length guess (controls only first iteration)"
		print "	alpha_scale_min= - float"
		print "		[1e-10]: Minimum scaling factor (c_optimal) for step-length allowed"
		print "	alpha_scale_max= - float"
		print "		[50.00]: Maximum scaling factor (c_optimal) for step-length allowed"
		print "	shrink= - float"
		print "		[0.25]: Shrinking factor if step length is not found at a given trial\n"

		print "	Sampling stepper parameters (Find step length on a given interval with given sample points): \n"
		print "	npoints= - int"
		print "		[7]: Number of points to be tested in the interval"
		print "	scalemin= - float"
		print "		[0.5]: Scaling factor for left bound on the search interval"
		print "	scalemax= - float"
		print "		[1.5]: Scaling factor for right bound on the search interval"
		print "	ntry= - int"
		print "		[10]: Number of trials for finding the step length"
		print "	alpha= - float"
		print "		[0.]: Initial step-length guess (controls only first iteration)"
		print "	shrink= - float"
		print "		[0.25]: Shrinking factor if step length is not found at a given trial\n"

		print "	-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
		print "	!!!NOTE: the command files (fwd_nl_cmd_file,fwd_cmd_file,adj_cmd_file,fwd_cmd_reg_file,adj_cmd_reg_file,H0_cmd_file) must take a single input file and output a single file with names input.H and output.H !!!\n"
		print "	!!!REMARK: For the linearized operators (i.e., L(m0)dm), user must provide the m0 with name input_m0.H !!!\n"
		print "	!!!TRICK: User can define temporary random names in the command files. The tag tmp_random_name[0-9]+ is going to be substituted by a unique random series of characters !!!\n"
		print "	-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"

	else:
		print "\nRunning generic non-linear solver for L2-norm problems"
		#Parsing command line
		[pars,files]=sep.parse_args(sys.argv)

		#Checking command line and setting parameters
		linear=False
		fwd_nl_cmd_file=sep.from_cmd_line("fwd_nl_cmd_file") 		#non-linear forward operator commands
		fwd_cmd_file=sep.from_cmd_line("fwd_cmd_file") 				#linearized forward operator commands
		adj_cmd_file=sep.from_cmd_line("adj_cmd_file") 				#linearized adjoint operator commands

		#Debugging tool
		debug_in=sep.from_cmd_line("debug",default="no") 			#turning debugging mode on
		if (not (debug_in=="no" or debug_in=="n" or debug_in=="0")): sep.debug=True

		#Parsing for regularization weight
		epsilon=sep.from_cmd_line("epsilon",default=0.0,conv="float")
		epsilon_scale_in=sep.from_cmd_line("epsilon_scale",default="no")#Evaluate epsilon scale?
		epsilon_scale = False
		if (not (epsilon_scale_in=="no" or epsilon_scale_in=="n" or epsilon_scale_in=="0")): epsilon_scale = True
		if(epsilon<0.0):
			assert False, "Error, provide a positive epsilon (%s) for regularized inversion"%(epsilon)
		elif(epsilon!=0.0 or epsilon_scale):
			fwd_cmd_reg_file=sep.from_cmd_line("fwd_cmd_reg_file",default='identity')
			adj_cmd_reg_file=sep.from_cmd_line("adj_cmd_reg_file",default='identity')
			ref_model=sep.from_cmd_line("ref_model",default=None)

		data=sep.from_cmd_line("data")					 			#data file
		init_model=sep.from_cmd_line("init_model") 					#initial model file
		wrk_dir=sep.from_cmd_line("wrk_dir","./")	 				#name of inversion directory
		if (wrk_dir[-1]!="/"):wrk_dir+="/"							#adding slash to directory name if necessary
		sep.RunShellCmd("mkdir -p %s"%(wrk_dir))					#make sure the folder for inversion exists
		log_file=sep.from_cmd_line("log_file","inversion_log.txt") 	#log file to write solver information
		log_file=wrk_dir+log_file
		if (not (debug_in=="no" or debug_in=="n" or debug_in=="0")):
			sep.debug_file=wrk_dir+"debug_log.txt"					#log file for debug mode
		restarting=sep.from_cmd_line("restart",default="no")		#Restarting?
		if(restarting=="no" or restarting=="n" or restarting=="0"):
			restarting = "no"
			sep.RunShellCmd("rm -f %s"%(log_file))					#Erase previous log file if any
		dotprod=sep.from_cmd_line("dotprod",default="0")  			#run dot-product?

		#Solver to use
		solvertorun=sep.from_cmd_line("solver",default="nlcg")  	#solver to run

		####### Run the dot-product test if requested ############
		if(not (dotprod=="no" or dotprod=="n" or dotprod=="0")):
			if(epsilon>0.0):
				info = "	DOT-PRODUCT TEST FOR EXTENDED REGULARIZED OPERATOR"
				print info; solv.write_log_file(log_file,info)
				dot_test_reg_prob=non_linear_reg_prob_obj.nl_reg_prob(linear,init_model,data,nobj_functions=2)
				dot_test_reg_prob.set_prob(fwd_nl_cmd_file,fwd_cmd_file,adj_cmd_file)
				dot_test_reg_prob.set_reg(epsilon,ref_model,fwd_cmd_reg_file,adj_cmd_reg_file)
				dot_test_reg_prob.dot_prod(log_file)
				sys.exit()
			else:
				dot_test_prob=nl_prob(linear,init_model,data)
				dot_test_prob.set_prob(fwd_nl_cmd_file,fwd_cmd_file,adj_cmd_file)
				dot_test_prob.dot_prod(log_file)
				sys.exit()
		##########################################################

		###### Create stepper ############################
		kind_step=sep.from_cmd_line("stepper",default="parabolic") 				#type of stepper requested
		#Common parameters
		maxval_step=sep.from_cmd_line("maxval",default=None,conv="float") 		#max value for inverted model during inversion
		minval_step=sep.from_cmd_line("minval",default=None,conv="float") 		#min value for inverted model during inversion
		if (kind_step == 'parabolic'):
			c1_step=sep.from_cmd_line("c1",default=1.0,conv="float")
			c2_step=sep.from_cmd_line("c2",default=2.0,conv="float")
			ntry_step=sep.from_cmd_line("ntry",default=10,conv="int")
			alpha_step=sep.from_cmd_line("alpha",default=0.,conv="float")
			alpha_scale_min_in=sep.from_cmd_line("alpha_scale_min",default=1.0e-10,conv="float")
			alpha_scale_max_in=sep.from_cmd_line("alpha_scale_max",default=50.00,conv="float")
			shrink_step=sep.from_cmd_line("shrink",default=0.25,conv="float")
			#Shrink factor must be smaller than 1.0
			if(not 0<shrink_step<1): assert False, 'Shrinking factor not: 0 < %s < 1'%(shrink_step)
			stepper=parab_step.stepper_parab(c1=c1_step,c2=c2_step,ntry=ntry_step,alpha=alpha_step,alpha_scale_min=alpha_scale_min_in,alpha_scale_max=alpha_scale_max_in,shrink=shrink_step,maxval=maxval_step,minval=minval_step)
		elif (kind_step == 'sampler'):
			npoints_step=sep.from_cmd_line("npoints",default=7,conv="int")
			scalemin_step=sep.from_cmd_line("scalemin",default=0.5,conv="float")
			scalemax_step=sep.from_cmd_line("scalemax",default=1.5,conv="float")
			#Scalemin must be smaller than scalemax
			if(not scalemin_step<scalemax_step): assert False, 'scalemin not smaller than scalemax: %s < %s'%(scalemin_step,scalemax_step)
			ntry_step=sep.from_cmd_line("ntry",default=10,conv="int")
			alpha_step=sep.from_cmd_line("alpha",default=0.,conv="float")
			shrink_step=sep.from_cmd_line("shrink",default=0.25,conv="float")
			#Shrink factor must be smaller than 1.0
			if(not 0<shrink_step<1): assert False, 'Shrinking factor not: 0 < %s < 1'%(shrink_step)
			stepper=sample_step.stepper_sample(npoints=npoints_step,scalemin=scalemin_step,scalemax=scalemax_step,ntry=ntry_step,alpha=alpha_step,shrink=shrink_step,maxval=maxval_step,minval=minval_step)
		elif (kind_step == 'linear'):
			stepper=linear_step.stepper_linear(maxval=maxval_step,minval=minval_step)
		else:
			assert False,'Error in choosing stepper: %s not a stepper type'%(stepper)

		##################################################

		#####PROBLEM INSTANTIATION######################################################
		if(epsilon_scale):
			inv_model="dummy_model_name.H"
		else:
			inv_model=sep.from_cmd_line("inv_model") 				#inverted-model-file name
		if(not "/" in inv_model): inv_model=wrk_dir+inv_model		#Place inverted model in wrk_dir if a path is not provided
		if(epsilon>0.0 or epsilon_scale):
			#Create problem instance
			nl_inverse_problem=non_linear_reg_prob_obj.nl_reg_prob(linear,init_model,data,inv_model,nobj_functions=2)
			#Set forward and adjoint operator
			nl_inverse_problem.set_prob(fwd_nl_cmd_file,fwd_cmd_file,adj_cmd_file)

			if(epsilon_scale):
				#Balancing the gradients in the data space
				epsilon_test=1.0
				nl_inverse_problem.set_reg(epsilon_test,ref_model,fwd_reg_cmd_file=fwd_cmd_reg_file,adj_reg_cmd_file=adj_cmd_reg_file)
				#Evaluation of epsilon scale (will stop the main program)
				try:
					nl_inverse_problem.epsilon_scale(log_file)
				except (ValueError,AssertionError):
					print "!!!RunTime error: temporary files are not removed!!!"
					nl_inverse_problem.removefiles=False

			nl_inverse_problem.set_reg(epsilon,ref_model,fwd_cmd_reg_file,adj_cmd_reg_file)
		else:
			#Create problem instance
			nl_inverse_problem=nl_prob(linear,init_model,data,inv_model)
			#Set forward and adjoint operator
			nl_inverse_problem.set_prob(fwd_nl_cmd_file,fwd_cmd_file,adj_cmd_file)
		if(restarting!="no"): nl_inverse_problem.restart.restarting=True
		################################################################################

		#Continues getting other necessary parameters
		suffix=sep.from_cmd_line("suffix") 									 #suffix for movie file names
		niter_in=sep.from_cmd_line("niter",conv="int")  					 #number of iterations
		maxfevals_in=sep.from_cmd_line("maxfevals",default=0,conv="int")  	 #number of function evaluations
		maxhours_in=sep.from_cmd_line("maxhours",default=0.0,conv="float")   #Total running time
		tolr_in=sep.from_cmd_line("tolr",default=1.0e-18,conv="float") 		 #Tolerance on residual norm
		tolg_in=sep.from_cmd_line("tolg",default=1.0e-18,conv="float") 		 #Tolerance on gradient norm
		tolobj_in=sep.from_cmd_line("tolobj",default=None,conv="float") 	 #Tolerance on objective function value
		tolobjrel_in=sep.from_cmd_line("tolobjrel",default=None,conv="float")#Tolerance on relative objective function value
		if(tolobjrel_in!=None):
			if(tolobjrel_in>1 or tolobjrel_in<0):assert False, "Error, provide a relative objective function tolerance between 0 and 1, current relative tolerance %s"%(tolobjrel_in)


		#####SOLVER INSTANTIATION###############
		#Instantiate stopping criteria obj
		stop1=stop.stopper_basic(niter=niter_in,maxfevals=maxfevals_in,maxhours=maxhours_in,tolr=tolr_in,tolg=tolg_in,tolobj=tolobj_in,tolobjrel=tolobjrel_in)
		#Creating solver instance
		if (solvertorun=='nlcg'):
			conj_method_step=sep.from_cmd_line("conj_method",default="FR") 		#Conjugate method to be used during the inversion
			assert (conj_method_step in ["FR","PRP","HS","CD","LS","DY","HZ","BAN"]), "%s is not a conjugate method"%(conj_method_step)
			print "Solver in use: Non-linear conjugate gradient"
			solver=nlcg.nlcg_solver(stop1,beta_type=conj_method_step)
		elif(solvertorun=='steepest-descent'):
			conj_method_step="steepest descent"
			print "Solver in use: Steepest-descent method"
			solver=nlcg.nlcg_solver(stop1,beta_type=conj_method_step)
		elif (solvertorun=='lbfgs'):
			msteps_in=sep.from_cmd_line("msteps",default=niter_in,conv="int")  				#number of steps for inverse Hessian estimation
			if(msteps_in<niter_in):  print "Solver in use: L-BFGS"							#Using less steps than number of iterations
			if(msteps_in>=niter_in): print "Solver in use: BFGS"							#Using same or "more" steps than number of iterations
			H0_in=sep.from_cmd_line("H0_cmd_file",default=None)								#Command file containing template for initial inverse Hessian estimate
			if (H0_in != None):
				H0_in=op.Operator("Hessian initial estimate",H0_in,"input.H","output.H")	#Operator to apply initial inverse Hessian estimate
			save_estimate=sep.from_cmd_line("save_estimate",default="0")  					#Save vectors of inverse Hessian estimate
			if (not (save_estimate=="no" or save_estimate=="n" or save_estimate=="0")):
				estimation_folder=wrk_dir+"hessian_vectors/"
				sep.RunShellCmd("mkdir -p %s"%(estimation_folder))							#make sure the folder for hessian estimation exists
				suffix=sep.from_cmd_line("suffix") 											#suffix for estimation file names
				step_file_prefix=estimation_folder+"lbfgs_model_step"+suffix				#prefix for files containing model steps
				grad_diff_file_prefix=estimation_folder+"lbfgs_grad_diff"+suffix			#prefix for files containing gradient differences
			else:
				#Do not save the inverse Hessian estimate
				step_file_prefix=None
				grad_diff_file_prefix=None
			solver=lbfgs.lbfgs_solver(msteps_in,stop1,H0_in,step_file_prefix,grad_diff_file_prefix)
		elif (solvertorun=='tnewton'):
			H_in=sep.from_cmd_line("H_cmd_file",default=None)								#Command file containing template for Hessian operator
			if(H_in==None):
				assert(epsilon == 0.0), "!!!ERROR: If regularization is used, user must provide definition of Hessian matrix to be solved (e.g., (F'F + epsilon^2A'A) where F = df/dm)!!!"
				print "Solver in use: Truncated Gauss-Newton method"
				#Combining Linearized operators into Gauss-Newton Hessian
				GN_op=op.combine_operators(nl_inverse_problem.op_fwd,nl_inverse_problem.op_adj)
				#Redundant setting of input_m0 (Solver will check as well)
				GN_op.input_m0="input_m0.H"
			else:
				#Reading user-defined Hessian operator
				print "Solver in use: Truncated Newton method"
				GN_op=op.Operator("User-defined Hessian matrix",H_in,"input.H","output.H","input_m0.H",["input_res.H"])
			#Creating the instance of the problem
			linear_inverse_problem=lin_prob(True,init_model,init_model) #NOTE: init_model used as dummy model to create linear problem instance
			#Setting linear problem operators
			linear_inverse_problem.op_fwd=GN_op
			linear_inverse_problem.op_adj=copy.deepcopy(GN_op)							#Strange problem in gradient function without deepcopy
			#Reading linear inversion parameters
			niter_max_lin=sep.from_cmd_line("niter_max_lin",conv="int")  				#number of iterations of linear problem
			niter_min_lin=sep.from_cmd_line("niter_min_lin",default=None,conv="int")  	#number of iterations of linear problem when starting from previous run
			warm_starts_in=sep.from_cmd_line("warm_starts",default="no")  				#requesting warm starts?
			warm_starts=False
			if (not (warm_starts_in=="no" or warm_starts_in=="n" or warm_starts_in=="0")): warm_starts=True
			maxfevals_lin=sep.from_cmd_line("maxfevals_lin",default=0,conv="int")  	 	#number of function evaluations
			maxhours_lin=sep.from_cmd_line("maxhours_lin",default=0.0,conv="float")   	#Total running time of the linear inversion
			tolr_lin=sep.from_cmd_line("tolr_lin",default=1.0e-18,conv="float") 		#Tolerance on residual norm
			tolg_lin=sep.from_cmd_line("tolg_lin",default=1.0e-18,conv="float") 		#Tolerance on gradient norm
			tolobj_lin=sep.from_cmd_line("tolobj_lin",default=None,conv="float")      	#Tolerance on objective function value
			tolobjrel_lin=sep.from_cmd_line("tolobjrel_lin",default=None,conv="float")	#Tolerance on relative objective function value
			if(tolobjrel_lin!=None):
				assert (not(tolobjrel_lin>1 or tolobjrel_lin<0)), "Error, provide a relative objective function tolerance between 0 and 1, current relative tolerance %s"%(tolobjrel_lin)
			#Instantiating stoppers and checking solver to use
			stop1_lin=stop.stopper_basic(niter=niter_max_lin,maxfevals=maxfevals_lin,maxhours=maxhours_lin,tolr=tolr_lin,tolg=tolg_lin,tolobj=tolobj_lin,tolobjrel=tolobjrel_lin)
			stop2_lin=stop_sym.stopper_symmetric(niter=niter_max_lin,maxfevals=maxfevals_lin,maxhours=maxhours_lin,tolr=tolr_lin,tolobj=tolobj_lin,toleta=tolobjrel_lin) #for symmetric operators
			#Adding hard bounds if any
			maxval_lin=sep.from_cmd_line("maxval_lin",default=None,conv="float") 		#max value for inverted model during inversion of linear problem
			minval_lin=sep.from_cmd_line("minval_lin",default=None,conv="float") 		#min value for inverted model during inversion of linear problem
			stepper_lin=solv.stepper(maxval=maxval_lin,minval=minval_lin)
			#Instantiating linear solver for truncated-Newton system
			linear_solver_in=sep.from_cmd_line("linear_solver",default="symmetric-lcg")
			if(linear_solver_in=="symmetric-lcg"):
				#Instantiation of conjugate-gradient symmetric solver
				lcg_solv=lcg_sym.lcg_symmetric_solver(stop2_lin,stpr=stepper_lin)
				linear_inverse_problem.sym=True #Necessary to compute the right objective function
			elif(linear_solver_in=="lcg"):
				#Instantiation of conjugate-gradient solver
				lcg_solv=lcg.lcg_solver(stop1_lin,stpr=stepper_lin)
				linear_inverse_problem.sym=False
			elif(linear_solver_in=="steepest-descent"):
				#Instantiation of steepest-descent solver
				lcg_solv=lcg.lcg_solver(stop1_lin,stpr=stepper_lin,steepest=True)
				linear_inverse_problem.sym=False
			else:
				print "Error, %s not a solver option for linear system!"%(linear_solver_in)
				sys.exit()
			#Instantiating truncated-Newton solver
			solver=truncnewton.trunc_newton_solver(lcg_solv,linear_inverse_problem,wrk_dir,niter_max_lin,suffix,niter_min=niter_min_lin,warm_start=warm_starts,stoppr=stop1)
		else:
			print "Error, %s not a solver option!"%(solvertorun)
			sys.exit()

		########################################

		#Movie files definition section
		if('iteration_movies' in pars):
			movies=pars['iteration_movies']
			obj_movie="";model_movie="";grad_movie="";res_movie="";
			if('obj' in movies): obj_movie=wrk_dir+"obj%s.H"%(suffix)
			if('model' in movies): model_movie=wrk_dir+"model%s.H"%(suffix)
			if('gradient' in movies): grad_movie=wrk_dir+"gradient%s.H"%(suffix)
			if('residual' in movies): res_movie=wrk_dir+"residual%s.H"%(suffix)
			nl_inverse_problem.set_movies(obj_movie,model_movie,grad_movie,res_movie)



		####### Run the solver ############
		print "		Starting inversion"
		try:
			solver.run(nl_inverse_problem,stepper,log_file)
			print "		Solver has finished\n"
		except (ValueError,ZeroDivisionError,AssertionError):
			print "!!!RunTime error: temporary files are not removed!!!"
			solv.write_log_file(log_file,"!!!RunTime error: Inversion Crashed!!!")
			if hasattr(solver, 'removefiles'): solver.removefiles=False
			if hasattr(nl_inverse_problem, 'removefiles'): nl_inverse_problem.removefiles=False
			if hasattr(stepper, 'removefiles'): stepper.removefiles=False
			assert False, "!!!RunTime error: Inversion Crashed!!!"
		###################################


################################################################################
######################	                   END                   ###############
###################### 	  GENERIC SOLVER FOR NON-LINEAR PROBLEM  ###############
################################################################################
