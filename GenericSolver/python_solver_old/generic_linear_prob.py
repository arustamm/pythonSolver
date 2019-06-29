#!/usr/bin/env python
#Generic linear problem solver with CG (L2-norm problem: 1/2*|Lm-d|^2)
import python_lcg_outcore as lcg
import python_lcg_symmetric_outcore as lcg_sym
import python_solver_outcore as solv
import sep_python as sep # import sep functions
import operator_obj as op
import stopper_basic as stop
import stopper_symmetric as stop_sym
import sys
import linear_reg_prob as lin_reg_prob_obj


class lin_prob(solv.problem):
	"""General linear problem object"""
	def set_prob(self,fwd_cmd_file,adj_cmd_file,symmetric=False):
		#In this problem we assume that the template series of commands takes one input file (input.H) and spits out one output file (output.H)
		self.op_fwd=op.Operator("operator forward",fwd_cmd_file,"input.H","output.H")
		self.op_adj=op.Operator("operator adjoint",adj_cmd_file,"input.H","output.H")
		self.sym=symmetric #whether the problem is symmetric or not (for square matrices)
		return

	# define function that computes objective function value
	def objf(self,res):
		"""0.5 norm squared of residual"""
		if(self.sym):
			cmd="Add %s %s scale=1,-1 > %s"%(res,self.data,self.obj)
			sep.RunShellCmd(cmd)
			sep.Dot(self.model,self.obj,self.obj)
		else:
			sep.Dot(res,res,self.obj)
		cmd="Solver_ops file1=%s scale1_r=0.5 op=scale"%(self.obj)
		sep.RunShellCmd(cmd)
		obj=self.obj
		return obj

	# define function that computes residuals
	def resf(self,model):
		"""B m - d"""
		self.op_fwd.set_input_output(model,self.res)
		#Don't run the forward operator if model is zero vector
		if(sep.Norm_incore(model) != 0.0):
			stat=self.op_fwd.run() #Apply fwd modeling operator
			if(stat!=0): assert False, "problem running forward operator to compute the residual"
		else:
			sep.Cp(self.data,self.res)
			sep.Zero(self.res) #Zeroing out a copy of the data
		sep.Sum(self.res,self.data,1.0,-1.0)
		res=self.res
		return res

	def dresf(self,model,dmodel):
		"""B dm = dr"""
		#Apply a forward modeling
		self.op_fwd.set_input_output(dmodel,self.dres)
		stat=self.op_fwd.run() #Apply fwd modeling operator
		if(stat!=0): assert False, "problem running forward operator"
		dres = self.dres
		return dres

	def gradf(self,model,residual):
		"""B'r = g"""
		#Apply an adjoint modeling
		self.op_adj.set_input_output(residual,self.grad)
		stat=self.op_adj.run() #Apply adj modeling operator
		if(stat!=0): assert False, "problem running adjoint operator"
		grad=self.grad
		return grad

################################################################################
################################ MAIN ##########################################
################################################################################

if __name__ == '__main__':
################################################################################
###################### GENERIC CG SOLVER FOR LINEAR PROBLEM  ###################
################################################################################
	#Parsing command line
	if(len(sys.argv) == 1):
		print "NAME"
		print "	generic_linear_prob - Solves a linear problem \n"
		print "SYNOPSIS"
		print "	generic_linear_prob.py fwd_cmd_file=fwd_cmds.txt adj_cmd_file=adj_cmds.txt data=data.H init_model=init_model.H inv_model=inv_model.H suffix=problem1 iteration_movies=obj,model,gradient,residual niter=10 dotprod=0 \n"
		print "DESCRIPTION"
		print "	Out-of-core CG for solving L2-norm residual difference for linear problem (1/2*|Lm-d|^2) or Am = d (if A is symmetric)"
		print "INPUT PARAMETERS"
		print "	fwd_cmd_file= - char"
		print "		[no default]: Text file containing commands to be run to apply forward\n"
		print "	adj_cmd_file= - char"
		print "		[no default]: Text file containing commands to be run to apply adjoint (not parsed if symmetric problem)\n"
		print "	symmetric= - char"
		print "		[no]: if problem is symmetric, the solver will use a CG method where only fwd is employed (not parsed if regularized problem or steepest-descent is used)\n"
		print "	steepest-descent= - char"
		print "		[no]: whether to run steepest descent instead of linear conjugate gradient\n"
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
		print "	maxval= - float"
		print "		[None]: Hard bound for maximum values of the inverted model during inversion\n"
		print "	minval= - float"
		print "		[none]: Hard bound for minimum values of the inverted model during inversion\n"
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
		print "			[1.0e-18]: Tolerance on gradient norm (Note: ignore for symmetric system)\n"
		print "		tolobj= - float"
		print "			[none]: Tolerance on objective function value (Not relative value compared to initial one)\n"
		print "		tolobjrel= - float"
		print "			[none]: Tolerance on relative objective function value (Should range between 0 and 1)"
		print "			        For symmetric problem this criterion is on |Am - d|/|d|\n"
		print "	restart= - char"
		print "		[no]: Restart the inversion from previous run (restarting files' folder written in log_file, erased if solver normally ends)\n"
		print "	debug= - char"
		print "		[no]: Whether to print all command screen outputs during inversion and writes them in debug_log.txt in wrk_dir (see sep_python.py function RunShellCmd)\n"
		print "	dotprod= - int"
		print "		[no]: Whether to run dot-product test on linear operator or regularized extended operator"
		print "		     NOTE: WON'T RUN INVERSION, ONLY DOT-PRODUCT TEST\n"
		print "INPUT PARAMETERS FOR REGULARIZED PROBLEM (default not regularized inversion)"
		print "	For solving 1/2*{|Lm-d|^2 + epsilon^2|A(m-m_ref)|^2}\n"
		print "	fwd_cmd_reg_file= - char"
		print "		[identity]: Text file containing commands to be run to apply forward regularization operator\n"
		print "	adj_cmd_reg_file= - char"
		print "		[identity]: Text file containing commands to be run to apply adjoint regularization operator\n"
		print "	epsilon= - float"
		print "		[0.0]: Positive weighting factor on regularization term (if 0.0 not regularized inversion)\n"
		print "	epsilon_scale= - char"
		print "		[no]: Provide the user with an estimated epsilon scale to balance the first gradient in the extended-data space"
		print "		      NOTE: WON'T RUN INVERSION\n"
		print "	ref_model= - char"
		print "		[None]: Reference model file, if not specified, set to zero\n"
		print "	----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
		print "	!!!NOTE: the command files (fwd_cmd_file,adj_cmd_file,fwd_cmd_reg_file,adj_cmd_reg_file) must take a single input file and output a single file with names input.H and output.H!!!\n"
		print "	!!!TRICK: User can define temporary random names in the command files. The tag tmp_random_name[0-9]+ is going to be substituted by a unique random series of characters !!!\n"
		print "	----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"

	else:
		print "\nRunning generic linear solver for linear problems"
		#Parsing command line
		[pars,files]=sep.parse_args(sys.argv)

		#Checking command line and setting parameters
		linear=True
		fwd_cmd_file=sep.from_cmd_line("fwd_cmd_file") 					#forward operator commands

		#Debugging tool
		debug_in=sep.from_cmd_line("debug",default="no") 			#turning debugging mode on
		if (not (debug_in=="no" or debug_in=="n" or debug_in=="0")): sep.debug=True

		#Check if steepest descent is requested
		steepest_in=sep.from_cmd_line("steepest-descent","no")			#run steepest-descent?
		steepest=True
		if(steepest_in=="no" or steepest_in=="n" or steepest_in=="0"):
			steepest=False

		#Parsing for regularization weight
		epsilon=sep.from_cmd_line("epsilon",default=0.0,conv="float")
		epsilon_scale_in=sep.from_cmd_line("epsilon_scale",default="no")#Evaluate epsilon scale?
		epsilon_scale = False
		if (not (epsilon_scale_in=="no" or epsilon_scale_in=="n" or epsilon_scale_in=="0")): epsilon_scale = True
		if(epsilon==0.0 and not epsilon_scale and not steepest):
			symmetric_in=sep.from_cmd_line("symmetric","no")			#is problem symmetric?
		elif(epsilon<0.0):
			assert False, "Error, provide a positive epsilon (%s) for regularized inversion"%(epsilon)
		else:
			symmetric_in="no"
			fwd_cmd_reg_file=sep.from_cmd_line("fwd_cmd_reg_file",default='identity')
			adj_cmd_reg_file=sep.from_cmd_line("adj_cmd_reg_file",default='identity')
			ref_model=sep.from_cmd_line("ref_model",default=None)

		#Checking whether to run symmetric matrix CG method
		if (not (symmetric_in=="no" or symmetric_in=="n" or symmetric_in=="0")):
			adj_cmd_file=fwd_cmd_file									#symmetric problem fwd=adj
			symmetric=True
		else:
			adj_cmd_file=sep.from_cmd_line("adj_cmd_file") 				#adjoint operator commands
			symmetric=False

		data=sep.from_cmd_line("data")					 				#data file
		init_model=sep.from_cmd_line("init_model") 						#initial model file
		wrk_dir=sep.from_cmd_line("wrk_dir","./")	 					#name of inversion directory
		if (wrk_dir[-1]!="/"):wrk_dir+="/"								#adding slash to directory name if necessary
		sep.RunShellCmd("mkdir -p %s"%(wrk_dir))						#make sure the folder for inversion exists
		log_file=sep.from_cmd_line("log_file","inversion_log.txt") 		#log file to write solver information
		log_file=wrk_dir+log_file
		if (not (debug_in=="no" or debug_in=="n" or debug_in=="0")):
			sep.debug_file=wrk_dir+"debug_log.txt"						#log file for debug mode
		restarting=sep.from_cmd_line("restart",default="no")			#Restarting?
		if(restarting=="no" or restarting=="n" or restarting=="0"):
			restarting = "no"
			sep.RunShellCmd("rm -f %s"%(log_file))						#Erase previous log file if any
		dotprod=sep.from_cmd_line("dotprod",default="no")  				#run dot-product?

		####### Run the dot-product test if requested ############
		if(not (dotprod=="no" or dotprod=="n" or dotprod=="0")):
			if(epsilon>0.0):
				info = "	DOT-PRODUCT TEST FOR EXTENDED REGULARIZED OPERATOR"
				print info; solv.write_log_file(log_file,info)
				dot_test_reg_prob=lin_reg_prob_obj.lin_reg_prob(linear,init_model,data,nobj_functions=2)
				dot_test_reg_prob.set_prob(fwd_cmd_file,adj_cmd_file)
				dot_test_reg_prob.set_reg(epsilon,ref_model,fwd_cmd_reg_file,adj_cmd_reg_file)
				dot_test_reg_prob.dot_prod(log_file)
				sys.exit()
			else:
				dot_test_prob=lin_prob(linear,init_model,data)
				dot_test_prob.set_prob(fwd_cmd_file,adj_cmd_file)
				dot_test_prob.dot_prod(log_file)
				sys.exit()
		##########################################################

		#####PROBLEM INSTANTIATION######################################################
		if(epsilon_scale):
			inv_model="dummy_model_name.H"
		else:
			inv_model=sep.from_cmd_line("inv_model") 				#inverted-model-file name
		if(not "/" in inv_model): inv_model=wrk_dir+inv_model		#Place inverted model in wrk_dir if a path is not provided
		if(epsilon>0.0 or epsilon_scale):
			#Create problem instance
			linear_inverse_problem=lin_reg_prob_obj.lin_reg_prob(linear,init_model,data,inv_model,nobj_functions=2)
			linear_inverse_problem.set_prob(fwd_cmd_file,adj_cmd_file)

			if(epsilon_scale):
				#Balancing the gradients in the data space
				epsilon_test=1.0
				linear_inverse_problem.set_reg(epsilon_test,fwd_reg_cmd_file=fwd_cmd_reg_file,adj_reg_cmd_file=adj_cmd_reg_file)
				#Evaluation of epsilon scale (will stop the main program)
				try:
					linear_inverse_problem.epsilon_scale(log_file)
				except (ValueError,AssertionError):
					print "!!!RunTime error: temporary files are not removed!!!"
					linear_inverse_problem.removefiles=False

			linear_inverse_problem.set_reg(epsilon,ref_model,fwd_cmd_reg_file,adj_cmd_reg_file)
		else:
			#Create problem instance
			linear_inverse_problem=lin_prob(linear,init_model,data,inv_model)
			#Set forward and adjoint operator
			linear_inverse_problem.set_prob(fwd_cmd_file,adj_cmd_file,symmetric)
		if(restarting!="no"): linear_inverse_problem.restart.restarting=True
		################################################################################

		#Continues getting other necessary parameters
		niter_in=sep.from_cmd_line("niter",conv="int")  					 #number of iterations
		maxfevals_in=sep.from_cmd_line("maxfevals",default=0,conv="int")  	 #number of function evaluations
		maxhours_in=sep.from_cmd_line("maxhours",default=0.0,conv="float")   #Total running time
		tolr_in=sep.from_cmd_line("tolr",default=1.0e-18,conv="float") 		 #Tolerance on residual norm
		tolg_in=sep.from_cmd_line("tolg",default=1.0e-18,conv="float") 		 #Tolerance on gradient norm
		tolobj_in=sep.from_cmd_line("tolobj",default=None,conv="float")      #Tolerance on objective function value
		tolobjrel_in=sep.from_cmd_line("tolobjrel",default=None,conv="float")#Tolerance on relative objective function value
		if(tolobjrel_in!=None):
			assert (not(tolobjrel_in>1 or tolobjrel_in<0)), "Error, provide a relative objective function tolerance between 0 and 1, current relative tolerance %s"%(tolobjrel_in)


		#####SOLVER INSTANTIATION###############
		#Instantiate stopping criteria obj
		stop1=stop.stopper_basic(niter=niter_in,maxfevals=maxfevals_in,maxhours=maxhours_in,tolr=tolr_in,tolg=tolg_in,tolobj=tolobj_in,tolobjrel=tolobjrel_in)
		stop2=stop_sym.stopper_symmetric(niter=niter_in,maxfevals=maxfevals_in,maxhours=maxhours_in,tolr=tolr_in,tolobj=tolobj_in,toleta=tolobjrel_in) #for symmetric operators
		#Adding hard bounds if any
		maxval_step=sep.from_cmd_line("maxval",default=None,conv="float") 		#max value for inverted model during inversion
		minval_step=sep.from_cmd_line("minval",default=None,conv="float") 		#min value for inverted model during inversion
		stepper=solv.stepper(maxval=maxval_step,minval=minval_step)
		#Creating solver instance
		if(not steepest):
			#Instantiation of conjugate-gradient solver
			lcg_solv=lcg.lcg_solver(stop1,stpr=stepper)
		else:
			#Instantiation of steepest-descent solver
			lcg_solv=lcg.lcg_solver(stop1,stpr=stepper,steepest=True)
		#Instantiation of conjugate-gradient symmetric solver
		lcg_sym_solv=lcg_sym.lcg_symmetric_solver(stop2,stpr=stepper)
		########################################


		#Movie files definition section
		if('iteration_movies' in pars):
			movies=pars['iteration_movies']
			suffix=sep.from_cmd_line("suffix") 		#suffix for movie file names
			obj_movie="";model_movie="";grad_movie="";res_movie="";
			if('obj' in movies): obj_movie=wrk_dir+"obj%s.H"%(suffix)
			if('model' in movies): model_movie=wrk_dir+"model%s.H"%(suffix)
			if('gradient' in movies and (not symmetric)): grad_movie=wrk_dir+"gradient%s.H"%(suffix)
			if('residual' in movies): res_movie=wrk_dir+"residual%s.H"%(suffix)
			linear_inverse_problem.set_movies(obj_movie,model_movie,grad_movie,res_movie)

		####### Run the solver ############
		print "		Starting inversion"
		try:
			if (symmetric):
				lcg_sym_solv.run(linear_inverse_problem,log_file)
			else:
				lcg_solv.run(linear_inverse_problem,log_file)
			print "		Solver has finished\n"
		except (ValueError,ZeroDivisionError,AssertionError):
			print "!!!RunTime error: temporary files are not removed!!!"
			solv.write_log_file(log_file,"!!!RunTime error: Inversion Crashed!!!")
			if (symmetric):
				if hasattr(lcg_sym_solv, 'removefiles'): lcg_sym_solv.removefiles=False
			else:
				if hasattr(lcg_solv, 'removefiles'): lcg_solv.removefiles=False
			if hasattr(linear_inverse_problem, 'removefiles'): linear_inverse_problem.removefiles=False
			if hasattr(stepper, 'removefiles'): stepper.removefiles=False
			assert False, "!!!RunTime error: Inversion Crashed!!!"
		###################################


################################################################################
######################	           END            ##############################
###################### GENERIC CG SOLVER FOR LINEAR PROBLEM  ###################
################################################################################
