#Generic non-linear problem with L2 linear regularization term (L2-norm problem: 1/2*|f(m)-d|^2 + epsi^2/2|A(m-m_ref)|)
import python_solver_outcore as solv
import sep_python as sep # import sep functions
import operator_obj as op
import stepper_linear as step_lin
import math
import sys


class nl_reg_prob(solv.problem):
	"""General non-linear problem object"""
	def set_prob(self,fwd_nl_cmd_file,fwd_cmd_file,adj_cmd_file):
		#In this problem we assume that the template series of commands takes one input file (input.H) and spits out one output file (output.H)
		self.op_nl_fwd=op.Operator("nonlinear operator forward",fwd_nl_cmd_file,"input.H","output.H")
		self.op_fwd=op.Operator("operator forward",fwd_cmd_file,"input.H","output.H","input_m0.H")
		self.op_adj=op.Operator("operator adjoint",adj_cmd_file,"input.H","output.H","input_m0.H")
		return

	def set_reg(self,epsilon,ref_model=None,fwd_reg_cmd_file='identity',adj_reg_cmd_file='identity'):
		#In this problem we assume that the template series of commands takes one input file (input.H) and spits out one output file (output.H)
		"""Operators for regularization"""
		if(fwd_reg_cmd_file=='identity' and adj_reg_cmd_file=='identity'):
			self.op_reg_fwd=self.op_reg_adj='identity'
		elif(fwd_reg_cmd_file!='identity' and adj_reg_cmd_file=='identity'):
			assert False, 'Error only forward operator set for regularization term. Set the adjoint as well.'
		elif(fwd_reg_cmd_file=='identity' and adj_reg_cmd_file!='identity'):
			assert False, 'Error only adjoint operator set for regularization term. Set the forward as well.'
		else:
			self.op_reg_fwd=op.Operator("regularization operator forward",fwd_reg_cmd_file,"input.H","output.H")
			self.op_reg_adj=op.Operator("regularization operator adjoint",adj_reg_cmd_file,"input.H","output.H")
		self.epsilon=epsilon
		#Reference model
		if(ref_model!=None):
			self.ref_model=ref_model
		else:
			self.ref_model=sep.tmp_file("tmp_ref_model.H")
			sep.Cp(self.model,self.ref_model)
			sep.Zero(self.ref_model)
			self.files_to_clean.append(self.ref_model)
		self.tmp_reg_file=sep.tmp_file("tmp_reg_file.H")
		self.files_to_clean.append(self.tmp_reg_file)
		return

	def __del__(self):
		"""Overriding default destructor"""
		solv.problem.__del__(self)
		import sep_python as sep
		if hasattr(self, 'removefiles'):
			if (self.removefiles):
				sep.Rm(self.files_to_clean)
			else:
				for ifile in self.files_to_clean:
					print "	Temporary file not removed for non-linear regularized problem: %s"%(ifile)
		return

	def epsilon_scale(self,log_file=None):
		"""Function to obtain epsilon that balances the first objective function values"""
		solv.write_log_file(log_file,info="REGULARIZED PROBLEM log file\n")
		info = "Epsilon Scale evaluation"
		print info
		solv.write_log_file(log_file,info+"\n")

		#Compute residuals
		res=self.get_res(self.model)

		#Balancing the first gradient in the extended-data space
		res_data_norm=sep.Norm_incore(res[0])
		res_model_norm=sep.Norm_incore(res[1])
		#If regularization term is zero, try to perform linearized step
		if(res_model_norm==0.0):
			if(sep.Norm_incore(self.model)==0.0):
				info = "WARNING: initial model norm equal zero!"
				print info
				solv.write_log_file(log_file,info)
			info = "Trying to perform a linearized step"
			print info
			solv.write_log_file(log_file,info)
			#Compute the gradient
			grad=self.get_grad(self.model)
			#Gradient in the data space
			dgrad=self.get_dres(self.model,grad)
			#Computing linear step length
			dgrad0_res=sep.Dot_incore(res[0],dgrad[0])
			dgrad0_dgrad0=sep.Dot_incore(dgrad[0],dgrad[0])
			assert (not(math.isnan(dgrad0_res) or math.isnan(dgrad0_dgrad0))), "Error! Obtained NaN: gradient-dataspace-norm = %s, gradient-dataspace-dot-residuals = %s"%(dgrad0_dgrad0,dgrad0_res)
			if(dgrad0_dgrad0!=0.):
				alpha=-dgrad0_res/dgrad0_dgrad0
			else:
				info = "Cannot compute linearized alpha for the given problem"
				solv.write_log_file(log_file,info)
				assert False, info
			#model=model+alpha*grad
			sep.Sum(self.model,grad,1.0,alpha)
			res=self.resf(self.model)
			#Recompute the new objective function
			res_data_norm=sep.Norm_incore(res[0])
			res_model_norm=sep.Norm_incore(res[1])
		#If regularization term is still zero, stop the solver
		if(res_model_norm!=0.0):
			epsilon=res_data_norm/res_model_norm
			info = "	Epsilon balancing the objective functions is: %s"%(epsilon)
			print info
			solv.write_log_file(log_file,info+"\n")
			print "Terminating solver\n"
		else:
			info = "Model residual component norm is zero, cannot find epsilon scale"
			solv.write_log_file(log_file,info)
			assert False, info
		solv.write_log_file(log_file,info="REGULARIZED PROBLEM end log file\n")
		sys.exit() #Stopping main program
		return


	# define function that computes objective function value
	def objf(self,res):
		"""0.5 norm squared of residual"""
		#Computing separately objective functions
		#data residuals
		sep.Dot(res[0],res[0],self.obj_list[0])
		cmd="Solver_ops file1=%s scale1_r=0.5 op=scale"%(self.obj_list[0])
		sep.RunShellCmd(cmd)
		#model residuals
		sep.Dot(res[1],res[1],self.obj_list[1])
		cmd="Solver_ops file1=%s scale1_r=0.5 op=scale"%(self.obj_list[1])
		sep.RunShellCmd(cmd)
		#Computing total objective function value
		sep.Add(self.obj_list[0],self.obj_list[1],self.obj)
		obj=self.obj
		return obj

	# define function that computes residuals
	def resf(self,model):
		"""f(m) - d"""
		self.op_nl_fwd.set_input_output(model,self.res[0])
		stat=self.op_nl_fwd.run() #Apply fwd non-linear modeling operator
		if(stat!=0): assert False, "problem running forward operator to compute the residual"
		sep.Sum(self.res[0],self.data,1.0,-1.0)
		#Run regularization part
		sep.Cp(self.ref_model,self.tmp_reg_file)
		#epsilon*(m-m_ref)
		sep.Sum(self.tmp_reg_file,model,-self.epsilon,self.epsilon)
		if(self.op_reg_fwd!='identity'):
			self.op_reg_fwd.set_input_output(self.tmp_reg_file,self.res[1])
			#A*epsilon*(m-m_ref)
			stat=self.op_reg_fwd.run() #Apply fwd regularization operator
			if(stat!=0): assert False, "problem running forward operator to compute regularization term"
		else: #Identity operator
			sep.Cp(self.tmp_reg_file,self.res[1])
		#Return
		res=self.res
		return res

	def dresf(self,model,dmodel):
		"""F dm = dr"""
		#Apply a forward modeling
		self.op_fwd.set_input_output(dmodel,self.dres[0],model)
		stat=self.op_fwd.run() #Apply linearized fwd modeling operator
		if(stat!=0): assert False, "problem running linearized forward operator"
		#Apply a forward modeling of regularization term
		if(self.op_reg_fwd!='identity'):
			self.op_reg_fwd.set_input_output(dmodel,self.dres[1])
			#A dm = dr_model
			stat=self.op_reg_fwd.run() #Apply fwd modeling operator of regularization term
			if(stat!=0): assert False, "problem running forward operator for regularization term"
		else: #Identity operator
			sep.Cp(dmodel,self.dres[1])
		#epsilon*A dm = epsilon*dr_model
		sep.Scale(self.dres[1],self.epsilon)
		dres = self.dres
		return dres

	def gradf(self,model,residual):
		"""F'r = g"""
		#Apply an adjoint modeling
		self.op_adj.set_input_output(residual[0],self.grad,model)
		stat=self.op_adj.run() #Apply linearized adj modeling operator
		if(stat!=0): assert False, "problem running adjoint operator"
		#Apply an adjoint operator of regularization term
		if(self.op_reg_fwd!='identity'):
			self.op_reg_adj.set_input_output(residual[1],self.tmp_reg_file)
			stat=self.op_reg_adj.run() #Apply adj operator of regularization term
			if(stat!=0): assert False, "problem running adjoint operator of regularization"
		else: #Identity operator
			sep.Cp(residual[1],self.tmp_reg_file)
		#B'r_data + epsilon*A'r_model
		sep.Sum(self.grad,self.tmp_reg_file,1.0,self.epsilon)
		grad=self.grad
		return grad
