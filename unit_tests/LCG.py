#!/usr/bin/python
#Unit test for Linear CG with out-of-core python solver
import python_lcg_outcore as lcg
import python_solver_outcore as solv 
import sep_python as sep # import sep functions
import operator_obj as op
import stopper_basic as stop

#Diagonal matrix problem
class diag_prob(solv.problem):
	"""Linear diagonal problem object"""
	
	def set_prob_par(self,alpha):
		self.alpha=alpha
		return
	
	# define function that computes objective function value
	def objf(self,res):
		"""0.5 norm of residual"""
		sep.Norm(res,self.obj)
		cmd="Solver_ops file1=%s scale1_r=0.5 op=scale"%(self.obj)
		sep.RunShellCmd(cmd)
		obj=self.obj
		return obj

	# define function that computes residuals		
	def resf(self,model):
		"""alpha*I m - d"""
		cmd="Add %s %s scale=%s,-1 >%s"%(model,self.data,self.alpha,self.res)
		sep.RunShellCmd(cmd)
		res=self.res
		return res
		
	def dresf(self,model,dmodel):
		"""alpha*I dm"""
		#Apply a linearized forward modeling
		cmd="Add %s scale=%s >%s"%(dmodel,self.alpha,self.dres)
		sep.RunShellCmd(cmd)
		dres=self.dres
		return dres
		
	def gradf(self,model,residual):
		"""adjoint operator:alpha*I r"""
		cmd="Add %s scale=%s >%s"%(residual,self.alpha,self.grad)
		sep.RunShellCmd(cmd)
		grad=self.grad
		return grad

#Lower triangular matrix problem	
class lowtri_prob(solv.problem):
	"""Linear lower triangular problem object"""
	__lowtri_prob_count=0
	
	def set_prob_par(self,ndata,nmodel,matrix="tmp_matrix.H"):
		self.ndata=ndata
		self.nmodel=nmodel
		self.matrix=matrix
		lowtri_prob.__lowtri_prob_count+=1
		self.tmp_res=sep.tmp_file("tmp_lowtri_%s.H"%(lowtri_prob.__lowtri_prob_count))
		#Create matrix
		cmd="./python_solver/matrix1 matrix_name=%s ndata=%s nmodel=%s build_m=1  >/dev/null"%(self.matrix,self.ndata,self.nmodel)
		sep.RunShellCmd(cmd)
		return
	
	def objf(self,res):
		"""0.5 norm of residual"""
		sep.Norm(res,self.obj)
		cmd="Solver_ops file1=%s scale1_r=0.5 op=scale"%(self.obj)
		sep.RunShellCmd(cmd)
		obj=self.obj
		return obj
		
	def resf(self,model):
		"""A m - d"""
		cmd="./python_solver/matrix1 matrix_name=%s <%s adj=0  >%s "%(self.matrix,model,self.tmp_res)
		sep.RunShellCmd(cmd)
		cmd="Add %s %s scale=1,-1 >%s out=%s@"%(self.tmp_res,self.data,self.res,self.res)
		sep.RunShellCmd(cmd)
		res=self.res
		return res
		
	def dresf(self,model,dmodel):
		"""A dm"""
		#Apply a linearized forward modeling
		cmd="./python_solver/matrix1 matrix_name=%s <%s adj=0 >%s out=%s@"%(self.matrix,dmodel,self.dres,self.dres)
		sep.RunShellCmd(cmd)
		dres=self.dres
		return dres
		
	def gradf(self,model,residual):
		"""adjoint operator:alpha*I r"""
		cmd="./python_solver/matrix1 matrix_name=%s <%s adj=1  >%s out=%s@"%(self.matrix,residual,self.grad,self.grad)
		sep.RunShellCmd(cmd)
		grad=self.grad
		return grad
		
	def __del__(self):
		"""Overriding default destructor"""
		import sep_python as sep
		sep.Rm(self.tmp_res)
		lowtri_prob.__lowtri_prob_count-=1
		return


if __name__ == '__main__':
	print "Unit tests for out-of-core linear CG"
################################################################################	
###################### FIRST TEST DIAGONAL MATRIX ##############################
################################################################################
	print "		First test diagonal: alpha*I m = alpha*[1...1]'"
	alpha=1e-5
	nmodel=400;
	ndata=nmodel;
	print "		alpha=%s, nmodel=ndata=%s"%(alpha,nmodel)
	initial_model="init_model.H"
	data_vector="data.H"
	inv_model="inv_model.H"
	#Creating initial model vector
	cmd="Spike n1=%s mag=0 >%s"%(nmodel,initial_model)
	sep.RunShellCmd(cmd)
	#Creating data vector
	cmd="Math file1=%s exp='%s+file1' >%s "%(initial_model,alpha,data_vector)
	sep.RunShellCmd(cmd)
	#Instantiate problem object
	linear=True
	prob1=diag_prob(linear,initial_model,data_vector,inv_model)
	prob1.set_prob_par(alpha)
	#setting movie files in the problem
	prob1.set_movies(obj_movie="obj_prob1.H",model_movie="model_prob1.H",grad_movie="grad_prob1.H",res_movie="res_prob1.H")
	#Instantiate stopping criteria obj
	stop1=stop.stopper_basic(niter=500)
	#Creating solver instance
	lcg_solv=lcg.lcg_solver(stop1)
	print "		Starting inversion"
	#Run inversion
	lcg_solv.run(prob1)
	print "		Solver has finished"
	#Show solution vector
	print "		Inverted model vector:"
	print sep.Get_value(inv_model,nmodel)
 	sep.Rm(initial_model+" "+data_vector+" "+inv_model)
################################################################################
######################	           END            ##############################
###################### FIRST TEST DIAGONAL MATRIX ##############################	
################################################################################	
	
################################################################################	
###################### SECOND TEST TRIANGULAR MTRX #############################
################################################################################	
	print "\n		Second test lower triangular matrix: A m = [1...1]''"
	nmodel=100;
	ndata=nmodel;
	print "		nmodel=ndata=%s"%(nmodel)
	initial_model="init_model.H"
	data_vector="data.H"
	inv_model="inv_model.H"
	#Creating initial model vector
	cmd="Spike n1=%s mag=0 >%s"%(nmodel,initial_model)
	sep.RunShellCmd(cmd)
	#Creating data vector
	tmp_file=sep.tmp_file("tmp_data.H")
	cmd="Spike n1=%s mag=0 >%s"%(ndata,tmp_file)
	sep.RunShellCmd(cmd)
	cmd="Math file1=%s exp='1.0+file1' >%s "%(tmp_file,data_vector)
	sep.RunShellCmd(cmd)
	sep.Rm(tmp_file)
	#Instantiate problem object
	linear=True
	prob2=lowtri_prob(linear,initial_model,data_vector,inv_model)
	matrix="tmp_matrix_prob2.H"
	prob2.set_prob_par(ndata,nmodel,matrix)
	#setting movie files in the problem
	prob2.set_movies(obj_movie="obj_prob2.H",model_movie="model_prob2.H",grad_movie="grad_prob2.H",res_movie="res_prob2.H")
	print "		Starting inversion"
	#Run inversion
	lcg_solv.run(prob2)
	print "		Solver has finished"
	#Show solution vector
	print "		Inverted model vector:"
	print sep.Get_value(inv_model,nmodel)
	print "		Predicted data vector:"
	print sep.Get_value(prob2.dresf(inv_model,inv_model),nmodel)
	sep.Rm(initial_model+" "+data_vector+" "+inv_model+" "+matrix)
################################################################################
######################	           END            ##############################
###################### SECOND TEST TRIANGULAR MTRX #############################
################################################################################
	
	
	
	
	
	