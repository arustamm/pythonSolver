#!/usr/bin/python
#Unit test for Non-linear CG with out-of-core python solver
import python_nlcg_outcore as nlcg
import python_solver_outcore as solv 
import sep_python as sep # import sep functions
import operator_obj as op
import stopper_basic as stop
import LCG as lin_prob
import stepper_parab as parab_step

class Rosenbrock_prob(solv.problem):
#Rosenbrock function: f(x,y)=100*(y-x^2)^2+(1-x)^2

	def objf(self,res):
		"""Rosenbrock objective function"""
		sep.Cp(res,self.obj)
		obj=self.obj
		return obj
		
	def resf(self,model):
		"""Rosenbrock residual same as objective function"""
		tmp1_ros=sep.tmp_file("tmp1_res.H")
		tmp2_ros=sep.tmp_file("tmp2_res.H")
		cmd="Window <%s n1=1 f1=0 >%s out=%s@;"%(model,tmp1_ros,tmp1_ros) #x model
		cmd1="Window <%s n1=1 f1=1 >%s out=%s@"%(model,tmp2_ros,tmp2_ros) #y model
		sep.RunShellCmd(cmd+cmd1)
		cmd="Math file1=%s file2=%s exp='100.0*(file2-file1*file1)*(file2-file1*file1)+(1.0-file1)*(1.0-file1)' >%s out=%s@"%(tmp1_ros,tmp2_ros,self.res,self.res)
		sep.RunShellCmd(cmd)
		sep.Rm(tmp1_ros+" "+tmp2_ros)
		res=self.res
		return res
		
	def dresf(self,model,dmodel):
		"""Linearized residual"""
		tmp1_ros=sep.tmp_file("tmp1_grad.H")
		tmp2_ros=sep.tmp_file("tmp2_grad.H")
		tmp3_ros=sep.tmp_file("tmp3_grad.H")
		tmp4_ros=sep.tmp_file("tmp4_grad.H")
		cmd="Window <%s n1=1 f1=0 >%s out=%s@;"%(model,tmp1_ros,tmp1_ros) #x model
		cmd1="Window <%s n1=1 f1=1 >%s out=%s@"%(model,tmp2_ros,tmp2_ros) #y model
		sep.RunShellCmd(cmd+cmd1)
		#Derivative with respect to x
		cmd="Math file1=%s file2=%s exp='-400.0*file1*(file2-file1*file1)-2.0*(1.0-file1)' >%s out=%s@"%(tmp1_ros,tmp2_ros,tmp3_ros,tmp3_ros)
		sep.RunShellCmd(cmd)
		#Derivative with respect to y
		cmd="Math file1=%s file2=%s exp='200.0*(file2-file1*file1)' >%s out=%s@"%(tmp1_ros,tmp2_ros,tmp4_ros,tmp4_ros)
		sep.RunShellCmd(cmd)
		#Putting the two gradient component together
		cmd="Cat %s %s axis=1 >%s out=%s@"%(tmp3_ros,tmp4_ros,tmp1_ros,tmp1_ros)
		sep.RunShellCmd(cmd)
		#(dF/dm)'*dm
		sep.Dot(tmp1_ros,dmodel,self.dres)
		sep.Rm(tmp1_ros+" "+tmp2_ros+" "+tmp3_ros+" "+tmp4_ros)
		dres=self.dres
		return dres
		
	def gradf(self,model,residual):
		"""Gradient of Rosenbrock function"""
		tmp1_ros=sep.tmp_file("tmp1_grad.H")
		tmp2_ros=sep.tmp_file("tmp2_grad.H")
		tmp3_ros=sep.tmp_file("tmp3_grad.H")
		tmp4_ros=sep.tmp_file("tmp4_grad.H")
		cmd="Window <%s n1=1 f1=0 >%s out=%s@;"%(model,tmp1_ros,tmp1_ros) #x model
		cmd1="Window <%s n1=1 f1=1 >%s out=%s@"%(model,tmp2_ros,tmp2_ros) #y model
		sep.RunShellCmd(cmd+cmd1)
		#Derivative with respect to x
		cmd="Math file1=%s file2=%s exp='-400.0*file1*(file2-file1*file1)-2.0*(1.0-file1)' >%s out=%s@"%(tmp1_ros,tmp2_ros,tmp3_ros,tmp3_ros)
		sep.RunShellCmd(cmd)
		#Derivative with respect to y
		cmd="Math file1=%s file2=%s exp='200.0*(file2-file1*file1)' >%s out=%s@"%(tmp1_ros,tmp2_ros,tmp4_ros,tmp4_ros)
		sep.RunShellCmd(cmd)
		#Putting the two gradient component together
		cmd="Cat %s %s axis=1 >%s out=%s@"%(tmp3_ros,tmp4_ros,self.grad,self.grad)
		sep.RunShellCmd(cmd)
		sep.Rm(tmp1_ros+" "+tmp2_ros+" "+tmp3_ros+" "+tmp4_ros)
		grad=self.grad
		return grad


if __name__ == '__main__':
	print "Unit tests for out-of-core non-linear CG"
################################################################################	
###################### FIRST TEST DIAGONAL MATRIX ##############################
################################################################################
	print "		First test diagonal: alpha*I m = alpha*[1...1]'"
	print "		Test non-linear solver to invert a linear problem"
	alpha=1e-5
	nmodel=400;
	ndata=nmodel;
	print "		alpha=%s, nmodel=ndata=%s"%(alpha,nmodel)
	initial_model="init_model_diag.H"
	data_vector="data_diag.H"
	inv_model="inv_model_diag.H"
	#Creating initial model vector
	cmd="Spike n1=%s mag=0 >%s"%(nmodel,initial_model)
	sep.RunShellCmd(cmd)
	#Creating data vector
	cmd="Math file1=%s exp='%s+file1' >%s "%(initial_model,alpha,data_vector)
	sep.RunShellCmd(cmd)
	#Instantiate problem object
	linear=True
	prob1=lin_prob.diag_prob(linear,initial_model,data_vector,inv_model)
	prob1.set_prob_par(alpha)
	#setting movie files in the problem
	prob1.set_movies(obj_movie="obj_prob1.H",model_movie="model_prob1.H",grad_movie="grad_prob1.H",res_movie="res_prob1.H")
	#Instantiate stopping criteria obj
	stop1=stop.stopper_basic(niter=100)
	#Creating solver instance
	nlcg_solv=nlcg.nlcg_solver(stop1)
	print "		Starting inversion"
	#Run inversion
	nlcg_solv.run(prob1)
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
###################### SECOND TEST ROSENBROCK FNC ##############################
################################################################################
	print "\n		Test for Rosenbrock function optimization with out-of-core solver"
	print "		Initial point: -1,-1"
	#Creating initial model
	init_model="initial_model_ros.H"
	inv_model_ros="inv_mod_ros.H"
	cmd="Spike n1=2 k1=1,2 mag=-1,-1 nsp=2 >%s"%(init_model)
	sep.RunShellCmd(cmd)
	#Instantiate problem object
	linear=False
	prob2=Rosenbrock_prob(linear,init_model,inverted_model_file=inv_model_ros)
	#setting movie files in the problem
	prob1.set_movies(obj_movie="obj_prob2.H",model_movie="model_prob2.H")
	#Run inversion
	print "		Starting inversion"
	parabolic_step1=parab_step.stepper_parab()
	nlcg_solv.run(prob2,parabolic_step1)
# 	nlcg_solv.run(prob2)
	print "		Solver has finished"
	#Show inverted solution
	print "		Inverted solution point (true solution [1.0,1.0]):"
	print sep.Get_value(inv_model_ros,2)
	sep.Rm(initial_model+" "+inv_model_ros)
################################################################################
######################	           END            ##############################
###################### SECOND TEST ROSENBROCK FNC ##############################
################################################################################








	