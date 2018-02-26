#!/usr/bin/env python
import sep_python as sep
import operator_obj as operator
import sys,os




if __name__ == "__main__":
	if(len(sys.argv) == 1):
		print "NAME"
		print "	power_method - Applies power iteration method to a given operator \n"
		print "SYNOPSIS"
		print "	power_method.py cmd_file=cmd_file.txt niter=10 model_vector=model.H \n"
		print "DESCRIPTION"
		print "	Computes the maximum and minimum eigenvalues of a given operator"
		print "INPUT PARAMETERS"
		print "	cmd_file= - char"
		print "		[no default]: Text file containing commands to be run to apply operator\n"
		print "	niter= - char"
		print "		[no default]: Number of iterations to run for power method (maximum eigenvalue)\n"
		print "	niter_min= - char"
		print "		[niter]: Number of iterations to run for power method (minimum eigenvalue)\n"
		print "		         If not provided it is equal to niter\n"
		print "	model_vector= - char"
		print "		[no default]: Sep formatted file containing a template model vector\n"
		print "	max_lambda_file= - char"
		print "		[None]: Sep formatted file containing the estimate of the maximum egeinvalue\n"
		print "	min_lambda_file= - char"
		print "		[None]: Sep formatted file containing the estimate of the minimum egeinvalue\n"
		print "	max_eigenvector= - char"
		print "		[None]: Sep formatted file containing the estimated eigenvector associated with the largest eigenvalue\n"
		print "	min_eigenvector= - char"
		print "		[None]: Sep formatted file containing the estimated eigenvector associated with the smallest eigenvalue\n"
	else:
		#Parsing command line
		fwd_cmd_file=sep.from_cmd_line("cmd_file")
		niter=sep.from_cmd_line("niter",conv="int")										#number of iterations for maximum eigenvalue estimation
		niter_min=sep.from_cmd_line("niter_min",default=niter,conv="int")				#number of iterations for minimum eigenvalue estimation
		model_vector=sep.from_cmd_line("model_vector")
		max_lambda_file=sep.from_cmd_line("max_lambda_file",default=None)				#maximum lambda file
		min_lambda_file=sep.from_cmd_line("min_lambda_file",default=None)				#minimum lambda file
		max_vector_file=sep.from_cmd_line("max_eigenvector",default=None)				#maximum eigenvector file
		min_vector_file=sep.from_cmd_line("min_eigenvector",default=None)				#minimum eigenvector file
		tmp_lambda_file=sep.tmp_file("tmp_lambda.H")									#Temporary file to be appended

		#Creating operator object and setting input/output files
		fwd_op=operator.Operator("operator forward",fwd_cmd_file,"input.H","output.H")	#A matrix
		tmp_input_file=sep.tmp_file("tmp_input_power_method.H") 						#x vector
		tmp_output_file=sep.tmp_file("tmp_output_power_method.H")						#y vector
		fwd_op.set_input_output(tmp_input_file,tmp_output_file)							#setting y = Ax

		#Getting Append command path
		python_path_list=os.environ['PYTHONPATH'].split(":")
		indx = [i for i, s in enumerate(python_path_list) if '/python_solver' in s]
		pathsource=python_path_list[indx[0]]

		#Setting x vector
		sep.Cp(model_vector,tmp_input_file)
		#Randomize x vector
		sep.RunShellCmd("Solver_ops file1=%s op=rand"%(tmp_input_file))
		#Normalize x vector
		sep.Scale(tmp_input_file,1.0/sep.Norm_incore(tmp_input_file))

		#Variables necessary to save eigenvector estimate
		#Getting vector number of axes
		n_axes=sep.get_num_axes(tmp_input_file)
		python_path_list=os.environ['PYTHONPATH'].split(":")
		indx = [i for i, s in enumerate(python_path_list) if '/python_solver' in s]
		pathsource=python_path_list[indx[0]]

		#Running the power iteration method
		try:
			print "Running power method to estimate maximum eigenvalue for %s iterations"%(niter)
			for iter in range(niter):
				#Apply the operator to x
				#y = Ax
				stat=fwd_op.run()
				assert stat==0, "!!!Error applying provided operator!!!"
				#Estimating eigenvalue (Rayleigh quotient)
				#lambda = x'Ax
				lambda_est=sep.Dot_incore(tmp_input_file,tmp_output_file)
				#x = y / |y|_2
				sep.Cp(tmp_output_file,tmp_input_file)
				sep.Scale(tmp_input_file,1.0/sep.Norm_incore(tmp_input_file))
				print "	Estimated maximum eigenvalue at iter %s: %s"%(iter,lambda_est)
				if(iter==0):
					if (max_lambda_file!=None):
						sep.RunShellCmd("Spike n1=1 | Add scale=%s >%s"%(lambda_est,max_lambda_file))
						cmd="echo \"d1=1.0 o1=0.0 label1=\'iteration number\' label2=\'eigenvalue\' \" >> %s"%(max_lambda_file)
						sep.RunShellCmd(cmd)
						#If requested save max eigenvector
					if(max_vector_file!=None):
						sep.Cp(tmp_input_file,max_vector_file)
						#Adding axis information
						cmd="echo \"d%s=1.0 o%s=0.0 label%s=\'iteration number\' \" >> %s"%(n_axes+1,n_axes+1,n_axes+1,max_vector_file)
						sep.RunShellCmd(cmd)
				else:
					if (max_lambda_file!=None):
						sep.RunShellCmd("Spike n1=1 | Add scale=%s >%s"%(lambda_est,tmp_lambda_file))
						cmd=pathsource+"/Append axis=1 "
						sep.RunShellCmd(cmd+max_lambda_file+" "+tmp_lambda_file)
						cmd="echo \"d1=1.0 o1=0.0 label1=\'iteration number\' label2=\'eigenvalue\' \" >> %s"%(max_lambda_file)
						sep.RunShellCmd(cmd)
					#If requested save max eigenvector
					if(max_vector_file!=None):
						cmd=pathsource+"/Append axis=%s "%(n_axes+1)
						sep.RunShellCmd(cmd+max_vector_file+" "+tmp_input_file)
						#Adding axis information
						cmd="echo \"d%s=1.0 o%s=0.0 label%s=\'iteration number\' \" >> %s"%(n_axes+1,n_axes+1,n_axes+1,max_vector_file)
						sep.RunShellCmd(cmd)
			print "Running power method to estimate minimum eigenvalue for %s iterations"%(niter_min)
			#Re-randomize x vector
			sep.RunShellCmd("Solver_ops file1=%s op=rand"%(tmp_input_file))
			max_lambda_est_shift=-lambda_est
			#Shifting all eigenvalues by maximum one (i.e., A-muI)
			for iter in range(niter_min):
				#Apply the operator to x
				#y = Ax
				stat=fwd_op.run()
				assert stat==0, "!!!Error applying provided operator!!!"
				#y = Ax - mu*Ix
				sep.Sum(tmp_output_file,tmp_input_file,scale2=max_lambda_est_shift)
				#Estimating eigenvalue (Rayleigh quotient)
				#lambda = x'Ax
				lambda_est=sep.Dot_incore(tmp_input_file,tmp_output_file)
				#x = y / |y|_2
				sep.Cp(tmp_output_file,tmp_input_file)
				sep.Scale(tmp_input_file,1.0/sep.Norm_incore(tmp_input_file))
				print "	Estimated minimum eigenvalue at iter %s: %s"%(iter,lambda_est-max_lambda_est_shift)
				if(iter==0):
					if (min_lambda_file!=None):
						sep.RunShellCmd("Spike n1=1 | Add scale=%s >%s"%(lambda_est-max_lambda_est_shift,min_lambda_file))
						cmd="echo \"d1=1.0 o1=0.0 label1=\'iteration number\' label2=\'eigenvalue\' \" >> %s"%(min_lambda_file)
						sep.RunShellCmd(cmd)
					#If requested save max eigenvector
					if(min_vector_file!=None):
						sep.Cp(tmp_input_file,min_vector_file)
						#Adding axis information
						cmd="echo \"d%s=1.0 o%s=0.0 label%s=\'iteration number\' \" >> %s"%(n_axes+1,n_axes+1,n_axes+1,min_vector_file)
						sep.RunShellCmd(cmd)
				else:
					if (min_lambda_file!=None):
						sep.RunShellCmd("Spike n1=1 | Add scale=%s >%s"%(lambda_est-max_lambda_est_shift,tmp_lambda_file))
						cmd=pathsource+"/Append axis=1 "
						sep.RunShellCmd(cmd+min_lambda_file+" "+tmp_lambda_file)
						cmd="echo \"d1=1.0 o1=0.0 label1=\'iteration number\' label2=\'eigenvalue\' \" >> %s"%(min_lambda_file)
						sep.RunShellCmd(cmd)
					#If requested save max eigenvector
					if(min_vector_file!=None):
						cmd=pathsource+"/Append axis=%s "%(n_axes+1)
						sep.RunShellCmd(cmd+min_vector_file+" "+tmp_input_file)
						#Adding axis information
						cmd="echo \"d%s=1.0 o%s=0.0 label%s=\'iteration number\' \" >> %s"%(n_axes+1,n_axes+1,n_axes+1,min_vector_file)
						sep.RunShellCmd(cmd)
		finally:
			sep.Rm(tmp_input_file)
			sep.Rm(tmp_output_file)
			sep.Rm(tmp_lambda_file)
