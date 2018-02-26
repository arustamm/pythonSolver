#!/usr/bin/env python
import python_lbfgs_outcore as lbfgs
import sep_python as sep
import operator_obj as op
import sys


if __name__ == '__main__':
################################################################################
###################### 	  L-BFGS HESSIAN INVERSE APPLICATION     ###############
################################################################################
	#Parsing command line
	if(len(sys.argv) == 1):
		print "NAME"
		print "	Hessian_BFGS_apply - Script to apply estimate Hessian inverse to a given vector \n"
		print "SYNOPSIS"
		print "	Hessian_BFGS_apply.py estimate_dir= msteps= input_file= output_file= \n"
		print "DESCRIPTION"
		print "	Script to apply the estimated Hessian inverse from an inversion run of L-BFGS to a provided model vector\n"
		print "INPUT PARAMETERS"
		print "	estimate_dir= - char"
		print "		[no default]: Folder path containing Hessian vector estimate\n"
		print "	msteps= - int"
		print "		[no default]: Number of vectors to use in the estimation."
		print "		              NOTE: must be the maximum of number of vectors used during the inversion; otherwise estimate will be wrong!\n"
		print "	iter= - int"
		print "		[m_steps]: Iteration at which Hessian estimate is requested. Default is number of msteps to handle BFGS case.\n"
		print "	H0_cmd_file= - char"
		print "		[identity]: Text file containing commands to be run to apply initial estimated Hessian inverse\n"
		print "	input_file= - char"
		print "		[no default]: Name of input_file model file\n"
		print "	output_file= - char"
		print "		[no default]: Name of output_file model file\n"
	else:
		estimate_dir=sep.from_cmd_line("estimate_dir")
		if (estimate_dir[-1]!="/"):estimate_dir+="/"									#adding slash to directory name if necessary
		m_steps=sep.from_cmd_line("msteps",conv="int")
		iter=sep.from_cmd_line("iter",default=m_steps,conv="int")
		input_file=sep.from_cmd_line("input_file")
		output_file=sep.from_cmd_line("output_file")
		H0_in=sep.from_cmd_line("H0_cmd_file",default=None)								#Command file containing template for initial inverse Hessian estimate
		if (H0_in != None):
			H0_in=op.Operator("Hessian initial estimate",H0_in,"input.H","output.H")	#Operator to apply initial inverse Hessian estimate

		#Instantiating a L-BFGS solver to use BFGSMultiply
		solver=lbfgs.lbfgs_solver(m_steps,H0=H0_in)
		solver.grad_diff_files=['']*m_steps
		solver.step_files=['']*m_steps

		#Setting inverse Hessian estimate
		step_count=1
		for istep in range(iter-1,-1,-1):
			if(step_count <= solver.m_steps):
				step_index=istep%solver.m_steps #Modulo to handle limited memory
				filename=estimate_dir+"lbfgs_grad_diff*%s.H"%(istep)
				solver.grad_diff_files[step_index]=sep.RunShellCmd_output("ls %s | sort -V | sed -n 1p"%(filename))
				filename=estimate_dir+"lbfgs_model_step*%s.H"%(istep)
				solver.step_files[step_index]=sep.RunShellCmd_output("ls %s | sort -V | sed -n 1p"%(filename))
				denom_dot=sep.Dot_incore(solver.grad_diff_files[step_index],solver.step_files[step_index])
				#Recomputing rho for restart
				solver.check_rho(denom_dot,step_index)
				step_count+=1
			else:
				break

		#Applying estimated inverse Hessian
		sep.Cp(input_file,output_file)
		solver.BFGSMultiply(output_file,input_file,iter)
		#To keep same sign to the output model vector (implementation related)
		sep.Scale(output_file,-1.0)
