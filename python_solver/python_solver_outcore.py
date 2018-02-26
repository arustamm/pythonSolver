#!/usr/bin/python
#module containing solver out of core class and functions
import sys, os
import operator
import math
import sep_python as sep
import time
import datetime


def write_log_file(log_file=None,info=None):
		"""Function to write log file"""
		#info: line to print in the log file
		if (log_file != None):
			with open(log_file,"a") as f:
				f.write(info+"\n")
		return

class solver:
	"""Solver parent object"""

	def __init__(self,stpr,stoppr):
		"""Dummy constructor."""
		return


	def run(self,prblm):
		"""Dummy solver running routine"""
		assert False, "!Implement run for solver in the derived class."
		return


class stepper:
	"""Stepper parent object"""

	def __init__(self,maxval=None,minval=None):
		"""Dummy constructor."""
		python_path_list=os.environ['PYTHONPATH'].split(":")
		indx = [i for i, s in enumerate(python_path_list) if '/python_solver' in s]
		self.__pathsource=python_path_list[indx[0]]
		self.maxval=maxval
		self.minval=minval
		self.files_to_clean=[]
		self.tmp=sep.tmp_file("tmp_clip.H"); self.files_to_clean.append(self.tmp)
		self.tmp_results=[] #Tuple containing objective function and residuals struct: [[res1_p1,res2_p1][res1_p2,res2_p2]] or [res1,res2]
		return

	def __del__(self):
		"""Overriding default destructor"""
		import sep_python as sep
		if hasattr(self, 'removefiles'):
			sep.Rm(self.files_to_clean)
		return

	def run(self,prblm,modl,dmodl,log_file=None):
		"""Dummy stepper running routine"""
		assert False, "!Implement run stepper in the derived class."
		#Return step length and whether the stepper was successful or not
		return alpha, success

	def clipping(self,model,log_file=None):
		"""Function to clip model value to certain value"""
		clipped=False
		if(self.maxval != None):
			cmd=self.__pathsource+"/Limit clip=%s chop=g <"%(self.maxval)
			sep.RunShellCmd(cmd+model+" >"+self.tmp)
			if(sep.Is_different(model,self.tmp)):
				sep.Cp(self.tmp,model)
				write_log_file(log_file,"	Maximum values clipped")
				clipped=True
		if(self.minval != None):
			cmd=self.__pathsource+"/Limit clip=%s chop=l <"%(self.minval)
			sep.RunShellCmd(cmd+model+" >"+self.tmp)
			if(sep.Is_different(model,self.tmp)):
				sep.Cp(self.tmp,model)
				write_log_file(log_file,"	Minimum values clipped")
				clipped=True
		return clipped

	def copy_tmp_result(self,res):
		"""Function to avoid reduntant computations (copying resudials only)"""
		#Copying objective function
		if(isinstance(res,list)):
			tmp_res=[]
			for ifile, ii in zip(res,range(len(res))):
				tmp_res_file=sep.tmp_file("tmp_copy_result_res%s.H"%ii); self.files_to_clean.append(tmp_res_file)
				sep.Cp(ifile,tmp_res_file)
				tmp_res.append(tmp_res_file)
		else:
			tmp_res_file=sep.tmp_file("tmp_copy_result_res.H"); self.files_to_clean.append(tmp_res_file)
			sep.Cp(res,tmp_res_file)
			tmp_res=tmp_res_file
		self.tmp_results.append(tmp_res)
		return

	def set_from_tmp_result(self,prblm,modl,ind_point):
		"""Function to set residuals from previous tested point"""
		#Setting model at ind_point (as to be the model that generated the residuals at ind_point!!!)
		prblm.set_model(modl)
		#Setting data residuals
		prblm.set_residual(self.tmp_results[ind_point])
		return

	def clear_tmp_result(self):
		"""Function to clear temporary rediuals"""
		for ifile in self.tmp_results:
			sep.Rm(ifile)
			if(isinstance(ifile,list)):
				for jfile in ifile:
					self.files_to_clean.remove(jfile)
			else:
				self.files_to_clean.remove(ifile)
		self.tmp_results=[]
		return

class stopper:
	"""Stopper parent object"""

	def __init__(self):
		"""Dummy constructor."""
		return

	def run(self,prblm,log_file=None):
		"""Dummy stopper running routine"""
		assert False, "!Implement run stopper in the derived class."
		return stop



class problem:
	"""Problem parent object"""
	#Counter for number of problem instantiated to unique name for tmp files
	__problem_count=0


	def __init__(self,linear,model_file="model.H",data_file=None,inverted_model_file="inverted_model.H",nobj_functions=1):
		"""Simple problem constructor."""
		#Portion to find folder with source code of solver
		self.files_to_clean=[]
		python_path_list=os.environ['PYTHONPATH'].split(":")
		indx = [i for i, s in enumerate(python_path_list) if '/python_solver' in s]
		self.__pathsource=python_path_list[indx[0]]
		self.linear=linear #Logical stating whether the problem is linear or not (to obtain correct fevals estimation)
		#Set internal copy of model, data and dmodel files
		self.model=sep.tmp_file("model_tmp.H");self.files_to_clean.append(self.model)
		#In case no data vector is necessary
		if(data_file!=None):
			self.data=sep.tmp_file("data.H")
		else:
			self.data=None
		self.dmodel=sep.tmp_file("dmodel_tmp.H");self.files_to_clean.append(self.dmodel)
		#internal files
		sep.Cp(model_file,self.model) #creating internal copy of the model file
		if(data_file!=None): sep.RunShellCmd("cp %s %s "%(data_file,self.data)) #copying header file of the data file
		sep.Cp(model_file,self.dmodel) #creating internal copy of the dmodel file
		sep.Zero(self.dmodel) #Zeroing dmodel file
		self.grad=sep.tmp_file("gradient.H");self.files_to_clean.append(self.grad)
		if(nobj_functions==1):
			self.res=sep.tmp_file("residual.H");self.files_to_clean.append(self.res)
			self.dres=sep.tmp_file("dresidual.H");self.files_to_clean.append(self.dres)
		elif(nobj_functions>1):
			self.res=[]
			self.dres=[]
			self.obj_list=[]
			for ifile in range(nobj_functions):
				file_name=sep.tmp_file("residual%s.H"%(ifile));self.files_to_clean.append(file_name)
				self.res.append(file_name)
				file_name=sep.tmp_file("dresidual%s.H"%(ifile));self.files_to_clean.append(file_name)
				self.dres.append(file_name)
				file_name=sep.tmp_file("obj%s.H"%(ifile));self.files_to_clean.append(file_name)
				self.obj_list.append(file_name)

		else:
			assert False, "Error nobj_functions must be one or greater, current nobj_functions: %s"%(nobj_functions)
		self.obj=sep.tmp_file("obj.H");self.files_to_clean.append(self.obj)
		self.inverted_model=inverted_model_file
		#Model and data space number of axes
		self.nmodel_axes=sep.get_num_axes(self.model)
		#Internal variables
		self.obj_updated=False
		self.res_updated=False
		self.grad_updated=False
		self.dres_updated=False
		self.fevals=0
		self.counter=0
		self.initial_obj_value=None #For relative objective function value
		#Output file names
		self.obj_movie=""
		self.model_movie=""
		self.grad_movie=""
		self.res_movie=""
		#Tmp files for problem object
		problem.__problem_count+=1
		#Restarting object
		now=datetime.datetime.now()
		restart_folder=sep.sep_find_datapath()+"restart_dir_"+now.isoformat()
		restart_folder=restart_folder.replace(":","-")
		self.restart=problem_restart(restart_folder)
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
					print "	Temporary file not removed for debugging in PROBLEM: %s"%(ifile)
		if(self.data!=None):sep.RunShellCmd("rm -f %s"%(self.data))
		problem.__problem_count-=1
		return

	def reset_problem(self,model_file,data_file=None,inverted_model_file=None):
		"""Function to reset problem object"""
		import re
		#Checking provided new model is consistent with internal one
		nmodel_axes_reset=sep.get_num_axes(model_file)
		assert (self.nmodel_axes==nmodel_axes_reset), "Reset model file inconsistent with internal one: naxes-internal=%s naxes-reset=%s"%(self.nmodel_axes,nmodel_axes_reset)
		#Resetting the model file
		sep.Cp(model_file,self.model) #creating internal copy of the model file
		#Resetting the data file if any
		if(self.data!=None and data_file!=None):
			#Checking provided new data is consistent with internal one
			ndata_axes_reset=sep.get_num_axes(data_file)
			ndata_axes=sep.get_num_axes(self.data)
			assert (ndata_axes==ndata_axes_reset), "Reset data file inconsistent with internal one: naxes-internal=%s naxes-reset=%s"%(ndata_axes,ndata_axes_reset)
			sep.RunShellCmd("cp %s %s "%(data_file,self.data)) #copying header file of the data file
		#Resetting the dmodel file
		sep.Zero(self.dmodel) #Zeroing dmodel file
		#Resetting inverted model file name
		if(inverted_model_file!=None): self.inverted_model=inverted_model_file
		#Resetting internal variables
		self.obj_updated=False
		self.res_updated=False
		self.grad_updated=False
		self.dres_updated=False
		self.fevals=0
		self.counter=0
		self.initial_obj_value=None #For relative objective function value
		#Resetting movie files
		self.obj_movie=""
		self.model_movie=""
		self.grad_movie=""
		self.res_movie=""
		#Resetting restarting object
		now=datetime.datetime.now()
		restart_folder=sep.sep_find_datapath()+"restart_dir_"+now.isoformat()
		restart_folder=restart_folder.replace(":","-")
		self.restart=problem_restart(restart_folder)
		self.removefiles=True
		return


	def set_movies(self,obj_movie="",model_movie="",grad_movie="",res_movie=""):
		"""Set movie files for inversion"""
		self.obj_movie = obj_movie
		self.model_movie = model_movie
		self.grad_movie = grad_movie
		self.res_movie = res_movie
		return

	def set_model(self,model):
		"""Setting internal model file"""
		if(sep.Is_different(self.model,model)):
			sep.Cp(model,self.model)
			self.obj_updated=False
			self.res_updated=False
			self.grad_updated=False
			self.dres_updated=False
		return

	def set_residual(self,residual):
		"""Setting internal residual file"""
		#Useful for linear inversion (so residual computation not necessary)
		sep.Cp(residual,self.res)
		self.res_updated=True
		return

	def get_model(self):
		model=self.model
		return model

	def get_data(self):
		data=self.data
		return data

	def get_dmodel(self):
		dmodel=self.dmodel
		return dmodel

	def get_obj(self,model):
		self.set_model(model)
		if(not(self.obj_updated)):
			self.res = self.get_res(self.model)
			self.obj = self.objf(self.res)
			self.obj_updated=True
		obj=self.obj
		return obj

	def get_res(self,model):
		self.set_model(model)
		if(not(self.res_updated)):
			self.res = self.resf(self.model)
			self.fevals += 1
			self.res_updated=True
		res=self.res
		return res

	def get_grad(self,model):
		self.set_model(model)
		if(not(self.grad_updated)):
			self.res  = self.get_res(self.model)
			self.grad = self.gradf(self.model,self.res)
			if self.linear: self.fevals += 1
			if not self.linear: self.fevals += 2
			self.grad_updated=True
		grad=self.grad
		return grad

	def get_dres(self,model,dmodel):
		self.set_model(model)
		if(not(self.dres_updated) or sep.Is_different(self.dmodel,dmodel)):
			sep.Cp(dmodel,self.dmodel)
			self.dres = self.dresf(self.model,self.dmodel)
			if self.linear: self.fevals += 1
			if not self.linear: self.fevals += 2
			self.dres_updated=True
		dres=self.dres
		return dres

	def get_fevals(self):
		fevals = self.fevals
		return fevals

	def get_rnorm(self):
		self.res=self.get_res(self.model)
		rnorm=sep.Norm_incore(self.res)
		return rnorm

	def get_gnorm(self):
		self.grad=self.get_grad(self.model)
		gnorm=sep.Norm_incore(self.grad)
		return gnorm

	def get_obj_value(self):
		obj_value=sep.Get_value(self.get_obj(self.model))
		return obj_value

	def output(self,model):
		#Output the inverted model so far only
		sep.Cp(model,self.inverted_model)
		self.counter += 1
		#Files where to save how the inversion proceeds by iterations
		if(self.counter == 1):
			if(self.obj_movie!=""):
				obj=self.get_obj(model)
				sep.Cp(obj,self.obj_movie)
				#Adding axis information
				cmd="echo \"d1=1.0 o1=0.0 label1=\'iteration number\' label2=\'objective function value\' \" >> %s"%(self.obj_movie)
				sep.RunShellCmd(cmd)
				#Outputting different objective functions if more than one
				if hasattr(self,'obj_list'):
					index_file=1
					for ifileobj in self.obj_list:
						obj_split=self.obj_movie.split(".H")
						#changing output objective functions' file names
						obj_movie_tmp=obj_split[0]+"%s.H"%(index_file)
						sep.Cp(ifileobj,obj_movie_tmp)
						#Adding axis information
						cmd="echo \"d1=1.0 o1=0.0 label1=\'iteration number\' label2=\'objective function value\' \" >> %s"%(obj_movie_tmp)
						sep.RunShellCmd(cmd)
						index_file+=1
			if(self.model_movie!=""):
				sep.Cp(model,self.model_movie)
				#Adding axis information
				cmd="echo \"d%s=1.0 o%s=0.0 label%s=\'iteration number\' \" >> %s"%(self.nmodel_axes+1,self.nmodel_axes+1,self.nmodel_axes+1,self.model_movie)
				sep.RunShellCmd(cmd)
			if(self.grad_movie!=""):
				grad=self.get_grad(model)
				sep.Cp(grad,self.grad_movie)
				#Adding axis information
				cmd="echo \"d%s=1.0 o%s=0.0 label%s=\'iteration number\' \" >> %s"%(self.nmodel_axes+1,self.nmodel_axes+1,self.nmodel_axes+1,self.model_movie)
				sep.RunShellCmd(cmd)
			if(self.res_movie!=""):
				res=self.get_res(model)
				if(isinstance(res,list)):
					index_file=1
					for ifileres in res:
						#number of axes for a given residual file
						ndata_axes=sep.get_num_axes(ifileres)
						res_split=self.res_movie.split(".H")
						#changing output residual file name
						res_movie_tmp=res_split[0]+"%s.H"%(index_file)
						sep.Cp(ifileres,res_movie_tmp)
						#Adding axis information
						cmd="echo \"d%s=1.0 o%s=0.0 label%s=\'iteration number\' \" >> %s"%(ndata_axes+1,ndata_axes+1,ndata_axes+1,res_movie_tmp)
						sep.RunShellCmd(cmd)
						index_file+=1
				else:
					ndata_axes=sep.get_num_axes(res)
					sep.Cp(res,self.res_movie)
					#Adding axis information
					cmd="echo \"d%s=1.0 o%s=0.0 label%s=\'iteration number\' \" >> %s"%(ndata_axes+1,ndata_axes+1,ndata_axes+1,self.res_movie)
					sep.RunShellCmd(cmd)
		else:
			if(self.obj_movie!=""):
				obj=self.get_obj(model)
				cmd=self.__pathsource+"/Append axis=1 "+self.obj_movie+" "+obj
				sep.RunShellCmd(cmd)
				#Adding axis information
				cmd="echo \"d1=1.0 o1=0.0 label1=\'iteration number\' label2=\'objective function value\' \" >> %s"%(self.obj_movie)
				sep.RunShellCmd(cmd)
				#Outputting different objective functions if more than one
				if hasattr(self,'obj_list'):
					index_file=1
					for ifileobj in self.obj_list:
						obj_split=self.obj_movie.split(".H")
						#changing output objective functions' file names
						obj_movie_tmp=obj_split[0]+"%s.H"%(index_file)
						cmd=self.__pathsource+"/Append axis=1 "
						sep.RunShellCmd(cmd+obj_movie_tmp+" "+ifileobj)
						#Adding axis information
						cmd="echo \"d1=1.0 o1=0.0 label1=\'iteration number\' label2=\'objective function value\' \" >> %s"%(obj_movie_tmp)
						sep.RunShellCmd(cmd)
						index_file+=1
			if(self.model_movie!=""):
				cmd=self.__pathsource+"/Append axis=%s "%(self.nmodel_axes+1)
				sep.RunShellCmd(cmd+self.model_movie+" "+model)
				#Adding axis information
				cmd="echo \"d%s=1.0 o%s=0.0 label%s=\'iteration number\' \" >> %s"%(self.nmodel_axes+1,self.nmodel_axes+1,self.nmodel_axes+1,self.model_movie)
				sep.RunShellCmd(cmd)
			if(self.grad_movie!=""):
				cmd=self.__pathsource+"/Append axis=%s "%(self.nmodel_axes+1)
				grad=self.get_grad(model)
				sep.RunShellCmd(cmd+self.grad_movie+" "+grad)
				#Adding axis information
				cmd="echo \"d%s=1.0 o%s=0.0 label%s=\'iteration number\' \" >> %s"%(self.nmodel_axes+1,self.nmodel_axes+1,self.nmodel_axes+1,self.grad_movie)
				sep.RunShellCmd(cmd)
			if(self.res_movie!=""):
				res=self.get_res(model)
				if(isinstance(res,list)):
					index_file=1
					for ifileres in res:
						#number of axes for a given residual file
						ndata_axes=sep.get_num_axes(ifileres)
						res_split=self.res_movie.split(".H")
						#changing output residual file name
						res_movie_tmp=res_split[0]+"%s.H"%(index_file)
						cmd=self.__pathsource+"/Append axis=%s "%(ndata_axes+1)
						sep.RunShellCmd(cmd+res_movie_tmp+" "+ifileres)
						#Adding axis information
						cmd="echo \"d%s=1.0 o%s=0.0 label%s=\'iteration number\' \" >> %s"%(ndata_axes+1,ndata_axes+1,ndata_axes+1,res_movie_tmp)
						sep.RunShellCmd(cmd)
						index_file+=1
				else:
					ndata_axes=sep.get_num_axes(res)
					cmd=self.__pathsource+"/Append axis=%s "%(ndata_axes+1)
					sep.RunShellCmd(cmd+self.res_movie+" "+res)
					#Adding axis information
					cmd="echo \"d%s=1.0 o%s=0.0 label%s=\'iteration number\' \" >> %s"%(ndata_axes+1,ndata_axes+1,ndata_axes+1,self.res_movie)
					sep.RunShellCmd(cmd)
		return

	def objf(self,res):
		"""Dummy objf running routine"""
		assert False, "!Implement objf for problem in the derived class."
		return obj

	def resf(self,model):
		"""Dummy resf running routine"""
		assert False, "!Implement resf for problem in the derived class."
		return res

	def dresf(self,model,dmodel):
		"""Dummy dresf running routine"""
		assert False, "!Implement dresf for problem in the derived class."
		return dres

	def gradf(self,model,residual):
		"""Dummy gradf running routine"""
		assert False, "!Implement gradf for problem in the derived class."
		return grad

	def dot_prod(self,log_file=None):
		"""Method to test dot-product of linerized operators"""
		info = "-------------------------------------------------"
		print info
		write_log_file(log_file, info)
		info = "Dot-product test on linearized forward and adjoint\n"
		print info; write_log_file(log_file, info)
		m1_file=sep.tmp_file("m1_tmp.H");self.files_to_clean.append(m1_file)
		d2_file=self.res
		#Copying model file and filling with random numbers
		sep.Cp(self.model,m1_file); sep.Rand(m1_file,snr=0.1)
		#Computing data from random model
		info = "	Running forward operator"
		print info; write_log_file(log_file, info)
		start = time.time()
		d1_file=self.dresf(self.model,m1_file)
		end = time.time()
		#Copying data file
		sep.Cp(d1_file,d2_file); sep.Rand(d2_file,snr=0.1)
		info = "	Runs in: %s seconds"%(end-start)
		print info; write_log_file(log_file, info)
		info =  "	Running adjoint operator"
		print info; write_log_file(log_file, info)
		start = time.time()
		m2_file=self.gradf(self.model,d2_file)
		end = time.time()
		info = "	Runs in: %s seconds"%(end-start)
		print info
		write_log_file(log_file, info)
		#Computing dot-products
		dot_data=sep.Dot_incore(d1_file,d2_file)
		dot_model=sep.Dot_incore(m1_file,m2_file)
		info = "	Dot-product data space: %s"%(dot_data)
		print info; write_log_file(log_file, info)
		info = "	Dot-product model space: %s"%(dot_model)
		print info; write_log_file(log_file, info)
		info = "	Absolute error: %s"%(dot_data-dot_model)
		print info; write_log_file(log_file, info)
		info = "	Relative error: %s \n"%((dot_data-dot_model)/dot_data)
		print info; write_log_file(log_file, info)
		info = "               End dot-product test"
		print info; write_log_file(log_file, info)
		info = "-------------------------------------------------"
		print info; write_log_file(log_file, info)
		return


class problem_restart:
	"""Generic problem restart object"""


	def __init__(self,restart_folder):
		"""Constructor for generic problem restart"""
		if(restart_folder[-1]!="/"): restart_folder+="/"
		self.restart_folder=restart_folder
		self.log_file=self.restart_folder+"restart_log.txt"
		self.restarting=False
		return

	def create_restart_folder(self):
		"""Function to create restart folder"""
		#Doesn't create folder if restarting the problem
		if(not self.restarting): sep.RunShellCmd("mkdir -p %s"%self.restart_folder)
		return

	def get_restart_folder(self,log_file):
		"""Function to get restart folder"""
		cmd="more %s | grep \'Restarting folder\' | colrm 1 18 | tail -1"%(log_file)
		stat,restart_folder=sep.RunShellCmd_nocheck(cmd)
		return restart_folder.strip(" ")

	def set_restart(self,problem,log_file):
		"""Function to set variables for restarting"""
		#Obtaining previous run restarting directory
		self.restart_folder=self.get_restart_folder(log_file)
		self.log_file=self.restart_folder+"restart_log.txt"
		#Obtaining iteration number
		value=self.get_info_log("iter")
		if (value != None):
			iter=int(value)
		else:
			assert False, "Problem restarting: iteration number not found!"
		value=self.get_info_log("feval")
		if (value != None):
			problem.fevals=int(value)
		else:
			assert False, "Problem restarting: feval number not found!"
		#Resetting counter for outputting files
		problem.counter = iter
		return iter

	def get_info_log(self,info,iter_num=None):
		"""Function returning the value of the infomation required from restart log file"""
		import re
		#Obtaining information from last or iter_num
		if (iter_num == None):
			cmd = "more %s | grep -w %s | tail -1"%(self.log_file,info)
		else:
			cmd = "more %s | grep -w %s | grep -w 'iter = %s'"%(self.log_file,info,iter_num)
		stat,out=sep.RunShellCmd_nocheck(cmd)
		#Matches integer and floating point numbers with or without exponent
		string_expression = re.compile("%s = ([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)"%(info))
		find=string_expression.search(out)
		if find:
			value=find.group(1)
		else:
			#if not found return None
			value = None
		return value

	def write_file(self,file_to_copy,file_to_write):
		"""Function to copy file to restart folder"""
		cmd="Cp %s %s out=%s@"%(file_to_copy,self.restart_folder+file_to_write,self.restart_folder+file_to_write)
		sep.RunShellCmd(cmd)
		write_log_file(self.log_file,info="Written %s in %s"%(file_to_write,self.restart_folder))
		return

	def copy_file_from_restart(self,file_to_copy,file_to_write):
		"""Function to copy file from restart folder"""
		cmd="Cp %s %s"%(self.restart_folder+file_to_copy,file_to_write)
		sep.RunShellCmd(cmd)
		write_log_file(self.log_file,info="Copied %s to %s"%(file_to_write,file_to_write))
		return

	def clean_problem_restart(self):
		"""Function to remove restart folder"""
		cmd = "rm -rf %s"%self.restart_folder
		sep.RunShellCmd(cmd)
		return
