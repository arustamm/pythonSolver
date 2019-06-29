#module containing object for basic stopper
import sys
import operator
import math
import sep_python as sep
import time
import python_solver_outcore as solv
from timeit import default_timer as timer





class stopper_basic(solv.stopper):


	def __init__(self,niter=0,maxfevals=0,maxhours=0.0,tolr=1.0e-18,tolg=1.0e-18,tolobj=None,tolobjrel=None):
		self.niter=niter
		self.maxfevals=maxfevals
		self.maxhours=maxhours
		self.tolr=tolr
		self.tolg=tolg
		self.tolobj=tolobj
		self.tolobjrel=tolobjrel
		#Starting timer
		self.__start=timer()
		return

	def reset_stopper(self):
		"""Function to reset timer of the stopper"""
		#Restarting timer
		self.__start=timer()
		return

		#Beware stopper is going to change the gradient/obj/res files
	def run(self,iter,prblm,log_file=None,verbose=True):
		#Variable to impose stopping to solver
		stop = False
		#Taking time run so far (hours)
		elapsed_time=(timer()-self.__start)/3600.0
		secs=elapsed_time*3600.0
		#Printing elapsed time in hours, minutes, seconds
		hours= secs//3600
		mins = (secs % 3600)//60
		secs = (secs % 60)
		info2print="Elapsed time: %d hours, %d minutes, %d seconds"%(hours,mins,secs)
		solv.write_log_file(log_file,info=info2print)
		solv.write_log_file(log_file,info="Current date & time: %s"%(time.strftime("%c")))
		#Stop by number of iterations
		if((self.niter > 0) and (iter > self.niter)):
			stop = True
			info =  "Terminate: maximum number of iterations reached"
			if(verbose): print info
			solv.write_log_file(log_file,info=info)
			return stop
		if((self.maxfevals > 0) and (prblm.get_fevals() >= self.maxfevals)):
			stop = True
			info =  "Terminate: maximum number of evaluations"
			if(verbose): print info
			solv.write_log_file(log_file,info=info)
			return stop
		if((self.maxhours > 0.) and (elapsed_time >= self.maxhours)):
			stop = True
			info =  "Terminate: maximum number hours reached %s"%(elapsed_time)
			if(verbose): print info
			solv.write_log_file(log_file,info=info)
			return stop
		if(prblm.get_rnorm() < self.tolr):
			stop = True
			info =  "Terminate: minimum residual tolerance reached %s"%(prblm.get_rnorm())
			if(verbose): print info
			solv.write_log_file(log_file,info=info)
			return stop
		if(prblm.get_gnorm() < self.tolg):
			stop = True
			info =  "Terminate: minimum gradient tolerance reached %s"%(prblm.get_gnorm())
			if(verbose): print info
			solv.write_log_file(log_file,info=info)
			return stop
		if(self.tolobj!=None):
			if(prblm.get_obj_value() < self.tolobj):
				stop = True
				info =  "Terminate: objective function value tolerance of %s reached, objective function value %s"%(self.tolobj,prblm.get_obj_value())
				if(verbose): print info
				solv.write_log_file(log_file,info=info)
				return stop
		if(self.tolobjrel!=None):
			if(prblm.initial_obj_value==None):
				assert False, "Initial objective function value not set by solver class in problem class! Stopping!"
			if(prblm.get_obj_value()/prblm.initial_obj_value < self.tolobjrel):
				stop = True
				info =  "Terminate: relative objective function value tolerance of %s reached,  relative objective function value %s"%(self.tolobjrel,prblm.get_obj_value()/prblm.initial_obj_value)
				if(verbose): print info
				solv.write_log_file(log_file,info=info)
				return stop
		return stop
