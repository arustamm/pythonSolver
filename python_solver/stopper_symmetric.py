#module containing object for linear symmetric stopper (gradient is equal to residual (1/2m'Am -b'm))
import sys
import operator
import math
import sep_python as sep
import python_solver_outcore as solv
import time
from timeit import default_timer as timer





class stopper_symmetric(solv.stopper):


	def __init__(self,niter=0,maxfevals=0,maxhours=0.0,tolr=1.0e-18,tolobj=None,toleta=None):
		self.niter=niter
		self.maxfevals=maxfevals
		self.maxhours=maxhours
		self.tolr=tolr
		self.tolobj=tolobj
		self.toleta=toleta # |Am - b|/|b|
		self.data_norm=None
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
		residual_norm=prblm.get_rnorm()
		#Necessary to compute eta =|Am - b|/|b|
		if(prblm.data != None and self.data_norm == None):
			self.data_norm = sep.Norm_incore(prblm.data)
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
		if(residual_norm < self.tolr):
			stop = True
			info =  "Terminate: minimum residual tolerance reached %s"%(residual_norm)
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
		if(self.data_norm!=None and self.toleta!=None):
			if(residual_norm < self.toleta*self.data_norm):
				stop = True
				info =  "Terminate: eta tolerance (i.e., |Am - b|/|b|) of %s reached, eta value %s"%(self.toleta,residual_norm/self.data_norm)
				if(verbose): print info
				solv.write_log_file(log_file,info=info)
				return stop
		return stop
