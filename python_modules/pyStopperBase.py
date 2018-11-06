#Module containing derived object for basic stopper
import pyStopper
import pyProblem
import time
from timeit import default_timer as timer

class BasicStopper(pyStopper.Stopper):
	"""Basic Stopper with different options"""
	
	def __init__(self,niter=0,maxfevals=0,maxhours=0.0,tolr=1.0e-18,tolg=1.0e-18,tolobj=None,tolobjrel=None,logger=None):
		"""Constructor for Basic Stopper"""
		#Criteria to evaluate whether or not to stop the solver
		self.niter=niter
		self.maxfevals=maxfevals
		self.maxhours=maxhours
		self.tolr=tolr
		self.tolg=tolg
		self.tolobj=tolobj
		self.tolobjrel=tolobjrel
		#Logger to write to file stopper information
		self.logger=logger
		#Starting timer
		self.__start=timer()
		return
		
	def reset_timer(self):
		"""Function to reset timer of the stopper"""
		#Restarting timer
		self.__start=timer()
		return
		
	#Beware stopper is going to change the gradient/obj/res files
	def run(self,prblm,iter,initial_obj_value,verbose=True):
		if(not isinstance(prblm,pyProblem.Problem)): raise TypeError("Input variable is not a Problem object")
		#Variable to impose stopping to solver
		stop = False
		#Taking time run so far (hours)
		elapsed_time=(timer()-self.__start)/3600.0
		secs=elapsed_time*3600.0
		#Printing elapsed time in hours, minutes, seconds
		hours= secs//3600
		mins = (secs % 3600)//60
		secs = (secs % 60)
		#Printing time stamp to log file if provided
		msg="Elapsed time: %d hours, %d minutes, %d seconds\n"%(hours,mins,secs)+"Current date & time: %s"%(time.strftime("%c"))
		res_norm  = prblm.get_rnorm()
		grad_norm = prblm.get_gnorm()
		if(self.logger): self.logger.addToLog(msg)
		#Stop by number of iterations
		if((self.niter > 0) and (iter >= self.niter)):
			stop = True
			msg  =  "Terminate: maximum number of iterations reached\n"
			if(verbose): print(msg)
			if(self.logger): self.logger.addToLog(msg)
			return stop
		if((self.maxfevals > 0) and (prblm.get_fevals() >= self.maxfevals)):
			stop = True
			msg  =  "Terminate: maximum number of evaluations\n"
			if(verbose): print(msg)
			if(self.logger): self.logger.addToLog(msg)
			return stop
		if((self.maxhours > 0.) and (elapsed_time >= self.maxhours)):
			stop = True
			msg  =  "Terminate: maximum number hours reached %s\n"%(elapsed_time)
			if(verbose): print(msg)
			if(self.logger): self.logger.addToLog(msg)
			return stop
		if(res_norm < self.tolr):
			stop = True
			msg  =  "Terminate: minimum residual tolerance reached %s\n"%(res_norm)
			if(verbose): print(msg)
			if(self.logger): self.logger.addToLog(msg)
			return stop
		if(grad_norm < self.tolg):
			stop = True
			msg  =  "Terminate: minimum gradient tolerance reached %s\n"%(grad_norm)
			if(verbose): print(msg)
			if(self.logger): self.logger.addToLog(msg)
			return stop
		if(self.tolobj!=None):
			if(prblm.get_obj() < self.tolobj):
				stop = True
				msg  =  "Terminate: objective function value tolerance of %s reached, objective function value %s\n"%(self.tolobj,prblm.get_obj_value())
				if(verbose): print(msg)
				if(self.logger): self.logger.addToLog(msg)
				return stop
		if(self.tolobjrel!=None):
			if(prblm.get_obj()/initial_obj_value < self.tolobjrel):
				stop = True
				msg  =  "Terminate: relative objective function value tolerance of %s reached,  relative objective function value %s\n"%(self.tolobjrel,prblm.get_obj_value()/prblm.initial_obj_value)
				if(verbose): print(msg)
				if(self.logger): self.logger.addToLog(msg)
				return stop
		return stop