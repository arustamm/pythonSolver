#Module containing containing Iterative Soft-Thresholding with Cooling solver for linear L1-regularized problems
from math import isnan
import pySolver


class ISTCsolver(pySolver.Solver):
	"""ISTC solver to solve: linear problem 1/2*| y - Am |_2 + lambda*| m |_1"""

	def __init__(self,stoppr,inner_it,cooling_start,cooling_end,steepest=False,logger=None):
		"""
		   Constructor for ISTC Solver
		"""
		#Defining stopper object
		self.stoppr=stoppr
		#Logger object to write on log file
		self.logger=logger
		#Overwriting logger of the Stopper object
		self.stoppr.logger=self.logger
		#Setting defaults for saving results
		self.setDefaults()

		#ISTC parameters
		self.inner_it=inner_it #number of inner iterations, the outer iterations are taken care by the stopper
		# cooling_start and cooling_end are numbers between 0 and 1 such that cooling_start <= cooling_end
		if(not (0<=cooling_start<=1) or not (0<=cooling_end<=1) or (cooling_end<cooling_start)): raise ValueError("ERROR! cooling_start and cooling_end must be within [0,1] interval and cooling_start <= cooling_end")
		self.cooling_start=cooling_start # start of cooling continuation as fraction of size of sorted array |A'y|
		self.cooling_end=cooling_end # end of cooling continuation as fraction of size of sorted array |A'y|
		return

	def __del__(self):
		"""Default destructor"""
		return


	def run(self,prblm,verbose=False,restart=False):
		"""Running ISTC solver"""
		if(not restart):
			msg="ITERATIVE SOFT-THRESHOLDING WITH COOLING SOLVER log file\n"
			#Printing restart folder
			msg+="Restart folder: %s\n"%(self.restart.restart_folder)
			if(verbose): print(msg.replace("log file",""))
			if(self.logger): self.logger.addToLog(msg)

			#Setting internal vectors (model, search direction, and previous gradient vectors)
			prblm_mdl=prblm.get_model()
			istc_mdl = prblm_mdl.clone()

			#Inversion always starts from m = 0 (I need to understand if it is possible to start from m different than 0)
			istc_mdl.zero()	# modl = 0
			#Other internal variables
			iter = 0
			#Computing cooling parameters
			prblm_grad=prblm.get_grad(istc_mdl)
			if(not instance(prblm_grad,))
		else:
			#Retrieving parameters and vectors to restart the solver
			msg="Restarting previous solver run from: %s"%(self.restart.restart_folder)
			if(verbose): print(msg)
			if(self.logger): self.logger.addToLog(msg)
			self.restart.read_restart()

		#Common variables unrelated to restart
		success = True
		#Outer iteration loop
		while True:


		#Writing last inverted model
		self.save_results(iter,prblm,force_save=True,force_write=True)
		if(self.logger): self.logger.addToLog("ITERATIVE SOFT-THRESHOLDING WITH COOLING SOLVER log file end")
		#Clear restart object
		self.restart.clear_restart()

		return
