#Module containing for truncated Newton solver class
import pySolver
import pyOperator as pyOp
import pyStepperCvSrch as StepperMT
import pyStopperBase as Stopper
import pyProblem as Prblm
import pySymLCGsolver as SymLCGsolver

import numpy as np
from math import isnan

import numpy as np
from math import isnan


class TNewtonsolver(pySolver.Solver):
    """Truncated Newton/Gauss-Newton solver object"""

    def __init__(self,stoppr,niter_max,HessianOp,stepper=None,niter_min=None,warm_start=True,Newton_prefix=None,logger=None):
        """
		   Constructor for Truncated-Newton Solver.
		   stoppr     = [no default] - stopper class; Stopper object necessary to terminate the solver
		   niter_max  = [no default] - int; Maximum number of iterations for solving Newton system when starting with a zero initial model
		   HessianOp  = [no default] - operator class; Operator to apply Hessian matrix onto model vector. The operator must contain a set_background function to set model vector on which the Hessian is computed. Note the symmetric solver will be used. Hence, this operator must be symmetric.
		   stepper    = [CvSrch] - stepper class; Stepper object necessary to perform line-search step
		   niter_min  = [niter_max] - int; Number of iterations for solving Newton system when linear inversion starts from previous search direction
		   warm_start = [False] - boolean; If True, the linear Hessian inversion is started from the previous search direction if aligned with the current gradient
		   logger 	  = [None] - logger class; Logger object to save inversion information at runtime
        """
		#Defining stopper
		self.stoppr=stoppr #Stopper for non-linear problem
		#Setting maximum and minimum number of iterations
		self.niter_max=niter_max
		self.niter_min=niter_max
		#Setting linear inversion iterations
		if(niter_min!=None):
			if (niter_min<=niter_max):
				raise ValueError("ERROR! niter_min of %s must be smaller or equal than niter_max of %s."%(niter_min,niter_max))
			self.niter_min=niter_min
		#Defining stepper object
		if(stepper != None):
			self.stepper=stepper
		else:
			self.stepper=StepperMT.CvSrchStep()
		#Warm starts requested?
		self.warm_start=warm_start
		#Hessian operator
		if(HessianOp != None):
			if(not "set_background" in dir(HessianOp):
				raise AttributeError("ERROR! Hessian operator must have a set_background function.")
		#Setting linear solver for solving Newton system and problem class
		StopLin  = Stopper.BasicStopper(niter=self.niter_max)
		self.lin_solver = SymLCGsolver.SymLCGsolver(Stop)
		self.NewtonPrblm = Prblm.ProblemLinearSymmetric(HessianOp.domain.clone(),HessianOp.domain.clone(),HessianOp)
		return

	def run(self,prblm,verbose=False,restart=False):
		"""Running Truncated Newton solver"""

		return
