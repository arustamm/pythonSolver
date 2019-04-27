#Module containing CvSrch Stepper definition (based on the code by Dianne O'Leary July 1991)
#Originally published by More and Thuente (1994) "Line Search Algorithms with Guaranteed Sufficient Decrease"
#Note that it will modify the search direction if the model hits the inversion bounds

import pyStepper
import numpy as np
from math import isnan
from copy import deepcopy

class CvSrchStep(pyStepper.Stepper):
	"""
	   Originally published by More and Thuente (1994) "Line Search Algorithms with Guaranteed Sufficient Decrease"
	   CvSrch stepper (from Dianne O'Leary's code):


	   THE PURPOSE OF CVSRCH IS TO FIND A STEP WHICH SATISFIES
	   A SUFFICIENT DECREASE CONDITION AND A CURVATURE CONDITION.

	   AT EACH STAGE THE SUBROUTINE UPDATES AN INTERVAL OF
	   UNCERTAINTY WITH ENDPOINTS STX AND STY. THE INTERVAL OF
	   UNCERTAINTY IS INITIALLY CHOSEN SO THAT IT CONTAINS A
	   MINIMIZER OF THE MODIFIED FUNCTION

			F(X+STP*S) - F(X) - FTOL*STP*(GRADF(X)'S).

	   IF A STEP IS OBTAINED FOR WHICH THE MODIFIED FUNCTION
	   HAS A NONPOSITIVE FUNCTION VALUE AND NONNEGATIVE DERIVATIVE,
	   THEN THE INTERVAL OF UNCERTAINTY IS CHOSEN SO THAT IT
	   CONTAINS A MINIMIZER OF F(X+STP*S).

	   THE ALGORITHM IS DESIGNED TO FIND A STEP WHICH SATISFIES
	   THE SUFFICIENT DECREASE CONDITION

			 F(X+STP*S) .LE. F(X) + FTOL*STP*(GRADF(X)'S),

	   AND THE CURVATURE CONDITION

			 ABS(GRADF(X+STP*S)'S)) .LE. GTOL*ABS(GRADF(X)'S).

	   IF FTOL IS LESS THAN GTOL AND IF, FOR EXAMPLE, THE FUNCTION
	   IS BOUNDED BELOW, THEN THERE IS ALWAYS A STEP WHICH SATISFIES
	   BOTH CONDITIONS. IF NO STEP CAN BE FOUND WHICH SATISFIES BOTH
	   CONDITIONS, THEN THE ALGORITHM USUALLY STOPS WHEN ROUNDING
	   ERRORS PREVENT FURTHER PROGRESS. IN THIS CASE STP ONLY
	   SATISFIES THE SUFFICIENT DECREASE CONDITION.
	"""

	def __init__(self,alpha=0.0,xtol=1.0e-16,ftol=1.0e-4,gtol=0.95,alpha_min=1.0e-10,alpha_max=1.0e10,maxfev=20,xtrapf=4.0,delta=0.66):
		"""
		   CvSrch constructor:
		   alpha 		 = [0.] - float; Initial step-length guess
		   xtol  	 	 = [1e-16] - float; Relative width tolerance: convergence is reached if width falls below xtol * maximum step size.
		   ftol  	 	 = [1e-16] - float; c1 value to test first Wolfe condition (should be between 0 and 1)
		   gtol  	 	 = [0.95] - float; c2 value to test second Wolfe condition (should be between c1 or ftol and 1). For Quasi-Newton (e.g., L-BFGS) choose default. Otherwise, for other methods (e.g., NLCG) choose 0.1
		   alpha_min  	 = [1e-10] - float; Minimum step length value of the step length interval
		   alpha_max  	 = [1e10] - float; Maximum step length value of the step length interval
		   maxfev  	     = [20] - int; Maximum number of function evaluation to step length
		   xtrapf  	     = [4.0] - float; Scaling factor to find right limit of uncertainty interval
		   delta  	     = [0.66] - float; Value to force sufficient decrease of interval size on successive iterations. Should be a positive value less than 1.
		"""
		self.alpha = alpha #Initial step length guess
		self.xtol = xtol
		self.ftol = ftol
		self.gtol = gtol
		self.alpha_min = alpha_min
		self.alpha_max = alpha_max
		self.maxfev = maxfev
		self.xtrapf = xtrapf
		self.delta = delta
		self.zero = 10**(np.floor(np.log10(np.abs(float(np.finfo(np.float64).tiny))))+2) #Check for avoid Overflow or Underflow
		#Checking stepper parameters
		if(xtol < 0.):
			raise ValueError("ERROR! xtol must be greater than 0.0, current value %s"%(xtol))
		if(ftol < 0. or ftol >= 1.):
			raise ValueError("ERROR! ftol must be greater than 0.0 and smaller than 1.0, current value %s"%(ftol))
		if(gtol < 0. or ftol >= 1.):
			raise ValueError("ERROR! gtol must be greater than 0.0 and smaller than 1.0, current value %s"%(gtol))
		if(alpha_min < 0.):
			raise ValueError("ERROR! alpha_min must be greater than 0.0, current value %s"%(alpha_min))
		if(xtrapf < 0.):
			raise ValueError("ERROR! xtrapf must be greater than 0.0, current value %s"%(xtrapf))
		if(delta < 0. or delta >= 1.):
			raise ValueError("ERROR! delta must be greater than 0.0 and smaller than 1.0, current value %s"%(delta))
		if(maxfev < 0):
			raise ValueError("ERROR! maxfev must be greater than 0, current value %s"%(maxfev))
		if(alpha_max < alpha_min):
			raise ValueError("ERROR! alpha_max must be greater than alpha_min, current values: alpha_min=%s; alpha_max=%s"%(alpha_min,alpha_max))
		return

	def cstep(self,stx,fx,dx,sty,fy,dy,stp,fp,dp,brackt,stpmin,stpmax,logger):
		"""
			Modified Cstep function (from the code by Dianne O'Leary July 1991):
			The purpose of cstep is to compute a safeguarded step for
			a linesearch and to update an interval of uncertainty for
			a minimizer of the function.

			The parameter stx contains the step with the least function
			value. The parameter stp contains the current step. It is
			assumed that the derivative at stx is negative in the
			direction of the step. If brackt is set true then a
			minimizer has been bracketed in an interval of uncertainty
			with endpoints stx and sty.
			The subroutine statement is

			subroutine cstep(stx,fx,dx,sty,fy,dy,stp,fp,dp,brackt,stpmin,stpmax,info)

			where

			stx, fx, and dx are variables which specify the step,
			the function, and the derivative at the best step obtained
			so far. The derivative must be negative in the direction
			of the step, that is, dx and stp-stx must have opposite
			signs. On output these parameters are updated appropriately.

			sty, fy, and dy are variables which specify the step,
			the function, and the derivative at the other endpoint of
			the interval of uncertainty. On output these parameters are
			updated appropriately.

			stp, fp, and dp are variables which specify the step,
			the function, and the derivative at the current step.
			If brackt is set true then on input stp must be
			between stx and sty. On output stp is set to the new step.

			brackt is a logical variable which specifies if a minimizer
			has been bracketed. If the minimizer has not been bracketed
			then on input brackt must be set false. If the minimizer
			is bracketed then on output brackt is set true.

			stpmin and stpmax are input variables which specify lower
			and upper bounds for the step.

			info is an integer output variable set as follows:
			If info = True, then the step has been computed
			according to one of the five cases below. Otherwise
			info = False, and this indicates improper input parameters.
		"""

		success = False #which is info

		#Check the input parameters for errors.
		if ((brackt and (stp <= np.minimum(stx,sty) or stp >= np.maximum(stx,sty))) or dx*(stp-stx) >= 0.0 or stpmax < stpmin):
			if(logger): logger.addToLog("	Function cstep could find step and update interval of uncertainty!")
			return stx,fx,dx,sty,fy,dy,stp,fp,dp,brackt,success

		#Determine if the derivatives have opposite sign.
		sgnd = dp*(dx/np.abs(dx))

		#First case. A higher function value.
		#The minimum is bracketed. If the cubic step is closer
		#to stx than the quadratic step, the cubic step is taken,
		#else the average of the cubic and quadratic steps is taken.

		if (fp > fx):
			success = True
			bound = True
			theta = 3.0*(fx - fp)/(stp - stx) + dx + dp
			s = np.linalg.norm([theta,dx,dp],np.inf)
			gamma = s*np.sqrt((theta/s)*(theta/s) - (dx/s)*(dp/s));
			if (stp < stx):
				gamma = -gamma
			p = (gamma - dx) + theta
			q = ((gamma - dx) + gamma) + dp
			r = p/q
			stpc = stx + r*(stp - stx)
			stpq = stx + ((dx/((fx-fp)/(stp-stx)+dx))/2.0)*(stp - stx)
			if (np.abs(stpc-stx) < np.abs(stpq-stx)):
				stpf = stpc
			else:
				stpf = stpc + (stpq - stpc)/2.0
			brackt = True

		#Second case. A lower function value and derivatives of
		#opposite sign. The minimum is bracketed. If the cubic
		#step is closer to stx than the quadratic (secant) step,
		#the cubic step is taken, else the quadratic step is taken.

		elif (sgnd < 0.0):
			success = True
			bound = False
			theta = 3.0*(fx - fp)/(stp - stx) + dx + dp
			s = np.linalg.norm([theta,dx,dp],np.inf)
			gamma = s*np.sqrt((theta/s)*(theta/s) - (dx/s)*(dp/s))
			if (stp > stx):
				gamma = -gamma
			p = (gamma - dp) + theta
			q = ((gamma - dp) + gamma) + dx
			r = p/q
			stpc = stp + r*(stx - stp)
			stpq = stp + (dp/(dp-dx))*(stx - stp)
			if (np.abs(stpc-stp) > np.abs(stpq-stp)):
				stpf = stpc
			else:
				stpf = stpq
			brackt = True

		#Third case. A lower function value, derivatives of the
		#same sign, and the magnitude of the derivative decreases.
		#The cubic step is only used if the cubic tends to infinity
		#in the direction of the step or if the minimum of the cubic
		#is beyond stp. Otherwise the cubic step is defined to be
		#either stpmin or stpmax. The quadratic (secant) step is also
		#computed and if the minimum is bracketed then the the step
		#closest to stx is taken, else the step farthest away is taken.

		elif (np.abs(dp) < np.abs(dx)):
			success = True
			bound = True
			theta = 3.0*(fx - fp)/(stp - stx) + dx + dp
			s = np.linalg.norm([theta,dx,dp],np.inf)

			#The case gamma = 0 only arises if the cubic does not tend
			#to infinity in the direction of the step.

			gamma = s*np.sqrt(mp.maximum(0.,(theta/s)*(theta/s) - (dx/s)*(dp/s)))
			if (stp > stx):
				gamma = -gamma

			p = (gamma - dp) + theta
			q = (gamma + (dx - dp)) + gamma
			r = p/q
			if (r < 0.0 and gamma != 0.0):
				stpc = stp + r*(stx - stp)
			elif (stp > stx):
				stpc = stpmax
			else:
				stpc = stpmin
			stpq = stp + (dp/(dp-dx))*(stx - stp)
			if (brackt):
				if (np.abs(stp-stpc) < np.abs(stp-stpq)):
					stpf = stpc
				else:
					stpf = stpq
			else:
				if (np.abs(stp-stpc) > np.abs(stp-stpq)):
					stpf = stpc
				else:
					stpf = stpq

		#Fourth case. A lower function value, derivatives of the
		#same sign, and the magnitude of the derivative does
		#not decrease. If the minimum is not bracketed, the step
		#is either stpmin or stpmax, else the cubic step is taken.

		else:
			success = True
			bound = False
			if (brackt):
				theta = 3.0*(fp - fy)/(sty - stp) + dy + dp
				s = np.linalg.norm([theta,dx,dp],np.inf)
				gamma = s*np.sqrt((theta/s)*(theta/s) - (dy/s)*(dp/s))
				if (stp > sty):
					gamma = -gamma
				p = (gamma - dp) + theta
				q = ((gamma - dp) + gamma) + dy
				r = p/q
				stpc = stp + r*(sty - stp)
				stpf = stpc
			elif (stp > stx):
				stpf = stpmax
			else:
				stpf = stpmin

		#Update the interval of uncertainty. This update does not
		#depend on the new step or the case analysis above.

		if (fp > fx):
			sty = stp;
			fy = fp;
			dy = dp;
		else:
			if (sgnd < 0.0):
				sty = stx
				fy = fx
				dy = dx
			stx = stp
			fx = fp
			dx = dp

		#Compute the new step and safeguard it.
		stpf = np.minimum(stpmax,stpf);
		stpf = np.maximum(stpmin,stpf);
		stp = stpf;
		if (brackt & bound):
			if (sty > stx):
				stp = np.minimum(stx+self.delta*(sty-stx),stp);
			else:
				stp = np.maximum(stx+self.delta*(sty-stx),stp);


		return stx,fx,dx,sty,fy,dy,stp,fp,dp,brackt,success

	def run(self,prblm,modl,dmodl,logger=None):
		"""Method to apply CvSrch stepper"""
		#Writing to log file if any
		if(logger): logger.addToLog("CVSRCH STEPPER BY STEP-LENGTH BRACKETING")
		if(logger): logger.addToLog("xtol=%s ftol=%s gtol=%s alpha_min=%s alpha_max=%s maxfev=%s xtrapf=%s"%(self.xtol,self.ftol,self.gtol,self.alpha_min,self.alpha_max,self.maxfev,self.xtrapf))
		success=False
		#Obtain objective function for provided model
		phi_init=prblm.get_obj(modl)
		#Getting pointer to problem's gradient vector
		prblm_grad = prblm.get_grad(modl)
		dphi_init = prblm_grad.dot(dmodl)
		if(dphi_init > 0.0):
			if(logger): logger.addToLog("	Warning! Current search direction is not a descent one!")
			return self.alpha,success
		#Model temporary vector
		model_step = modl.clone()
		#Getting pointer to problem's model vector
		prblm_mdl = prblm.get_model()
		#Initial step length value
		alpha=deepcopy(self.alpha)
		#Estimating initial step length value
		if(alpha < self.zero):
			alpha = self.estimate_initial_guess(prblm,modl,dmodl,logger)
		if(logger): logger.addToLog("	initial-steplength=%s"%(alpha))

		#Initializing parameters
		p5 = 0.5
		cstep_success = True
		fev = 0
		width=self.alpha_max-self.alpha_min
		width1=2*width
		brackt = False
		stage1 = True
		dphi_test = self.ftol*dphi_init

		#The variables alphax, phix, dphix contain the values of the step, function, and directional derivative at the best step.
		#The variables alphay, phiy, dphiy contain the value of the step, function, and derivative at the other endpoint of the interval of uncertainty.
		#The variables alpha, phi_c, dphi_c contain the values of the step, function, and derivative at the current step.
		alphax = 0.0
		phix = phi_init
		dphix = dphi_init
		alphay = 0.0
		phiy = phi_init
		dphiy = dphi_init

		#Start testing iteration
		while(True):
			#Set the minimum and maximum steps to correspond to the present interval of uncertainty.
			if (brackt):
				alpha_int_min = np.minimum(alphax,alphay)
				alpha_int_max = np.maximum(alphax,alphay)
			else:
				alpha_int_min = alphax
				alpha_int_max = alpha + self.xtrapf*(alpha - alphax)

			#Force the step to be within the bounds alpha_max and alpha_min.
			alpha = np.maximum(alpha,self.alpha_min)
			alpha = np.minimum(alpha,self.alpha_max)

			#If an unusual termination is to occur then choose alpha be the lowest point obtained so far.
			if ((brackt and (alpha <= alpha_int_min or alpha >= alpha_int_max)) or fev >= self.maxfev-1 or (not cstep_success) or (brackt and alpha_int_max-alpha_int_min <= self.xtol*alpha_int_max)):
				if(logger): logger.addToLog("	Unusual termination is to occur. Setting alpha to be the lowest point obtained so far.")
				alpha = alphax

			#Evaluate the function and gradient at alpha and compute the directional derivative.
			if(logger): logger.addToLog("	Current testing point (alpha=%s): m_current+alpha*dm"%(alpha))
			model_step.copy(modl)
			model_step.scaleAdd(dmodl,sc2=alpha)
			#Checking if model parameters hit the bounds
			prblm.set_model(model_step)
			#Projecting model onto the bounds (if any)
			if("bounds" in dir(prblm)): prblm.bounds.apply(model_step)
			if(prblm_mdl.isDifferent(model_step)):
				#Model hit bounds
				msg="	Model hit provided bounds. Projecting it onto them."
				if(logger): logger.addToLog(msg)
			phi_alpha=prblm.get_obj(model_step)
			fev+=1
			#Checking if a NaN is encountered
			if(isnan(phi_alpha)):
				if(logger): logger.addToLog("		!!!Objective function is NaN! Stepper unsuccessful!!!")
				prblm.set_model(modl)
				break
			if(logger): logger.addToLog("		Objective function value of %s (feval = %s)"%(phi_alpha,prblm.get_fevals()))
			prblm_grad = prblm.get_grad(model_step)
			dphi_alpha = prblm_grad.dot(dmodl)
			phi_test1 = phi_init + alpha * dphi_test

			#Test for convergence
			if ((brackt and (alpha <= alpha_int_min or alpha >= alpha_int_max)) or (not cstep_success)):
				if(logger): logger.addToLog("	Rounding errors prevent further progress. There may not be a step which satisfies the sufficient decrease and curvature conditions. Tolerances may be too small.")
				break
			if (alpha == self.alpha_max and phi_alpha <= phi_test1 and dphi_alpha <= dphi_test):
				if(logger): logger.addToLog("	The step-length value is at the upper bound (alpha_max) of %s."%(self.alpha_max))
				break
			if (alpha == self.alpha_min and (phi_alpha > phi_test1 or dphi_alpha >= dphi_test)):
				if(logger): logger.addToLog("	The step-length value is at the lower bound (alpha_min) of %s."%(self.alpha_min))
				break
			if (fev >= self.maxfev):
				if(logger): logger.addToLog("	Number of objective function evalutation reached maxfev of %s."%(self.maxfev))
				break
			if (brackt and alpha_int_max-alpha_int_min <= self.xtol*alpha_int_max):
				if(logger): logger.addToLog("	Relative width of the interval of uncertainty is at most xtol of %s."%(self.xtol))
				break
			if (phi_alpha <= phi_test1 and abs(dphi_test) <= self.gtol*(-dphi_init) and phi_alpha < phi_init):
				success=True
				if(logger): logger.addToLog("	The sufficient decrease condition and the directional derivative condition hold (i.e., Strong Wolfe conditions met).\n	Stepper successuful for step length value of %s and objective function of %s (feval = %s) "%(alpha,phi_alpha,prblm.get_fevals()))
				break

			#In the first stage we seek a step for which the modified function has a nonpositive value and nonnegative derivative.
			if (stage1 and (phi_alpha <= phi_test1) and (dphi_alpha >= np.minimum(self.ftol,self.gtol)*dphi_init)):
				stage1 = False

			#A modified function is used to predict the step only if
			#we have not obtained a step for which the modified
			#function has a nonpositive function value and nonnegative
			#derivative, and if a lower function value has been
			#obtained but the decrease is not sufficient.

			if (stage1 and (phi_alpha <= phix) and (phi_alpha > phi_test1)):
				#Define the modified function and derivative values.
				phim = phi_alpha - alpha*dphi_test
				phixm = phix - alphax*dphi_test
				phiym = phiy - alphay*dphi_test
				dphim = dphi_alpha - dphi_test
				dphixm = dphix - dphi_test
				dphiym = dphiy - dphi_test

				#Call cstep to update the interval of uncertainty and to compute the new step.
				[alphax,phixm,dphixm,alphay,phiym,dphiym,alpha,phim,dphim,brackt,cstep_success] = self.cstep(alphax,phixm,dphixm,alphay,phiym,dphiym,alpha,phim,dphim,brackt,alpha_int_min,alpha_int_max,logger)

				#Reset the function and gradient values for phi.
				phix = phixm + alphax*dphi_test
				phiy = phiym + alphay*dphi_test
				dphix = dphixm + dphi_test
				dphiy = dphiym + dphi_test

			else:
				#Call cstep to update the interval of uncertainty and to compute the new step.
				[alphax,phix,dphix,alphay,phiy,dphiy,alpha,phi_alpha,dphi_alpha,brackt,cstep_success] = self.cstep(alphax,phix,dphix,alphay,phiy,dphiy,alpha,phi_alpha,dphi_alpha,brackt,alpha_int_min,alpha_int_max,logger)

			#Force a sufficient decrease in the size of the interval of uncertainty.
			if (brackt):
				if (abs(alphay-alphax) >= self.delta*width1):
					alpha = alphax + p5*(alphay - alphax)
				width1 = width
				width = abs(alphay-alphax)

			#End of iteration.


		if(success):
			#Line search has finished, update model
			self.alpha=deepcopy(alpha)
			modl.copy(model_step)

		#Delete temporary vectors
		del model_step
		return alpha,success
