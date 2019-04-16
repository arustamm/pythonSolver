#Module containing Parabolic Stepper definition
#Note that it will modify the search direction if the model hits the inversion bounds
import pyStepper
import numpy as np
from math import isnan
from copy import deepcopy

class ParabolicStep(pyStepper.Stepper):
	"""Parabolic Stepper class with three-point interpolation"""

	def __init__(self, c1=1.0, c2=2.0, ntry=10, alpha=0., alpha_scale_min=1.0e-10, alpha_scale_max=1000.00, shrink=0.25, eval_parab=True):
		"""
		   Constructor for parabolic stepper with three-point interpolation:
		   c1  		   	   = [1.0] - float; Scaling factor of first search point (i.e., m1 = c1*alpha*dm + m_current)
		   c2  		   	   = [2.0] - float; Scaling factor of first search point (i.e., m2 = c2*alpha*dm + m_current)
		   ntry  	   	   = [10] - integer; Number of trials for finding the step length
		   alpha 		   = [0.] - float; Initial step-length guess
		   alpha_scale_min = [1.0e-10] - float; Minimum scaling factor (c_optimal) for step-length allowed
		   alpha_scale_max = [1000.00] - float; Maximum scaling factor (c_optimal) for step-length allowed
		   shrink 		   = [0.25] - float; Shrinking factor if step length is not found at a given trial
		   eval_parab 	   = [True] - boolean; Force parabola minimum to be computed. If False, the best point will be chosen from c1 or c2 and the parabola minimum is computed if necessary
		"""
		self.c1=c1 								#Scaling for first tested point
		self.c2=c2 								#Scaling for second tested point
		self.ntry=ntry 							#Number of total trials before re-estimating initial alpha value
		self.alpha=alpha 						#Initial step length guess
		self.alpha_scale_min=alpha_scale_min	#Maximum scaling value for the step length
		self.alpha_scale_max=alpha_scale_max	#Minimum scaling value for the step length
		self.shrink=shrink						#Shrinking scaling factor if trial is unsuccessful
		self.zero = 10**(np.floor(np.log10(np.abs(float(np.finfo(np.float64).tiny))))+2) #Check for avoid Overflow or Underflow
		self.eval_parab = eval_parab
		return


	def run(self,prblm,modl,dmodl,logger=None):
		"""Method to apply parabolic stepper"""
		#Writing to log file if any
		if(logger): logger.addToLog("PARABOLIC STEPPER USING THREE-POINT INTERPOLATION")
		if(logger): logger.addToLog("c1=%s c2=%s ntry=%s steplength-scaling-min=%s steplength-scaling-max=%s shrinking-factor=%s"%(self.c1,self.c2,self.ntry,self.alpha_scale_min,self.alpha_scale_max,self.shrink))
		success=False
		#Obtain objective function for provided model
		obj0=prblm.get_obj(modl)
		#Model temporary vector
		model_step = modl.clone()
		#Getting pointer to problem's model vector
		prblm_mdl = prblm.get_model()
		#Initial step length value
		alpha=deepcopy(self.alpha)
		itry=1
		total_trials = deepcopy(self.ntry)
		if(alpha != 0.):
			#If initial step length is different than zero, we test twice in case we need to re-estimate initial alpha
			total_trials *=2
		while (itry <= total_trials):
			#Writing info to log file
			if(logger): logger.addToLog("	trial number: %s"%(itry))
			if(logger): logger.addToLog("	initial-steplength=%s"%(alpha))
			#Find the first guess as if the problem was linear (Tangent method)
			if((itry==self.ntry) or (alpha < self.zero)):
				try:
					dres=prblm.get_dres(modl,dmodl)
				except NotImplementedError:
					if(logger): logger.addToLog("WARNING! dresf not implemented; stepper could not find a new step-length initial guess (i.e., line search unsuccessful)")
					break
				res=prblm.get_res(modl)
				dres_res=res.dot(dres)
				dres_dres=dres.dot(dres)
				if(dres_dres == 0.):
					if(logger): logger.addToLog("	!!!Gradient in the null space of linear forward operator!!!")
					alpha = 1.0
				else:
					alpha = -dres_res/dres_dres
				if(logger): logger.addToLog("	Guessing linear step length of: %s"%(alpha))
			#Test values of objective function for two scaled versions of the step length
			#Testing c1 scale
			if(logger): logger.addToLog("	Testing point (c1=%s): m_current+c1*alpha*dm"%(self.c1))
			model_step.copy(modl)
			model_step.scaleAdd(dmodl,sc2=self.c1*alpha)
			#Checking if model parameters hit the bounds
			prblm.set_model(model_step)
			#Projecting model onto the bounds (if any)
			if("bounds" in dir(prblm)): prblm.bounds.apply(model_step)
			if(prblm_mdl.isDifferent(model_step)):
				#Model hit bounds
				msg="	Model hit provided bounds. Projecting it onto them."
				if(logger): logger.addToLog(msg)
			obj1=prblm.get_obj(model_step)
			#Copying residuals for point c1
			res_prblm=prblm.get_res(model_step)
			res1=res_prblm.clone()
			if(logger): logger.addToLog("		Objective function value of %s"%(obj1))
			#Checking if a NaN is encountered in any of the two tested points
			if(isnan(obj1)):
				if(logger): logger.addToLog("		!!!Problem with step length and objective function!!!")
				if(itry>=self.ntry):
					if(logger): logger.addToLog("		!!!Check problem definition or change solver!!!")
					#Setting model to current one and resetting initial step length value
					alpha = 0.0
					self.alpha = 0.0
					prblm.set_model(modl)
					break
				else:
					if(logger): logger.addToLog("		!!!Guessing linear step length to try to solve problem!!!")
					itry=self.ntry #To not repeat computation of linear guess
					continue
			#Testing c2 scale
			msg = "	Testing point (c2=%s): m_current+c2*alpha*dm"%(self.c2)
			if(logger): logger.addToLog(msg)
			model_step.copy(modl)
			model_step.scaleAdd(dmodl,sc2=self.c2*alpha)
			#Checking if model parameters hit the bounds
			prblm.set_model(model_step)
			#Projecting model onto the bounds (if any)
			if("bounds" in dir(prblm)): prblm.bounds.apply(model_step)
			if(prblm_mdl.isDifferent(model_step)):
				#Model hit bounds
				msg="	Model hit provided bounds. Projecting it onto them."
				if(logger): logger.addToLog(msg)
			obj2=prblm.get_obj(model_step)
			#Copying residuals for point c1
			res_prblm=prblm.get_res(model_step)
			res2=res_prblm.clone()
			if(logger): logger.addToLog("		Objective function value of %s"%(obj2))
			#Checking for NaN
			if(isnan(obj2)):
				if(logger): logger.addToLog("		!!!Problem with step length and objective function!!!")
				if(itry>=self.ntry):
					if(logger): logger.addToLog("		!!!Check problem definition or change solver!!!")
					#Setting model to current one and resetting initial step length value
					alpha = 0.0
					self.alpha = 0.0
					prblm.set_model(modl)
					break
				else:
					if(logger): logger.addToLog("		!!!Guessing linear step length to try to solve problem!!!")
					itry=self.ntry #To not repeat computation of linear guess
					continue
			#Checking if parabolic point is necessary or not
			if(not self.eval_parab):
				#Setting third point to infinity
				obj3=np.inf
				#Check which one is the best step length
				msg="\n	As requested, parabola minimum was not evaluated!"
				if (obj1<obj0 and obj1<obj2 and obj1<obj3):
					success = True
					alpha *= self.c1
					if(logger): logger.addToLog("	c1 best step-length value of: %s"%(alpha)+msg)
					break
				elif (obj2<obj0 and obj2<obj1 and obj2<obj3):
					success = True
					alpha *= self.c2
					if(logger): logger.addToLog("	c2 best step-length value of: %s"%(alpha)+msg)
					break
			#If points lay on a horizontal line pick minimum alpha set by user
			if(obj0 == obj1 == obj2 or (self.c2*(obj1-obj0) + self.c1*(obj0-obj2)) == 0.):
				step_scale = self.alpha_scale_min
				if(logger): logger.addToLog("	Two testing points on a line: cannot fit a parabola, using minimum step-length of %s"%(step_scale*alpha))
			else:
			#Otherwise, find the optimal parabolic step length
				step_scale = 0.5*(self.c2*self.c2*(obj1-obj0) + self.c1*self.c1*(obj0-obj2))/(self.c2*(obj1-obj0) + self.c1*(obj0-obj2))
				if(logger): logger.addToLog("	Testing point (c_opt=%s): m_current+c_opt*alpha*dm (parabola minimum)"%(step_scale))
			#If step length negative, re-evaluate points
			if(step_scale*alpha < 0.):
				if(logger): logger.addToLog("	Encountered a negative step-length value: %s; Shrinking step-length value."%(step_scale*alpha))
				#Shrink line search
				alpha *= self.shrink
				itry+=1
				continue
			#Clipping the step-length scale
			if step_scale < self.alpha_scale_min:
				if(logger): logger.addToLog("	!!! step-length scale of %s smaller than provided lower bound. Clipping its value to bound value of %s !!!"%(step_scale,self.alpha_scale_min))
				step_scale = self.alpha_scale_min
			elif step_scale > self.alpha_scale_max:
				if(logger): logger.addToLog("	!!! step-length scale of %s greater than provided upper bound. Clipping its value to bound value of %s !!!"%(step_scale,self.alpha_scale_max))
				step_scale = self.alpha_scale_max

			#Testing parabolic scale
			#Compute new objective function at the minimum of the parabolic approximation
			model_step.copy(modl)
			model_step.scaleAdd(dmodl,sc2=step_scale*alpha)
			#Checking if model parameters hit the bounds
			prblm.set_model(model_step)
			#Projecting model onto the bounds (if any)
			if("bounds" in dir(prblm)): prblm.bounds.apply(model_step)
			if(prblm_mdl.isDifferent(model_step)):
				#Model hit bounds
				msg="	Model hit provided bounds. Projecting it onto them."
				if(logger): logger.addToLog(msg)
			obj3=prblm.get_obj(model_step)
			if(logger): logger.addToLog("		Objective function value of %s"%(obj3))

			#Writing info to log file
			if(logger): logger.addToLog("	Initial objective function value: %s, Objective function at c1*alpha*dm: %s, Objective function at c2*alpha*dm: %s, Objective function at parabola minimum: %s"%(obj0,obj1,obj2,obj3))
			itry+=1

			#Check which one is the best step length
			if (obj1<obj0 and obj1<obj2 and obj1<obj3):
				success = True
				alpha *= self.c1
				if(logger): logger.addToLog("	c1 best step-length value of: %s"%(alpha))
				break
			elif (obj2<obj0 and obj2<obj1 and obj2<obj3):
				success = True
				alpha *= self.c2
				if(logger): logger.addToLog("	c2 best step-length value of: %s"%(alpha))
				break
			elif (obj3<obj0 and obj3<=obj1 and obj3<=obj2):
				success = True
				alpha *= step_scale
				if(logger): logger.addToLog("	parabola minimum best step-length value of: %s"%(alpha))
				break
			else:
				#Shrink line search
				alpha *= self.shrink
				if(logger): logger.addToLog("	Shrinking search direction")

		if(success):
			#Line search has finished, update model
			self.alpha=deepcopy(alpha)
			model_step.copy(modl) # model_step = m_current
			model_step.scaleAdd(dmodl,sc2=self.alpha)
			#Projecting model onto the bounds (if any)
			if("bounds" in dir(prblm)): prblm.bounds.apply(model_step)
			if(prblm_mdl.isDifferent(model_step)):
				#Computing true scaled search direction dm = m_new_clipped - m_current
				dmodl.copy(model_step)
				dmodl.scaleAdd(modl,1.0,-1.0)
				#Scaled by the inverse of the step length
				dmodl.scale(1.0/self.alpha)
			#Setting model and residual vectors to c1 or c2 point if parabola minimum is not picked
			prblm.set_model(model_step)
			if (obj1<obj0 and obj1<obj2 and obj1<obj3):
				prblm.set_residual(res1)
			elif (obj2<obj0 and obj2<obj1 and obj2<obj3):
				prblm.set_residual(res2)
			modl.copy(model_step)
		#Delete temporary vectors
		del model_step, res1, res2
		return alpha,success


class ParabolicStepConst(pyStepper.Stepper):
	"""Parabolic Stepper class assuming constant local curvature"""

	def __init__(self, c1=1.0, ntry=10, alpha=0., alpha_scale_min=1.0e-10, alpha_scale_max=1000.00, shrink=0.25):
		"""
		   Constructor for parabolic stepper assuming constant local curvature:
		   c1  		   	   = [1.0] - float; Scaling factor of the search point (i.e., m1 = c1*alpha*dm + m_current)
		   ntry  	   	   = [10] - integer; Number of trials for finding the step length
		   alpha 		   = [0.] - float; Initial step-length guess
		   alpha_scale_min = [1.0e-10] - float; Minimum scaling factor (c_optimal) for step-length allowed
		   alpha_scale_max = [1000.00] - float; Maximum scaling factor (c_optimal) for step-length allowed
		   shrink 		   = [0.25] - float; Shrinking factor if step length is not found at a given trial
		"""
		self.c1=c1 								#Scaling for first tested point
		self.ntry=ntry 							#Number of total trials before re-estimating initial alpha value
		self.alpha=alpha 						#Initial step length guess
		self.alpha_scale_min=alpha_scale_min	#Maximum scaling value for the step length
		self.alpha_scale_max=alpha_scale_max	#Minimum scaling value for the step length
		self.shrink=shrink						#Shrinking scaling factor if trial is unsuccessful
		self.zero = 10**(np.floor(np.log10(np.abs(float(np.finfo(np.float64).tiny))))+2) #Check for avoid Overflow or Underflow
		return

	def run(self,prblm,modl,dmodl,logger=None):
		"""Method to apply parabolic stepper"""
		#Writing to log file if any
		if(logger): logger.addToLog("PARABOLIC STEPPER ASSUMING CONSTANT LOCAL CURVATURE")
		if(logger): logger.addToLog("c1=%s ntry=%s steplength-scaling-min=%s steplength-scaling-max=%s shrinking-factor=%s"%(self.c1,self.ntry,self.alpha_scale_min,self.alpha_scale_max,self.shrink))
		success=False
		#Obtain objective function for provided model
		obj0=prblm.get_obj(modl)
		#Model temporary vector
		model_step = modl.clone()
		#Getting pointer to problem's model vector
		prblm_mdl = prblm.get_model()
		#Getting pointer to problem's gradient vector
		prblm_grad = prblm.get_grad(prblm_mdl)
		#Initial step length value
		alpha=deepcopy(self.alpha)
		itry=1
		total_trials = deepcopy(self.ntry)
		if(alpha != 0.):
			#If initial step length is different than zero, we test twice in case we need to re-estimate initial alpha
			total_trials *=2
		while (itry <= total_trials):
			#Writing info to log file
			if(logger): logger.addToLog("	trial number: %s"%(itry))
			if(logger): logger.addToLog("	initial-steplength=%s"%(alpha))
			#Find the first guess as if the problem was linear (Tangent method)
			if((itry==self.ntry) or (alpha < self.zero)):
				try:
					dres=prblm.get_dres(modl,dmodl)
				except NotImplementedError:
					if(logger): logger.addToLog("WARNING! dresf not implemented; stepper could not find a new step-length initial guess (i.e., line search unsuccessful)")
					break
				res=prblm.get_res(modl)
				dres_res=res.dot(dres)
				dres_dres=dres.dot(dres)
				if(dres_dres == 0.):
					if(logger): logger.addToLog("	!!!Gradient in the null space of linear forward operator!!!")
					alpha = 1.0
				else:
					alpha = -dres_res/dres_dres
				if(logger): logger.addToLog("	Guessing linear step length of: %s"%(alpha))
			#Test values of objective function for two scaled versions of the step length
			#Testing c1 scale
			if(logger): logger.addToLog("	Testing point (c1=%s): m_current+c1*alpha*dm"%(self.c1))
			model_step.copy(modl)
			model_step.scaleAdd(dmodl,sc2=self.c1*alpha)
			#Checking if model parameters hit the bounds
			prblm.set_model(model_step)
			#Projecting model onto the bounds (if any) and rotate search direction
			if("bounds" in dir(prblm)): prblm.bounds.apply(model_step)
			if(prblm_mdl.isDifferent(model_step)):
				#Model hit bounds
				msg="	Model hit provided bounds. Projecting it onto them."
				#Computing true scaled search direction dm = m_new_clipped - m_current
				dmodl.copy(model_step)
				dmodl.scaleAdd(modl,1.0,-1.0)
				#Scaled by the inverse of the step length
				dmodl.scale(1.0/self.c1*alpha)
			obj1=prblm.get_obj(model_step)
			#Copying residuals for point c1
			res_prblm=prblm.get_res(model_step)
			res1=res_prblm.clone()
			if(logger): logger.addToLog("		Objective function value of %s"%(obj1))
			#Checking if a NaN is encountered in any of the two tested points
			if(isnan(obj1)):
				if(logger): logger.addToLog("		!!!Problem with step length and objective function!!!")
				if(itry>=self.ntry):
					if(logger): logger.addToLog("		!!!Check problem definition or change solver!!!")
					#Setting model to current one and resetting initial step length value
					alpha = 0.0
					self.alpha = 0.0
					prblm.set_model(modl)
					break
				else:
					if(logger): logger.addToLog("		!!!Guessing linear step length to try to solve problem!!!")
					itry=self.ntry #To not repeat computation of linear guess
					continue
			#Computing local constant curvature
			phi_der = prblm_grad.dot(dmodl) #First derivative of the objective function with respect to alpha
			c = 2.0 * ((obj1-obj0)/(self.c1*alpha*self.c1*alpha) - phi_der/(self.c1*alpha))
			#Checking the curvature value
			if(c <= 0.):
				#Shrink line search
				alpha *= self.shrink
				if(logger): logger.addToLog("	Estimated a negative curvature of %s. Shrinking search direction"%(c))
				itry+=1
				continue
			#Computing objective function at local parabola minimum
			alpha_parab = - phi_der/c
			step_scale = alpha_parab/alpha
			if(logger): logger.addToLog("	Testing point (c_opt=%s): m_current+c_opt*alpha*dm (parabola minimum)"%(step_scale))
			#If step length negative, re-evaluate points
			if(alpha_parab < 0.):
				if(logger): logger.addToLog("	Encountered a negative step-length value: %s; Shrinking step-length value."%(alpha_parab))
				#Shrink line search
				alpha *= self.shrink
				itry+=1
				continue
			#Clipping the step-length scale
			if step_scale < self.alpha_scale_min:
				if(logger): logger.addToLog("	!!! step-length scale of %s smaller than provided lower bound. Clipping its value to bound value of %s !!!"%(step_scale,self.alpha_scale_min))
				step_scale = self.alpha_scale_min
			elif step_scale > self.alpha_scale_max:
				if(logger): logger.addToLog("	!!! step-length scale of %s greater than provided upper bound. Clipping its value to bound value of %s !!!"%(step_scale,self.alpha_scale_max))
				step_scale = self.alpha_scale_max

			#Testing parabolic scale
			#Compute new objective function at the minimum of the parabolic approximation
			model_step.copy(modl)
			model_step.scaleAdd(dmodl,sc2=alpha*step_scale)
			#Checking if model parameters hit the bounds
			prblm.set_model(model_step)
			#Projecting model onto the bounds (if any)
			if("bounds" in dir(prblm)): prblm.bounds.apply(model_step)
			if(prblm_mdl.isDifferent(model_step)):
				#Model hit bounds
				msg="	Model hit provided bounds. Projecting it onto them."
				if(logger): logger.addToLog(msg)
			obj2=prblm.get_obj(model_step)
			if(logger): logger.addToLog("		Objective function value of %s"%(obj2))

			#Writing info to log file
			if(logger): logger.addToLog("	Initial objective function value: %s, Objective function at c1*alpha*dm: %s, Objective function at parabola minimum: %s"%(obj0,obj1,obj2))
			itry+=1

			#Check which one is the best step length
			if (obj1<obj0 and obj1<obj2):
				success = True
				alpha *= self.c1
				if(logger): logger.addToLog("	c1 best step-length value of: %s"%(alpha))
				break
			elif (obj2<obj0 and obj2<=obj1):
				success = True
				alpha *= step_scale
				if(logger): logger.addToLog("	parabola minimum best step-length value of: %s"%(alpha))
				break
			else:
				#Shrink line search
				alpha *= self.shrink
				if(logger): logger.addToLog("	Shrinking search direction")

		if(success):
			#Line search has finished, update model
			self.alpha=deepcopy(alpha)
			model_step.copy(modl) # model_step = m_current
			model_step.scaleAdd(dmodl,sc2=self.alpha)
			#Projecting model onto the bounds (if any)
			if("bounds" in dir(prblm)): prblm.bounds.apply(model_step)
			if(prblm_mdl.isDifferent(model_step)):
				#Computing true scaled search direction dm = m_new_clipped - m_current
				dmodl.copy(model_step)
				dmodl.scaleAdd(modl,1.0,-1.0)
				#Scaled by the inverse of the step length
				dmodl.scale(1.0/self.alpha)
			#Setting model and residual vectors to c1 or c2 point if parabola minimum is not picked
			prblm.set_model(model_step)
			if (obj1<obj0 and obj1<obj2):
				prblm.set_residual(res1)
			modl.copy(model_step)
		#Delete temporary vectors
		del model_step, res1
		return alpha,success
