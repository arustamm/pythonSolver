#Stepper object for back tracking method
import python_solver_outcore as solv
import sep_python as sep
from math import isnan


class stepper_parab(solv.stepper):

	def __init__(self,c1=1.0, c2=2.0, ntry=10, alpha=0., alpha_scale_min=1.0e-10, alpha_scale_max=50.00, shrink=0.25,maxval=None,minval=None):
		"""Constructor for sampling stepper"""
		solv.stepper.__init__(self,maxval,minval)
		self.c1=c1
		self.c2=c2
		self.ntry=ntry
		self.alpha=alpha
		self.alpha_scale_min=alpha_scale_min
		self.alpha_scale_max=alpha_scale_max
		self.shrink=shrink
		self.files_to_clean=[]
		#temporary model file
		self.model=sep.tmp_file("tmp1_stepper.H"); self.files_to_clean.append(self.model)
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
					print "	Temporary file not removed for debugging in PARABOLIC STEPPER: %s"%(ifile)
		return


	def run(self,prblm,modl,dmodl,log_file=None):
		#Writing info to log file
		solv.write_log_file(log_file,info="PARABOLIC STEPPER")
		solv.write_log_file(log_file,info="c1=%s c2=%s ntry=%s steplength-scaling-min=%s steplength-scaling-max=%s shrinking-factor=%s"%(self.c1,self.c2,self.ntry,self.alpha_scale_min,self.alpha_scale_max,self.shrink))
		success=False
		#Obtain objective function for provided model
		obj=prblm.get_obj(modl)
		obj0=sep.Get_value(obj)
		alpha=self.alpha
		itry=1
		while (itry <= 2*self.ntry):
			#Writing info to log file
			solv.write_log_file(log_file,info="	trial number: %s"%(itry))
			solv.write_log_file(log_file,info="	initial-steplength=%s"%(alpha))
			#Find the first guess as if the problem was linear
			if((itry==self.ntry) or (alpha == 0.)):
				dres=prblm.get_dres(modl,dmodl)
				res=prblm.get_res(modl)
				dres_res=sep.Dot_incore(res,dres)
				dres_dres=sep.Dot_incore(dres,dres)
				if(dres_dres==0.):
					solv.write_log_file(log_file,"	!!!Gradient in the null space of linear forward operator!!!")
					alpha = 1.0
				else:
					alpha = -dres_res/dres_dres
				info = "	Guessing linear step length of: %s"%(alpha)
				solv.write_log_file(log_file,info)
			#Test values of objective function for two scaled versions of step length
			#Testing c1 scale
			info = "	Testing point (c1=%s): m_current+c1*alpha*dm"%(self.c1)
			solv.write_log_file(log_file,info)
			sep.Cp(modl,self.model)
			sep.Sum(self.model,dmodl,1.0,self.c1*alpha)
			self.clipping(self.model,log_file)
			obj=prblm.get_obj(self.model)
			obj1=sep.Get_value(obj)
			#Copying residuals for point c1
			self.copy_tmp_result(prblm.get_res(self.model))
			info = "		Objective function value of %s"%(obj1)
			solv.write_log_file(log_file,info)
			#Checking if a Nan is encountered in any of the two tested points
			#Checking for NaN
			if(isnan(obj1)):
				info = "		!!!Problem with step length and objective function!!!"
				solv.write_log_file(log_file,info)
				if(itry>=self.ntry):
					info = "		!!!Check problem definition or change solver!!!"
					solv.write_log_file(log_file,info)
					break
				else:
					info = "		!!!Guessing linear step length to try to solve problem!!!"
					solv.write_log_file(log_file,info)
					itry=self.ntry #To not repeat computation of linear guess
					continue
			#Testing c2 scale
			info = "	Testing point (c2=%s): m_current+c2*alpha*dm"%(self.c2)
			solv.write_log_file(log_file,info)
			sep.Cp(modl,self.model)
			sep.Sum(self.model,dmodl,1.0,self.c2*alpha)
			self.clipping(self.model,log_file)
			obj=prblm.get_obj(self.model)
			obj2=sep.Get_value(obj)
			#Copying residuals for point c2
			self.copy_tmp_result(prblm.get_res(self.model))
			info = "		Objective function value of %s"%(obj2)
			solv.write_log_file(log_file,info)
			#Checking for NaN
			if(isnan(obj2)):
				info = "		!!!Problem with step length and objective function!!!"
				solv.write_log_file(log_file,info)
				if(itry>=self.ntry):
					info = "		!!!Check problem definition or change solver!!!"
					solv.write_log_file(log_file,info)
					break
				else:
					info = "		!!!Guessing linear step length to try to solve problem!!!"
					solv.write_log_file(log_file,info)
					itry=self.ntry #To not repeat computation of linear guess
					continue
			#If points lays on a horizontal line pick minimum alpha set by user
			if(obj0 == obj1 == obj2 or (self.c2*(obj1-obj0) + self.c1*(obj0-obj2)) == 0.):
				step_scale = self.alpha_scale_min
				info = "	Two testing points on a line: cannot fit a parabola, using minimum step-length of %s"%(step_scale*alpha)
				solv.write_log_file(log_file,info)
			else:
			#otherwise, find the optimal parabolic step length
				step_scale = 0.5*(self.c2*self.c2*(obj1-obj0) + self.c1*self.c1*(obj0-obj2))/(self.c2*(obj1-obj0) + self.c1*(obj0-obj2))
				info = "	Testing point (c_opt=%s): m_current+c_opt*alpha*dm (parabola minimum)"%(step_scale)
				solv.write_log_file(log_file,info)
			#Clipping the step-length scale
			if step_scale < self.alpha_scale_min:
				solv.write_log_file(log_file,"	!!! step-length scale of %s smaller than provided lower bound. Clipping its value to bound value of %s !!!"%(step_scale,self.alpha_scale_min))
				step_scale = max(step_scale,self.alpha_scale_min)
			elif step_scale > self.alpha_scale_max:
				solv.write_log_file(log_file,"	!!! step-length scale of %s greater than provided upper bound. Clipping its value to bound value of %s !!!"%(step_scale,self.alpha_scale_max))
				step_scale = min(step_scale,self.alpha_scale_max)

			#Testing parabolic scale
			#Compute new objective function at the minimum of the parabolic approximation
			sep.Cp(modl,self.model)
			sep.Sum(self.model,dmodl,1.0,step_scale*alpha)
			self.clipping(self.model,log_file)
			obj=prblm.get_obj(self.model)
			obj3=sep.Get_value(obj)
			info = "		Objective function value of %s"%(obj3)
			solv.write_log_file(log_file,info)

			#Writing info to log file
			info = "	Initial objective function value: %s, Objective function at c1*alpha*dm: %s, Objective function at c2*alpha*dm: %s, Objective function at parabola minimum: %s"%(obj0,obj1,obj2,obj3)
			solv.write_log_file(log_file,info)
			itry+=1


			#Check which one is the best step length
			if (obj1<obj0 and obj1<obj2 and obj1<obj3):
				success = True
				alpha *= self.c1
				solv.write_log_file(log_file,"	c1 best step-length value of: %s"%(alpha))
				ind_point=0
				break
			elif (obj2<obj0 and obj2<obj1 and obj2<obj3):
				success = True
				alpha *= self.c2
				solv.write_log_file(log_file,"	c2 best step-length value of: %s"%(alpha))
				ind_point=1
				break
			elif (obj3<obj0 and obj3<=obj1 and obj3<=obj2):
				success = True
				alpha *= step_scale
				solv.write_log_file(log_file,"	parabola minimum best step-length value of: %s"%(alpha))
				ind_point=None
				break
			else:
				#Shrink line search
				alpha *= self.shrink
				#Resetting saved residuals
				self.clear_tmp_result()
				solv.write_log_file(log_file,"	Shrinking search direction")

		if(success):
			#Line search has finished, update model
			self.alpha=alpha
			sep.Sum(modl,dmodl,1.0,self.alpha)
			self.clipping(modl)
			#Copying residuals from previous tested point if necessary
			if(ind_point!=None): self.set_from_tmp_result(prblm,modl,ind_point)
			#Resetting saved residuals
			self.clear_tmp_result()

		return alpha, success
