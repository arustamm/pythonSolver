#Stepper object for sampling stepper
import python_solver_outcore as solv
import sep_python as sep
import operator


class stepper_sample(solv.stepper):

	def __init__(self,npoints=7, scalemin=0.5, scalemax=1.5, ntry=10, alpha=0., shrink=0.25,maxval=None,minval=None):
		"""Constructor for sampling stepper"""
		solv.stepper.__init__(self,maxval,minval)
		self.npoints=npoints
		self.scalemin=scalemin
		self.scalemax=scalemax
		self.ntry=ntry
		self.alpha=alpha
		self.shrink=shrink
		self.objf=[0]*npoints
		self.c=[(scalemax-scalemin)*cval/(self.npoints-1.)+scalemin for cval in range(0,self.npoints)]
		self.objvalues=[0]*npoints
		self.files_to_clean=[]
		#Tmp files
		self.model=sep.tmp_file("model_stepper.H");	self.files_to_clean.append(self.model)
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
					print "	Temporary file not removed for debugging in SAMPLING STEPPER: %s"%(ifile)
		return

	def run(self,prblm,modl,dmodl,log_file=None):
		#Writing info to log file
		solv.write_log_file(log_file,info="SAMPLING STEPPER")
		solv.write_log_file(log_file,info="npoints=%s scalemin=%s scalemax=%s ntry=%s shrinking-factor=%s"%(self.npoints,self.scalemin,self.scalemax,self.ntry,self.shrink))
		success=False
		#Obtain objective function for provided model
		obj=prblm.get_obj(modl)
		obj0=sep.Get_value(obj)
		alpha=self.alpha
		for itry in range(1,2*self.ntry+1):
			#Writing info to log file
			solv.write_log_file(log_file,info="	trial number: %s"%(itry))
			solv.write_log_file(log_file,info="	initial-steplength=%s"%(alpha))
			#Obtaining scale for search direction using linear step length
			if((alpha == 0.) or (itry==self.ntry)):
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
			for ii in range(0,self.npoints):
				info = "	Testing point %s out of %s:"%(ii+1,self.npoints)
				solv.write_log_file(log_file,info)
				sep.Cp(modl,self.model)
				#modl = modl + c(i)*alpha*dmodl
				sep.Sum(self.model,dmodl,1.0,self.c[ii]*alpha)
				self.clipping(self.model,log_file)
				obj=prblm.get_obj(self.model)
				self.objvalues[ii]=sep.Get_value(obj)
				#Copying residuals for current tested point
				self.copy_tmp_result(prblm.get_res(self.model))
				info = "		- objective function value %s"%(self.objvalues[ii])
				solv.write_log_file(log_file,info)
			info = "	Testing scaling points: \n	%s for step length of: %s\n	Objective function values: \n	%s"%(self.c,alpha,self.objvalues)
			solv.write_log_file(log_file,info)
			if(min(self.objvalues) < obj0):
				min_index=min(enumerate(self.objvalues), key=operator.itemgetter(1))[0]
				scale=self.c[min_index]
				alpha*=scale
				info = "	Objective function minimum value of: %s found for step length: %s for scaling of: %s"%(min(self.objvalues),alpha,scale)
				solv.write_log_file(log_file,info)
				success = True
				break
			else:
				alpha*=self.shrink
				#Resetting saved residuals
				self.clear_tmp_result()
				solv.write_log_file(log_file,"	Shrinking search direction")
		if(success):
			#Line search has finished, update model
			self.alpha=alpha
			sep.Sum(modl,dmodl,1.0,self.alpha)
			self.clipping(modl)
			#Copying residuals from previous tested point
			self.set_from_tmp_result(prblm,modl,min_index)
			#Resetting saved residuals
			self.clear_tmp_result()

		return alpha, success
