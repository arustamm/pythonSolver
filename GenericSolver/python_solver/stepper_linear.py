#Stepper object for linear step length
import python_solver_outcore as solv
import sep_python as sep


class stepper_linear(solv.stepper):

	def __init__(self,maxval=None,minval=None):
		"""Constructor for linear stepper"""
		solv.stepper.__init__(self,maxval,minval)
		self.files_to_clean=[]
		#Temp files
		self.model=sep.tmp_file("model_stepper.H"); self.files_to_clean.append(self.model)
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
					print "	Temporary file not removed for debugging in LINEAR STEPPER: %s"%(ifile)
		return

	#Search direction not the gradient
	def run(self,prblm,modl,dmodl,log_file=None):
		solv.write_log_file(log_file,info="LINEAR STEPPER")
		success=False
		#Obtaining objective function value, residual, and gradient in the data space
		obj=prblm.get_obj(modl)
		obj0=sep.Get_value(obj)
		res=prblm.get_res(modl)
		dres=prblm.get_dres(modl,dmodl)

		#Computing linear step length
		dres_res=sep.Dot_incore(res,dres)
		dres_dres=sep.Dot_incore(dres,dres)
		if(dres_dres!=0.):
			alpha=-dres_res/dres_dres
			#Checking if the model updates decrease objective function value
			sep.Cp(modl,self.model)
			#modl = modl + alpha*dmodl
			sep.Sum(self.model,dmodl,1.0,alpha)
			self.clipping(self.model)
			obj=prblm.get_obj(self.model)
			obj1=sep.Get_value(obj)
			info = "Starting objective function value: %s, step length: %s, New objective function value: %s"%(obj0,alpha,obj1)
			solv.write_log_file(log_file,info)
			if(obj1<obj0):
				success=True
				sep.Sum(modl,dmodl,1.0,alpha)
				self.clipping(modl,log_file)
		else:
			alpha = 0.
			success = False

		return alpha, success
