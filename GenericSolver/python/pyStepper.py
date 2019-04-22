#Module containing generic Stepper definition

class Stepper:
	"""Stepper parent object"""

	#Default class methods/functions
	def __init__(self):
		"""Default class constructor for Stepper"""
		return

	def __del__(self):
		"""Default destructor"""
		return

	def run(self,model,search_dir):
		"""Dummy stepper running method"""
		raise NotImplementedError("Implement run stepper in the derived class.")
		return

	def estimate_initial_guess(self,prblm,modl,dmodl,logger):
		"""
		   Function to estimate initial step length value
		"""
		try:
			#Projecting search direction in the data space
			dres=prblm.get_dres(modl,dmodl)
		except NotImplementedError:
			if(logger): logger.addToLog("	!!!dresf not implemented; stepper will use inverse of search direction norm as initial step length value!!!")
			#Initial step length value of 1.0 / |dm|_2
			alpha_guess = 1.0/dmodl.norm()
			return alpha_guess
		res=prblm.get_res(modl)
		dres_res=res.dot(dres)
		dres_dres=dres.dot(dres)
		if(dres_dres == 0.):
			if(logger): logger.addToLog("	!!!Gradient in the null space of linear forward operator; using inverse of search direction norm as step length value!!!")
			#Initial step length value of 1.0 / |dm|_2
			alpha_guess = 1.0/dmodl.norm()
		else:
			#alpha = -phi'(0)/phi''(0) 
			alpha_guess = -dres_res/dres_dres
		return alpha_guess
