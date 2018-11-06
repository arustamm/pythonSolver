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