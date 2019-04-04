#Module containing generic stopper definition

class Stopper:
	"""Stopper parent object"""

	#Default class methods/functions
	def __init__(self):
		"""Default class constructor for Stopper"""
		return

	def __del__(self):
		"""Default destructor"""
		return

	def reset(self):
		"""Function to reset stopper variables"""
		raise NotImplementedError("Implement reset stopper in the derived class.")
		return

	def run(self,prblm):
		"""Dummy stopper running method"""
		raise NotImplementedError("Implement run stopper in the derived class.")
		return
