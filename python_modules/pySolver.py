#Module containing generic Solver and Restart definitions
import pyProblem
import pyVector as Vec
import numpy as np
import os
#Functions and modules necessary for writing on disk
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle
import atexit
from sys_util import mkdir
from shutil import rmtree
import datetime 
import sep_util

class Solver:
	"""Solver parent object"""
	
	#Default class methods/functions
	def __init__(self):
		"""Default class constructor for Solver"""
		return
	
	def __del__(self):
		"""Default destructor"""
		return
		
	def setDefaults(self,inv_mod_file=None,obj_file=None,model_file=None,res_file=None,grad_file=None,iter_buffer=None,iter_sampling=1):
		"""Method to set common parameters and variables to many inversion method"""
		#Default parameters 
		self.inv_mod_file=inv_mod_file				#File name in which the current inverted model is saved
		self.obj_file=obj_file						#File name in which the objective function is saved
		self.model_file=model_file					#File name in which the model vector is saved
		self.res_file=res_file						#File name in which the residual vector is saved
		self.grad_file=grad_file					#File name in which the gradient vector is saved
		self.iter_buffer=iter_buffer				#Number of steps to save before flushing results to disc (by default the solver waits until all iterations are done)
		self.iter_sampling=iter_sampling			#Sampling of the iteration axis
		
		#Containers for results
		self.ObjFuncSet=Vec.vectorSet()				#Set for objective function value (treated as a vector)
		self.ModelSet=Vec.vectorSet()				#Set for model vectors
		self.ResSet=Vec.vectorSet()					#Set for residual vectors
		self.GradSet=Vec.vectorSet()				#Set for gradient vectors
		#Set Restart object
		self.restart=Restart()
		return
		
		
	def save_results(self,iter,prblm,force_saving=False,force_flush=False):
		"""Method to save results"""
		if(not isinstance(prblm,pyProblem.Problem)): raise TypeError("Input variable is not a Problem object")
		#Using vector class for saving objective function value
		objf_vec=Vec.vectorIC(np.array(prblm.get_obj(prblm.get_model())))
		if(iter == 0): #Set vector spaces if we are at the first iteration 
			self.ObjFuncSet.setSpace(objf_vec)
			self.ModelSet.setSpace(prblm.get_model())
			self.ResSet.setSpace(prblm.get_res(prblm.get_model()))
			self.GradSet.setSpace(prblm.get_grad(prblm.get_model()))
		#Save if it is forced to or if the solver hits a sampled iteration number
		if(iter%self.iter_sampling == 0 or force_saving):
			if(self.inv_mod_file!=None):
				#Storing model vector into a temporary vector
				self.inv_model_tmp=prblm.model
			if(self.obj_file!=None):
				self.ObjFuncSet.append(objf_vec)
			if(self.model_file!=None):
				self.ModelSet.append(prblm.get_model())
			if(self.res_file!=None):
				self.ResSet.append(prblm.get_res(prblm.get_model()))
			if(self.grad_file!=None):
				self.GradSet.append(prblm.get_grad(prblm.get_model()))
		#Write on disc if necessary or requested
		self.flush_step_buffer(force_flush)
		return
		
	def flush_step_buffer(self,force_flush=False):
		"""Method to write inversion results on disc if forced to or if buffer is filled"""
		save = False
		#Save results if buffer size is hit
		if(self.iter_buffer!=None): 
			if(len(self.ModelSet.vecSet) >= self.iter_buffer): save = True
		#Save if requested
		if(force_flush): save = True
		#Saving on disc?
		if(save):	
			if(self.inv_mod_file!=None): self.inv_model_tmp.writeVec(self.inv_mod_file)
			if(self.obj_file!=None): self.ObjFuncSet.writeSet(self.obj_file)
			if(self.model_file!=None): self.ModelSet.writeSet(self.model_file)
			if(self.res_file!=None): self.ResSet.writeSet(self.res_file)
			if(self.grad_file!=None): self.GradSet.writeSet(self.grad_file)
		return

	def run(self,prblm):
		"""Dummy Solver running method"""
		raise NotImplementedError("Implement run Solver in the derived class.")
		return
		
class Restart:
	"""Class for restarting a solver run"""
	
	def __init__(self):
		"""Restart constructor"""
		self.par_dict=dict()
		self.vec_dict=dict()
		#Restart folder in case it is necessary to write restart  
		now=datetime.datetime.now()
		restart_folder=sep_util.datapath+"restart_"+now.isoformat()+"/"
		restart_folder=restart_folder.replace(":","-")
		self.restart_folder=restart_folder
		#Calling write_restart when python session dies
		atexit.register(self.write_restart)
		return 
		
	def save_vector(self,vec_name,vector_in):
		"""Method to save vector for restarting"""
		#Deleting the vector if present in the dictionary
		element = self.vec_dict.pop(vec_name, None)
		if(element): del element
		self.vec_dict.update({vec_name: vector_in.clone()})
		return
		
	def retrieve_vector(self,vec_name):
		"""Method to retrieve a vector from restart object"""
		return self.vec_dict[vec_name]
	
	def save_parameter(self,par_name,parameter_in):
		"""Method to save vector for restarting"""
		self.par_dict.update({par_name: parameter_in})
		return
	
	def retrieve_parameter(self,par_name):
		"""Method to retrieve a parameter from restart object"""
		return self.par_dict[par_name]

	def write_restart(self):
		"""Restart destructor: it will write vectors on disk if the solver breaks"""
		if(bool(self.par_dict) or bool(self.vec_dict)):	
			#Creating restarting directory
			mkdir(self.restart_folder)
			with open(self.restart_folder+'restart_obj.pkl', 'wb') as output:
				pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
			#Checking if a vectorOC was in the restart and preventing the removal of the vector file
			for vec_name,vec in self.vec_dict.items():
				if(isinstance(vec,Vec.vectorOC)): vec.remove_file = False
		return
	
	def read_restart(self):
		"""Method to read restart object from saved folder"""
		if(os.path.isdir(self.restart_folder)):
			with open(self.restart_folder+'restart_obj.pkl', 'rb') as input:
				restart = pickle.load(input)
			self.par_dict=restart.par_dict
			self.vec_dict=restart.vec_dict
			#Checking if a vectorOC was in the restart and setting the removal of the vector file
			for vec_name,vec in self.vec_dict.items():
				if(isinstance(vec,Vec.vectorOC)): vec.remove_file = True
			#Removing previous restart and deleting read object
			restart.clear_restart()
			del restart
		return
		
	def clear_restart(self):
		"""Method to clear the restart"""
		self.par_dict=dict()
		self.vec_dict=dict()
		#Removing restart folder if existing
		if(os.path.isdir(self.restart_folder)):
			#Removing folder
			rmtree(self.restart_folder)
		return
			
 