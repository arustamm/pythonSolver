#Module containing generic Solver and Restart definitions
import pyProblem
import pyVector as Vec
import numpy as np
import os
import re
#Functions and modules necessary for writing on disk
import pickle
import atexit
from sep_util import write_file
from sep_util import rm_file
from sys_util import mkdir
from shutil import rmtree
from copy import deepcopy
import datetime
import sep_util

class Solver:
	"""Solver parent object"""

	#Default class methods/functions
	def __init__(self):
		"""Default class constructor for Solver"""
		#Parameter for saving results
		self.save_obj = False
		self.save_res = False
		self.save_grad= False
		self.save_model = False
		self.flush_memory = False

		self.prefix = None

		#Iteration axis-sampling parameters
		self.iter_buffer_size=None
		self.iter_sampling=1

		#Lists of the results (list and vector Sets)
		self.obj=list()
		self.obj_terms=list()
		self.model=list()
		self.res=list()
		self.grad=list()
		self.modelSet=Vec.vectorSet()
		self.resSet=Vec.vectorSet()
		self.gradSet=Vec.vectorSet()
		self.inv_model=None
		self.iter_written=0

		#Set Restart object
		self.restart=Restart()
		return

	def __del__(self):
		"""Default destructor"""
		return

	def setPrefix(self,prefix):
		"""
		   Mutator to change prefix and file names for saving inversion results
		"""
		self.prefix = prefix
		return

	def setDefaults(self,save_obj=False,save_res=False,save_grad=False,save_model=False,prefix=None,iter_buffer_size=None,iter_sampling=1,flush_memory=False):
		"""
		   Function to set parameters for result saving.
		   save_obj    = [False] - boolean; Flag to save objective function values into the list self.obj
		   save_res    = [False] - boolean; Flag to save residual vectors into the list self.res
		   save_grad   = [False] - boolean; Flag to save gradient vectors into the list self.grad
		   save_model  = [False] - boolean; Flag to save model vectors into the list self.model. It will also say the last inverted model vector into self.inv_model
		   prefix = [None] - string; Prefix of the files in which requested results will be saved; If prefix is None, then nothing is going to be saved on disk
		   iter_buffer_size = [None] - int; Number of steps to save before flushing results to disk (by default the solver waits until all iterations are done)
		   iter_sampling = [1] - int; Sampling of the iteration axis
		   flush_memory = [False] - boolean; Whether to keep results into the object lists or clean those once inversion is completed or results have been written on disk
		"""

		#Parameter for saving results
		self.save_obj = save_obj							#Flag to save objective function value
		self.save_res = save_res							#Flag to save residual vector
		self.save_grad= save_grad							#Flag to save gradient vector
		self.save_model = save_model						#Flag to save model vector
		self.flush_memory = flush_memory					#Keep the results in the RAM or flush memory every time results are written on disk

		#Prefix of the saved files (if provided the results will be written on disk)
		self.prefix = prefix								#Prefix for saving inversion results on disk

		#Iteration axis-sampling parameters
		self.iter_buffer_size=iter_buffer_size		#Number of steps to save before flushing results to disk (by default the solver waits until all iterations are done)
		self.iter_sampling=iter_sampling			#Sampling of the iteration axis

		#Lists of the results (list and vector Sets)
		self.obj=list()								#List for objective function value
		self.obj_terms=list()						#List for objective function value for each terms
		self.model=list()							#List for model vectors (to save results in-core)
		self.res=list()								#List for residual vectors (to save results in-core)
		self.grad=list()							#List for gradient vectors (to save results in-core)
		self.modelSet=Vec.vectorSet()				#Set for model vectors
		self.resSet=Vec.vectorSet()				 	#Set for residual vectors
		self.gradSet=Vec.vectorSet()				#Set for gradient vectors
		self.inv_model=None							#Temporary saved inverted model
		self.iter_written=0							#Counter to keep track

		return

	def get_restart(self,log_file):
		"""
		   Function to retrieve restart folder from log file
		   It enables the user to use restart flag on self.run()
		   log_file = [None] - string;
		"""
		restart_folder = None
		#Obtaining restart folder path
		reg_prog = re.compile("Restart folder: ([^\s]+)")
		if(not os.path.isfile(log_file)):
			raise OSError("ERROR! No %s file found!"%(log_file))
		for line in reversed(open(log_file).readlines()):
			if (restart_folder == None):
				find = reg_prog.search(line)
				if find:
					restart_folder = find.group(1)
		#Setting restart folder if user needs to do so
		if(restart_folder != None):
			self.restart.restart_folder = restart_folder
		else:
			print("WARNING! No restart folder's path was found in %s"%(log_file))
		return


	def save_results(self,iter,prblm,model=None,force_save=False,force_write=False):
		"""
		   Method to save results
		   model		= [None];  Model vector to be saved
		   force_saving = [False]; Flag to ignore iteration sampling
		   force_write  = [False]; Force writing on disk if necessary (used to handle last iteration)
		"""
		if(not isinstance(prblm,pyProblem.Problem)):
			raise TypeError("Input variable is not a Problem object")
		#Getting a model from arguments if provided (necessary to remove preconditioning)
		if(model != None):
			mod_save = model
		else:
			mod_save = prblm.get_model()
		#Obtaining objective function value
		objf_value=prblm.get_obj(prblm.get_model())
		#Save if it is forced to or if the solver hits a sampled iteration number
		#The objective function is saved every iteration if requested
		if(self.save_obj):
			self.obj.append(objf_value)
			#Checking if the objective function has multiple terms
			if("obj_terms" in dir(prblm)):
				self.obj_terms.append(deepcopy(prblm.obj_terms))
		if(iter%self.iter_sampling == 0 or force_save):
			if(self.save_model):
				self.modelSet.append(mod_save)
				#Storing model vector into a temporary vector
				del self.inv_model #Deallocating previous saved model
				self.inv_model=mod_save.clone()
			if(self.save_res):
				res_vec = prblm.get_res(prblm.get_model())
				self.resSet.append(res_vec)
			if(self.save_grad):
				grad = prblm.get_grad(prblm.get_model())
				self.gradSet.append(grad)
		#Write on disk if necessary or requested
		self._write_steps(force_write)
		return

	def _write_steps(self,force_write=False):
		"""Method to write inversion results on disk if forced to or if buffer is filled"""
		save = False
		#Save results if buffer size is hit
		if(self.iter_buffer_size != None):
			if(max(len(self.modelSet.vecSet),len(self.resSet.vecSet),len(self.gradSet.vecSet)) >= self.iter_buffer_size): save = True
		#Save if requested
		if(force_write): save = True

		#Saving on disk?
		if(save):
			#Getting current saved results into an in-core list
			if(not self.flush_memory):
				self.model += self.modelSet.vecSet
				self.res   += self.resSet.vecSet
				self.grad  += self.gradSet.vecSet
			#Writing objective function value on disk if requested
			if(self.save_obj and self.prefix != None):
				obj_file = self.prefix+"_obj.H"						#File name in which the objective function is saved
				write_file(obj_file,np.array(self.obj))
				#Writing each term of the objective function
				if(self.obj_terms):
					for iterm in range(len(self.obj_terms[0])):
						obj_file = self.prefix+"_obj_comp%s.H"%(iterm+1)	#File name in which the objective function is saved
						write_file(obj_file,np.array([objs[iterm] for objs in self.obj_terms]))
			#Writing current inverted model and model vectors on disk if requested
			if(self.save_model and self.prefix != None):
				inv_mod_file = self.prefix+"_inv_mod.H"				#File name in which the current inverted model is saved
				model_file   = self.prefix+"_model.H"				#File name in which the model vector is saved
				self.modelSet.writeSet(model_file)
				#Using the modelSet to write inverted model vector
				self.modelSet.append(self.inv_model)				#Adding inverted model vector to set
				self.modelSet.writeSet(inv_mod_file,mode="w")		#Writing inverted model file
			#Writing gradient vectors on disk if requested
			if(self.save_grad and self.prefix != None):
				grad_file = self.prefix+"_gradient.H"					#File name in which the gradient vector is saved
				self.gradSet.writeSet(grad_file)
			#Writing residual vectors on disk if requested
			if(self.save_res and self.prefix != None):
				res_file = self.prefix+"_residual.H"				#File name in which the residual vector is saved
				self.resSet.writeSet(res_file)
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
