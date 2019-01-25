#Module containing generic Solver and Restart definitions
import pyProblem
import pyVector as Vec
import numpy as np
import os
#Functions and modules necessary for writing on disk
import pickle
import atexit
from sep_util import write_file
from sys_util import mkdir
from shutil import rmtree
import datetime
import sep_util

#Testing if genericIO and SepVector module is present
import imp
try:
	imp.find_module('genericIO')
	import genericIO
	SepVector=genericIO.SepVector
	genIO_found = True
except ImportError:
	genIO_found = False

class Solver:
	"""Solver parent object"""

	#Default class methods/functions
	def __init__(self):
		"""Default class constructor for Solver"""
		return

	def __del__(self):
		"""Default destructor"""
		return

	def setDefaults(self,save_obj=False,save_res=False,save_grad=False,save_model=False,prefix=None,iter_buffer_size=None,iter_sampling=1,restart_folder=None,flush_memory=False):
		"""
		   Function to set parameters for result saving.
		   save_obj    = [False]; Flag to save objective function values into the list self.obj
		   save_res    = [False]; Flag to save residual vectors into the list self.res
		   save_grad   = [False]; Flag to save gradient vectors into the list self.grad
		   save_model  = [False]; Flag to save model vectors into the list self.model. It will also say the last inverted model vector into self.inv_model
		   prefix = [None]; Prefix of the files in which requested results will be saved; If prefix is None, then nothing is going to be saved on disk
		   iter_buffer_size = [None]; Number of steps to save before flushing results to disk (by default the solver waits until all iterations are done)
		   iter_sampling = [1]; Sampling of the iteration axis
		   flush_memory = [False]; Whether to keep results into the object lists or clean those once inversion is completed or results have been written on disk
		   restart_folder = [None]; Restart folder name if disk restart is necessary. If a restart is needed within a Python session, then this argument is unneccessary
		"""

		#Parameter for saving results
		self.save_obj = save_obj							#Flag to save objective function value
		self.save_res = save_res							#Flag to save residual vector
		self.save_grad= save_grad							#Flag to save gradient vector
		self.save_model = save_model						#Flag to save model vector
		#Prefix of the saved files (if provided the results will be written on disk)
		self.prefix = prefix								#Prefix for saving inversion results on disk
		self.flush_memory = flush_memory					#Keep the results in the RAM or flush memory every time results are written on disk

		#Default parameters
		if(self.prefix != None):
			if(self.save_obj): self.obj_file=self.prefix+"_obj.H"						#File name in which the objective function is saved
			if(self.save_model): self.inv_mod_file=self.prefix+"_inv_mod.H"				#File name in which the current inverted model is saved
			if(self.save_model): self.model_file=self.prefix+"_model.H"					#File name in which the model vector is saved
			if(self.save_res): self.res_file=self.prefix+"_residual.H"					#File name in which the residual vector is saved
			if(self.save_grad): self.grad_file=self.prefix+"_gradient.H"				#File name in which the gradient vector is saved

		#Iteration axis-sampling parameters
		self.iter_buffer_size=iter_buffer_size		#Number of steps to save before flushing results to disk (by default the solver waits until all iterations are done)
		self.iter_sampling=iter_sampling			#Sampling of the iteration axis

		#Lists of the results
		self.obj=list()								#List for objective function value
		self.model=list()							#List for model vectors
		self.res=list()								#List for residual vectors
		self.grad=list()							#List for gradient vectors
		self.inv_model=None							#Temporary saved inverted model
		self.iter_written=0							#Counter to keep track

		#Set Restart object
		self.restart=Restart()
		#Setting restart folder if user needs to do so
		if(restart_folder != None):
			self.restart.restart_folder = restart_folder
		return


	def save_results(self,iter,prblm,force_save=False,force_write=False):
		"""
		   Method to save results
		   force_saving = [False]; Flag to ignore iteration sampling
		   force_write  = [False]; Force writing on disk if necessary (used to handle last iteration)
		"""
		if(not isinstance(prblm,pyProblem.Problem)): raise TypeError("Input variable is not a Problem object")
		#Obtaining objective function value
		prblm_mod = prblm.get_model()
		objf_value=prblm.get_obj(prblm_mod)
		#Verify that SepVectors are used
		self.use_SepVector = False
		if(genIO_found):
			self.use_SepVector = isinstance(prblm_mod,SepVector.vector)
		#Save if it is forced to or if the solver hits a sampled iteration number
		#The objective function is saved every iteration if requested
		if(self.save_obj):
			self.obj.append(objf_value)
		if(iter%self.iter_sampling == 0 or force_save):
			if(self.save_model):
				self.model.append(prblm_mod.clone())
				#Storing model vector into a temporary vector
				del self.inv_model #Deallocating previous saved model
				self.inv_model=prblm_mod.clone()
			if(self.save_res):
				res_vec = prblm.get_res(prblm.get_model())
				self.res.append(res_vec.clone())
			if(self.save_grad):
				grad = prblm.get_grad(prblm.get_model())
				self.grad.append(grad.clone())
		#Write on disk if necessary or requested
		self._write_steps(force_write)
		return

	def _write_steps(self,force_write=False):
		"""Method to write inversion results on disk if forced to or if buffer is filled"""
		save = False
		#Save results if buffer size is hit
		if(self.iter_buffer_size!=None):
			current_buffer_size = max(len(range(self.iter_written,len(self.model))),len(range(self.iter_written,len(self.res))),len(range(self.iter_written,len(self.grad))))
			if(current_buffer_size >= self.iter_buffer_size): save = True
		#Save if requested
		if(force_write): save = True

		#Saving on disk?
		if(save):
			#Writing objective function value on disk if requested
			if(self.save_obj and self.prefix != None):
				write_file(self.obj_file,np.array(self.obj))
			#Writing current inverted model and model vectors on disk if requested
			if(self.save_model and self.prefix != None):
				if(genIO_found and self.use_SepVector): #Writing using genericIO and SepVector
					genericIO.defaultIO.writeVector(self.inv_mod_file,self.inv_model) #Overwriting previous written model
					for ivec in range(self.iter_written,len(self.model)):
						#Appending to previous written vectors if any
						genericIO.defaultIO.appendVector(self.model_file,self.model[ivec],flush=self.iter_buffer_size)
					genericIO.defaultIO.closeAppendFile(self.model_file)
				else:
					self.inv_model.writeVec(self.inv_mod_file,mode='w') #Overwriting previous written model
					for ivec in range(self.iter_written,len(self.model)):
						self.model[ivec].writeVec(self.model_file,mode='a') #Appending to previous written vectors if any
			#Writing gradient vectors on disk if requested
			if(self.save_grad and self.prefix != None):
				if(genIO_found and self.use_SepVector): #Writing using genericIO and SepVector
					for ivec in range(self.iter_written,len(self.grad)):
						#Appending to previous written vectors if any
						genericIO.defaultIO.appendVector(self.grad_file,self.grad[ivec],flush=self.iter_buffer_size)
					genericIO.defaultIO.closeAppendFile(self.grad_file)
				else:
					for ivec in range(self.iter_written,len(self.grad)):
						self.grad[ivec].writeVec(self.grad_file,mode='a') #Appending to previous written vectors if any
			#Writing residual vectors on disk if requested
			if(self.save_res and self.prefix != None):
				if(genIO_found and self.use_SepVector): #Writing using genericIO and SepVector
					for ivec in range(self.iter_written,len(self.res)):
						#Appending to previous written vectors if any
						genericIO.defaultIO.appendVector(self.res_file,self.res[ivec],flush=self.iter_buffer_size)
					genericIO.defaultIO.closeAppendFile(self.res_file)
				else:
					for ivec in range(self.iter_written,len(self.res)):
						self.res[ivec].writeVec(self.res_file,mode='a') #Appending to previous written vectors if any

			#Setting the counter of the vectors already written
			self.iter_written = max(len(self.model),len(self.res),len(self.grad))
			if(self.flush_memory):
				#Cleaning result lists if not requested to be kept in memory
				del self.obj;   self.obj   = list()
				del self.model; self.model = list()
				del self.res;   self.res   = list()
				del self.grad;  self.grad  = list()
				self.iter_written = 0
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
