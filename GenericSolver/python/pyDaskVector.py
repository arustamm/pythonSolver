#Module containing the definition of Dask-based vector class
import pyVector as Vec
import dask.distributed as daskD
import numpy as np
#Functions necessary to submit method calls using Dask client
def call_norm(vecObj,N=2):
	"""Function to call norm method"""
	res = vecObj.norm(N)
	return res
def call_zero(vecObj):
	"""Function to call zero method"""
	res = vecObj.zero()
	return res
def call_max(vecObj,val):
	"""Function to call max method"""
	res = vecObj.max()
	return res
def call_min(vecObj,val):
	"""Function to call min method"""
	res = vecObj.min()
	return res
def call_set(vecObj,val):
	"""Function to call set method"""
	res = vecObj.set(val)
	return res
def call_scale(vecObj,sc):
	"""Function to call scale method"""
	res = vecObj.scale(sc)
	return res
def call_rand(vecObj):
	"""Function to call rand method"""
	res = vecObj.rand()
	return res

def call_clone(vecObj):
	"""Function to call set method"""
	res = vecObj.clone()
	return res

class VectorDask(Vec.vector):
	"""
	   Definition of a vector object that whose computation are performed through a Dask Client
	"""

	def __init__(self,client,**kwargs):
		"""
		   Dask Vector constructor
		   client = [no default] - dask client; client object to use when submitting tasks
		   kwargs:
			 - vector_template = [no default] - vector class; Vector to use to create chunks of vectors
			 - chunks          = [no default] - list; List defininig the size of the multiple instances of the vector template
			 or
			 - vectors         = [no default] - list; List containing vectors to be spread across Dask workers
			 - chunks          = [None] - list; List defininig how the vector list should be spread; if not specified the vectors will be evenly distributed
		"""
		#Client to submit tasks
		if(not isinstance(client,daskD.client.Client)):
			raise TypeError("ERROR! Passed client is not a Dask Client object!")
		self.client = client
		#List containing futures to vectors
		self.vecDask = []
		#Getting worker IDs
		self.wrkIds = list(self.client.get_worker_logs().keys())
		N_wrk = len(self.wrkIds)
		if("vector_template" in kwargs and "chunks" in kwargs):
			vec_tmplt = kwargs.get("vector_template")
			chunks = kwargs.get("chunks")
			#Spreading chunks across available workers
			chunks = [np.sum(ix) for ix in np.array_split(chunks,N_wrk)]
			#Copying vector template to all workers
			vecD = self.client.scatter(vec_tmplt,broadcast=True)
			#Spreading vectors
			for iwrk,wrkId in enumerate(self.wrkIds):
				for ivec in range(chunks[iwrk]):
					#Cloning remote vector to create
					self.vecDask.append(self.client.submit(call_clone,vecD,workers=[wrkId]))
		elif("vectors" in kwargs):
			#Vector list to be spread across workers
			vec_list = kwargs.get("vectors")
			chunks = kwargs.get("chunks",None)
			if(chunks is None):
				#Spread vectors evenly
				vec_chunks = np.array_split(vec_list,N_wrk)
			else:
				#Spread according to chunk size
				if(len(vec_list) != np.sum (chunks)):
					raise ValueError("ERROR! Total number of vectors in chunks not consistent with number of vectors!")
				#Spreading chunks across available workers
				chunks = [np.sum(ix) for ix in np.array_split(chunks,N_wrk)]
				vec_chunks = np.split(vec_list,np.cumsum(chunks))[:-1]
			#Spreading vectors
			for iwrk,wrkId in enumerate(self.wrkIds):
				for vec in vec_chunks[iwrk]:
					self.vecDask.append(self.client.scatter(vec,workers=[wrkId]))
		else:
			raise ValueError("ERROR! Wrong arguments passed to constructor! Please, read object help!")
		return

	def __del__(self):
		"""
		   Cancel/Delete all futures within the class (fees memory on workers)
		"""
		self.client.cancel(self.vecDask)
		return

	#Class vector operations
	def getNdArray(self):
		"""Function to return Ndarray of the vector"""
		raise NotImplementedError("getNdArray must be overwritten")
		return

	def norm(self,N=2):
		"""Function to compute vector N-norm"""
		norms = self.client.map(call_norm,self.vecDask,N=N)
		norm = 0.0
		for future, result in daskD.as_completed(norms, with_results=True):
			norm += np.power(result,N)
		return np.power(norm,1./N)

	def zero(self):
		"""Function to zero out a vector"""
		daskD.wait(self.client.map(call_zero,self.vecDask))
		return

	def set(self,val):
		"""Function to set all values in the vector"""
		daskD.wait(self.client.map(call_set,self.vecDask,val=val))
		return

	def rand(self):
		"""Function to randomize a vector"""
		daskD.wait(self.client.map(call_rand,self.vecDask))
		return
