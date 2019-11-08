#Module containing the definition of Dask-based vector class
import pyVector as Vec
import dask.distributed as daskD
from dask_util import DaskClient
import numpy as np

#Specific functions to use genericIO vectors
import imp
#Verify if SepVector modules are presents
try:
	imp.find_module('SepVector')
	import SepVector
	def call_constr_hyper(hyper):
		"""Function to remotely construct an SepVector using the Hypercube"""
		return SepVector.getSepVector(axes=hyper.axes)
	def copy_from_NdArray(vecObj,NdArray):
		"""Function to set vector values from numpy array"""
		vecObj.getNdArray()[:] = NdArray
		return
except ImportError:
	SepVector = None


#Functions necessary to submit method calls using Dask client
def call_getNdArray(vecObj):
	"""Function to call getNdArray method"""
	res = vecObj.getNdArray()
	return res
def call_norm(vecObj,N=2):
	"""Function to call norm method"""
	res = vecObj.norm(N)
	return res
def call_zero(vecObj):
	"""Function to call zero method"""
	res = vecObj.zero()
	return res
def call_max(vecObj):
	"""Function to call max method"""
	res = vecObj.max()
	return res
def call_min(vecObj):
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
	"""Function to call clone method"""
	res = vecObj.clone()
	return res
def call_cloneSpace(vecObj):
	"""Function to call cloneSpace method"""
	res = vecObj.cloneSpace()
	return res
def call_checkSame(vecObj,vec2):
	"""Function to call cloneSpace method"""
	res = vecObj.checkSame(vec2)
	return res
def call_copy(vecObj,vec2):
	"""Function to call copy method"""
	res = vecObj.copy(vec2)
	return res
def call_scaleAdd(vecObj,vec2,sc1,sc2):
	"""Function to call scaleAdd method"""
	res = vecObj.scaleAdd(vec2,sc1,sc2)
	return res
def call_dot(vecObj,vec2):
	"""Function to call dot method"""
	res = vecObj.dot(vec2)
	return res
def call_multiply(vecObj,vec2):
	"""Function to call multiply method"""
	res = vecObj.multiply(vec2)
	return res
def call_isDifferent(vecObj,vec2):
	"""Function to call isDifferent method"""
	res = vecObj.isDifferent(vec2)
	return res
def call_clipVector(vecObj,low,high):
	"""Function to call multiply method"""
	res = vecObj.clipVector(vec2,low,high)
	return res

#Check consistency between vectors
def checkVector(vec1,vec2):
	"""Function to check type and chunks of Dask-vector objects"""
	if(type(vec2) is not DaskVector): raise TypeError("Input variable is not a DaskVector")
	Nvec1 = len(vec1.vecDask)
	Nvec2 = len(vec2.vecDask)
	if(Nvec1 != Nvec2):
		raise ValueError("Number of chunks is different! (self chunks %s; vec2 chunks %s)"%(Nvec1,Nvec2))
	return

class DaskVector(Vec.vector):
	"""
	   Definition of a vector object whose computations are performed through a Dask Client
	"""

	def __init__(self,dask_client,**kwargs):
		"""
		   Dask Vector constructor
		   dask_client = [no default] - DaskClient; client object to use when submitting tasks (see dask_util module)
		   kwargs:
			 - vector_template = [no default] - vector class; Vector to use to create chunks of vectors
			 - chunks          = [no default] - list; List defininig the size of the multiple instances of the vector template
			 or
			 - vectors         = [no default] - list; List containing vectors to be spread across Dask workers
			 - copy            = [True] - boolean; Whether to copy the content of the vectors or not
			 - chunks          = [None] - list; List defininig how the vector list should be spread; if not specified the vectors will be evenly distributed
			 or
			 - dask_vectors    = [no default] - list; List containing pointers to futures to vector object (useful for clone function)
		"""
		#Client to submit tasks
		if not isinstance(dask_client,DaskClient):
			raise TypeError("Passed client is not a Dask Client object!")
		self.dask_client = dask_client
		self.client = self.dask_client.getClient()
		#List containing futures to vectors
		self.vecDask = []
		#Getting worker IDs
		wrkIds = self.dask_client.getWorkerIds()
		N_wrk = self.dask_client.getNworkers()
		if "vector_template" in kwargs and "chunks" in kwargs:
			vec_tmplt = kwargs.get("vector_template")
			chunks = kwargs.get("chunks")
			#Spreading chunks across available workers
			chunks = [np.sum(ix) for ix in np.array_split(chunks,N_wrk)]
			#Checking if an SepVector was passed (by getting Hypercube)
			hyper=False
			if SepVector:
				if(isinstance(vec_tmplt,SepVector.vector)):
					hyper=True
			#Broadcast vector space
			if hyper:
				vec_space = vec_tmplt.getHyper()
			else:
				vec_space = vec_tmplt.cloneSpace()
			vec_spaceD = self.client.scatter(vec_space,broadcast=True)
			daskD.wait(vec_spaceD)
			#Spreading vectors
			for iwrk,wrkId in enumerate(wrkIds):
				for ivec in range(chunks[iwrk]):
					if hyper:
						#Instantiating Sep vectors on remote machines
						self.vecDask.append(self.client.submit(call_constr_hyper,vec_spaceD,workers=[wrkId],pure=False))
					else:
						#Scattering vector to different workers
						self.vecDask.append(self.client.submit(call_clone,vec_spaceD,workers=[wrkId],pure=False))
		elif "vectors" in kwargs:
			#Vector list to be spread across workers
			vec_list = kwargs.get("vectors")
			copy = kwargs.get("copy",True)
			chunks = kwargs.get("chunks",None)
			if chunks is None:
				#Spread vectors evenly
				vec_chunks = np.array_split(vec_list,N_wrk)
			else:
				#Spread according to chunk size
				if len(vec_list) != np.sum(chunks):
					raise ValueError("Total number of vectors in chunks not consistent with number of vectors!")
				#Spreading chunks across available workers
				chunks = [np.sum(ix) for ix in np.array_split(chunks,N_wrk)]
				vec_chunks = np.split(vec_list,np.cumsum(chunks))[:-1]
			#Spreading vectors
			for iwrk,wrkId in enumerate(wrkIds):
				for vec in vec_chunks[iwrk]:
					#Checking if an SepVector was passed
					IsSepVec = False
					if SepVector:
						if(isinstance(vec,SepVector.vector)): IsSepVec=True
					if IsSepVec:
						#Instantiating Sep vectors on remote machines
						self.vecDask.append(self.client.submit(call_constr_hyper,vec.getHyper(),workers=[wrkId],pure=False))
						#Copying values from NdArray (Cannot scatter SepVector)
						daskD.wait(self.vecDask[-1])
						if(copy): daskD.wait(self.client.submit(copy_from_NdArray,self.vecDask[-1],vec.getNdArray(),pure=False))
					else:
						if(copy):
							self.vecDask.append(self.client.scatter(vec,workers=[wrkId]))
						else:
							self.vecDask.append(self.client.submit(call_clone,vec.cloneSpace(),workers=[wrkId],pure=False))
		elif "dask_vectors" in kwargs:
			dask_vectors = kwargs.get("dask_vectors")
			for dask_vec in dask_vectors:
				if not issubclass(dask_vec.type,Vec.vector):
					raise TypeError("One instance in dask_vectors is not a vector-derived object!")
			self.dask_client = dask_client
			self.client = self.dask_client.getClient()
			self.vecDask = dask_vectors
		else:
			raise ValueError("Wrong arguments passed to constructor! Please, read object help!")
		#Waiting vectors to be instantiated
		daskD.wait(self.vecDask)
		return

	def __del__(self):
		"""
		   Cancel/Delete all futures within the class (fees memory on workers)
		"""
		#If a future is deleted is cancelled, then all the related ones are cancelled too. This is a problem
		#for the methods clone and cloneSpace. Need to find a solution to the problem
		# self.client.cancel(self.vecDask)
		return

	#Class vector operations
	def getNdArray(self):
		"""
		   Function to return Ndarray of the vector
		   The function will return an Numpy array if dimensions among all the arrays are consistent with each other (i.e., slowest-axis concatenation). Otherwise, a list of all the arrays is going to be returned.
		"""
		futures = self.client.map(call_getNdArray,self.vecDask,pure=False)
		arrays = self.client.gather(futures)
		#Checking if dimension are consistent with each other
		shapes = [arr.shape for arr in arrays]
		#Find maximum number of axis
		Naxis = np.max([len(shp) for shp in shapes])
		#Expanding slowest axis if necessary
		for idx,arr in enumerate(arrays):
			dim_diff = Naxis - len(arr.shape)
			if(dim_diff == 1):
				arrays[idx] = np.expand_dims(arr,axis=0)
		try:
			NdArr = np.concatenate(arrays,axis=0)
			return NdArr
		except ValueError:
			return arrays

	def norm(self,N=2):
		"""Function to compute vector N-norm"""
		norms = self.client.map(call_norm,self.vecDask,N=N,pure=False)
		norm = 0.0
		for future, result in daskD.as_completed(norms, with_results=True):
			norm += np.power(result,N)
		return np.power(norm,1./N)

	def zero(self):
		"""Function to zero out a vector"""
		daskD.wait(self.client.map(call_zero,self.vecDask,pure=False))
		return

	def max(self):
		"""Function to obtain maximum value within a vector"""
		maxs = self.client.map(call_max,self.vecDask,pure=False)
		max_val = - np.inf
		for future, result in daskD.as_completed(maxs, with_results=True):
			if(result > max_val): max_val = result
		return max_val

	def min(self):
		"""Function to obtain minimum value within a vector"""
		mins = self.client.map(call_min,self.vecDask,pure=False)
		min_val = np.inf
		for future, result in daskD.as_completed(mins, with_results=True):
			if(result < min_val): min_val = result
		return min_val

	def set(self,val):
		"""Function to set all values in the vector"""
		daskD.wait(self.client.map(call_set,self.vecDask,val=val,pure=False))
		return

	def scale(self,sc):
		"""Function to scale a vector"""
		daskD.wait(self.client.map(call_scale,self.vecDask,sc=sc,pure=False))
		return

	def rand(self):
		"""Function to randomize a vector"""
		daskD.wait(self.client.map(call_rand,self.vecDask,pure=False))
		return

	def clone(self):
		"""Function to clone (deep copy) a vector from a vector or a Space"""
		vectors = self.client.map(call_clone,self.vecDask,pure=False)
		daskD.wait(vectors)
		return DaskVector(self.dask_client,dask_vectors=vectors)

	def cloneSpace(self):
		"""Function to clone vector space"""
		vectors = self.client.map(call_cloneSpace,self.vecDask,pure=False)
		daskD.wait(vectors)
		return DaskVector(self.dask_client,dask_vectors=vectors)

	def checkSame(self,vec2):
		"""Function to check to make sure the vectors exist in the same space"""
		checkVector(self,vec2)
		futures = self.client.map(call_checkSame,self.vecDask,vec2.vecDask,pure=False)
		results = self.client.gather(futures)
		return all(results)

	def writeVec(self,filename,mode='w',multi_file=True):
		"""
		Function to write vector to file:

		:param filename: string - Filename to write the vector to
		:param mode: string - Writing mode 'w'=overwrite file or 'a'=append to file ['w']
		:param multi_file: boolean - If True multiple files will be written with suffix _chunk1,2,3,...;
		otherwise, a single will be written [True]
		"""
		#Check writing mode
		if(not mode in 'wa'):
			raise ValueError("Mode must be appending 'a' or writing 'w' ")
		return

	#Methods combinaning different vectors

	def copy(self,vec2):
		"""Function to copy vector"""
		checkVector(self,vec2)
		futures = self.client.map(call_copy,self.vecDask,vec2.vecDask,pure=False)
		daskD.wait(futures)
		return

	def scaleAdd(self,vec2,sc1=1.0,sc2=1.0):
		"""Function to scale two vectors and add them to the first one"""
		checkVector(self,vec2)
		sc1 = [sc1]*len(self.vecDask)
		sc2 = [sc2]*len(self.vecDask)
		futures = self.client.map(call_scaleAdd,self.vecDask,vec2.vecDask,sc1,sc2,pure=False)
		daskD.wait(futures)
		return

	def dot(self,vec2):
		"""Function to compute dot product between two vectors"""
		checkVector(self,vec2)
		dots = self.client.map(call_dot,self.vecDask,vec2.vecDask,pure=False)
		#Adding all the results together
		dot = 0.0
		for future, result in daskD.as_completed(dots, with_results=True):
			dot += result
		return dot

	def multiply(self,vec2):
		"""Function to multiply element-wise two vectors"""
		checkVector(self,vec2)
		futures = self.client.map(call_multiply,self.vecDask,vec2.vecDask,pure=False)
		daskD.wait(futures)
		return

	def isDifferent(self,vec2):
		"""Function to check if two vectors are identical"""
		checkVector(self,vec2)
		futures = self.client.map(call_isDifferent,self.vecDask,vec2.vecDask,pure=False)
		results = self.client.gather(futures)
		return any(results)

	def clipVector(self,low,high):
		"""
		   Function to bound vector values based on input vectors min and max
		"""
		checkVector(self,low) #Checking low-bound vector
		checkVector(self,high) #Checking high-bound vector
		futures = self.client.map(call_clipVector,self.vecDask,low.vecDask,high.vecDask,pure=False)
		daskD.wait(futures)
		return
