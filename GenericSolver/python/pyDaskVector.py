#Module containing the definition of Dask-based vector class
import pyVector as Vec
import dask.distributed as daskD
import numpy as np
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
	if(type(vec2) is not VectorDask): raise TypeError("ERROR! Input variable is not a VectorDask")
	Nvec1 = len(vec1.vecDask)
	Nvec2 = len(vec2.vecDask)
	if(Nvec1 != Nvec2):
		raise ValueError("ERROR! Number of chunks is different! (self chunks %s; vec2 chunks %s)"%(Nvec1,Nvec2))
	return

class VectorDask(Vec.vector):
	"""
	   Definition of a vector object whose computations are performed through a Dask Client
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
			 or
			 - dask_vectors    = [no default] - list; List containing pointers to futures to vector object (useful for clone function)
		"""
		#Client to submit tasks
		if(not isinstance(client,daskD.client.Client)):
			raise TypeError("ERROR! Passed client is not a Dask Client object!")
		self.client = client
		#List containing futures to vectors
		self.vecDask = []
		#Getting worker IDs
		wrkIds = list(self.client.get_worker_logs().keys())
		N_wrk = len(wrkIds)
		if("vector_template" in kwargs and "chunks" in kwargs):
			vec_tmplt = kwargs.get("vector_template")
			chunks = kwargs.get("chunks")
			#Spreading chunks across available workers
			chunks = [np.sum(ix) for ix in np.array_split(chunks,N_wrk)]
			#Copying vector template to all workers
			vecD = self.client.scatter(vec_tmplt,broadcast=True)
			#Spreading vectors
			for iwrk,wrkId in enumerate(wrkIds):
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
			for iwrk,wrkId in enumerate(wrkIds):
				for vec in vec_chunks[iwrk]:
					self.vecDask.append(self.client.scatter(vec,workers=[wrkId]))
		elif("dask_vectors" in kwargs):
			dask_vectors = kwargs.get("dask_vectors")
			for dask_vec in dask_vectors:
				if(not issubclass(dask_vec.type,Vec.vector)):
					raise TypeError("ERROR! One instance in dask_vectors is not a vector-derived object!")
			self.client = client
			self.vecDask = dask_vectors
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
		"""
		   Function to return Ndarray of the vector
		   The function will return an Numpy array if dimensions among all the arrays are consistent with each other (i.e., slowest-axis concatenation). Otherwise, a list of all the arrays is going to be returned.
		"""
		futures = []
		for ivec in range(len(self.vecDask)):
			futures.append(self.client.submit(call_getNdArray,self.vecDask[ivec]))
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
		norms = self.client.map(call_norm,self.vecDask,N=N)
		norm = 0.0
		for future, result in daskD.as_completed(norms, with_results=True):
			norm += np.power(result,N)
		return np.power(norm,1./N)

	def zero(self):
		"""Function to zero out a vector"""
		daskD.wait(self.client.map(call_zero,self.vecDask))
		return

	def max(self):
		"""Function to obtain maximum value within a vector"""
		maxs = self.client.map(call_max,self.vecDask)
		max_val = - np.inf
		for future, result in daskD.as_completed(maxs, with_results=True):
			if(result > max_val): max_val = result
		return max_val

	def min(self):
		"""Function to obtain minimum value within a vector"""
		mins = self.client.map(call_min,self.vecDask)
		min_val = np.inf
		for future, result in daskD.as_completed(mins, with_results=True):
			if(result < min_val): min_val = result
		return min_val

	def set(self,val):
		"""Function to set all values in the vector"""
		daskD.wait(self.client.map(call_set,self.vecDask,val=val))
		return

	def scale(self,sc):
		"""Function to scale a vector"""
		daskD.wait(self.client.map(call_scale,self.vecDask,sc=sc))
		return

	def rand(self):
		"""Function to randomize a vector"""
		daskD.wait(self.client.map(call_rand,self.vecDask))
		return

	def clone(self):
		"""Function to clone (deep copy) a vector from a vector or a Space"""
		vectors = self.client.map(call_clone,self.vecDask)
		daskD.wait(vectors)
		return VectorDask(self.client,dask_vectors=vectors)

	def cloneSpace(self):
		"""Function to clone vector space"""
		vectors = self.client.map(call_cloneSpace,self.vecDask)
		daskD.wait(vectors)
		return VectorDask(self.client,dask_vectors=vectors)

	def checkSame(self,vec2):
		"""Function to check to make sure the vectors exist in the same space"""
		checkVector(self,vec2)
		futures = []
		for ivec in range(len(self.vecDask)):
			futures.append(self.client.submit(call_checkSame,self.vecDask[ivec],vec2.vecDask[ivec]))
		results = self.client.gather(futures)
		return all(results)

	def writeVec(self,filename,mode='w'):
		"""Function to write vector to file"""
		raise NotImplementedError("writeVec must be overwritten")
		return

	#Methods combinaning different vectors

	def copy(self,vec2):
		"""Function to copy vector"""
		checkVector(self,vec2)
		futures = []
		for ivec in range(len(self.vecDask)):
			futures.append(self.client.submit(call_copy,self.vecDask[ivec],vec2.vecDask[ivec]))
		daskD.wait(futures)
		return

	def scaleAdd(self,vec2,sc1=1.0,sc2=1.0):
		"""Function to scale two vectors and add them to the first one"""
		checkVector(self,vec2)
		futures = []
		for ivec in range(len(self.vecDask)):
			futures.append(self.client.submit(call_scaleAdd,self.vecDask[ivec],vec2.vecDask[ivec],sc1,sc2))
		daskD.wait(futures)
		return

	def dot(self,vec2):
		"""Function to compute dot product between two vectors"""
		checkVector(self,vec2)
		dots = []
		for ivec in range(len(self.vecDask)):
			dots.append(self.client.submit(call_dot,self.vecDask[ivec],vec2.vecDask[ivec]))
		#Adding all the results together
		dot = 0.0
		for future, result in daskD.as_completed(dots, with_results=True):
			dot += result
		return dot

	def multiply(self,vec2):
		"""Function to multiply element-wise two vectors"""
		checkVector(self,vec2)
		futures = []
		for ivec in range(len(self.vecDask)):
			futures.append(self.client.submit(call_multiply,self.vecDask[ivec],vec2.vecDask[ivec]))
		daskD.wait(futures)
		return

	def isDifferent(self,vec2):
		"""Function to check if two vectors are identical"""
		checkVector(self,vec2)
		futures = []
		for ivec in range(len(self.vecDask)):
			futures.append(self.client.submit(call_isDifferent,self.vecDask[ivec],vec2.vecDask[ivec]))
		results = self.client.gather(futures)
		return any(results)

	def clipVector(self,low,high):
		"""
		   Function to bound vector values based on input vectors min and max
		"""
		checkVector(self,low) #Checking low-bound vector
		checkVector(self,high) #Checking high-bound vector
		futures = []
		for ivec in range(len(self.vecDask)):
			futures.append(self.client.submit(call_clipVector,self.vecDask[ivec],low.vecDask[ivec],high.vecDask[ivec]))
		daskD.wait(futures)
		return
