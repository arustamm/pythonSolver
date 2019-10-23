#Module containing the definition of Dask-based operator class
from pyDaskVector import DaskVector
import pyOperator as Op
import dask.distributed as daskD
from dask_util import DaskClient
import numpy as np

def call_constructor(constr,args):
	"""Function to call the constructor"""
	print(args)
	op = constr(*args)
	return op
def call_getDomain(opObj):
	"""Function to call getDomain method"""
	domain = opObj.getDomain()
	return domain
def call_getRange(opObj):
	"""Function to call getRange method"""
	range = opObj.getRange()
	return range
def call_forward(opObj,add,model,data):
	"""Function to call forward operator"""
	res = opObj.forward(add,model,data)
	return res
def call_adjoint(opObj,add,model,data):
	"""Function to call adjoint operator"""
	res = opObj.adjoint(add,model,data)
	return res


class DaskOperator(Op.Operator):
	"""
	   Class to apply multiple operators in parallel through Dask and DaskVectors
	"""

	def __init__(self,dask_client,op_constructor,op_args,chunks):
		"""
		   Dask Operator constructor
		   dask_client = [no default] - DaskClient; client object to use when submitting tasks (see dask_util module)
		   op_constructor = [no default] - pointer to function; Pointer to constructor
		   op_args = [no default] - list; List containing lists of arguments to run the constructor. It can instantiate the same operator on multiple workers or different ones if requested by passing a list of list of arguments (e.g., [(arg1,arg2,arg3,...)])
		   chunks = [no default] - list; List defininig how many operators wants to instantiated. Note, the list must contain the same number of elements as the number of Dask worker in the DaskClient.
		"""
		#Client to submit tasks
		if not isinstance(dask_client,DaskClient):
			raise TypeError("Passed client is not a Dask Client object!")
		if not isinstance(op_args,list):
			raise TypeError("Passed operator arguments not a list!")
		self.dask_client = dask_client
		self.client = self.dask_client.getClient()
		wrkIds = dask_client.getWorkerIds()
		N_wrk = dask_client.getNworkers()
		#Check if number of provided chunks is the same as workers
		if(len(chunks) != N_wrk):
			raise ValueError("Number of provide chunks (%s) different than the number of workers (%s)"%(len(chunks),N_wrk))
		#Check if many arguments are passed to construct different operators
		N_args = len(op_args)
		N_ops = np.sum(chunks)
		if N_args > 1:
			if N_args != N_ops:
				raise ValueError("Number of lists of arguments (%s) different than the number of requested operators (%s)"%(N_args,N_ops))
		else:
			if N_ops > 1:
				op_args = [op_args for ii in range(N_ops)]
		#Instantiation of the operators on each worker
		self.dask_ops = []
		for iwrk,wrkId in enumerate(wrkIds):
			for iop in range(chunks[iwrk]):
				self.dask_ops.append(self.client.submit(call_constructor,op_constructor,op_args.pop(0),workers=[wrkId],pure=False))
		daskD.wait(self.dask_ops)
		#Creating domain and range of the Dask operator
		dom_vecs = [] #List of remote domain vectors
		rng_vecs = [] #List of remote range vectors
		for op in self.dask_ops:
			dom_vecs.append(self.client.submit(call_getDomain,op,pure=False))
			rng_vecs.append(self.client.submit(call_getRange,op,pure=False))
		daskD.wait(dom_vecs+rng_vecs)
		self.domain = DaskVector(self.dask_client,dask_vectors=dom_vecs)
		self.range = DaskVector(self.dask_client,dask_vectors=rng_vecs)
		return


	def forward(self,add,model,data):
		"""Forward Dask operator"""
		if not isinstance(model,DaskVector):
			raise TypeError("Model vector must be a DaskVector!")
		if not isinstance(data,DaskVector):
			raise TypeError("Data vector must be a DaskVector!")
		#Dimensionality check
		self.checkDomainRange(model,data)
		fwd_ftr = []
		for iop,op in enumerate(self.dask_ops):
			fwd_ftr.append(self.client.submit(call_forward,op,add,model.vecDask[iop],data.vecDask[iop],pure=False))
		daskD.wait(fwd_ftr)
		return

	def adjoint(self,add,model,data):
		"""Adjoint Dask operator"""
		if not isinstance(model,DaskVector):
			raise TypeError("Model vector must be a DaskVector!")
		if not isinstance(data,DaskVector):
			raise TypeError("Data vector must be a DaskVector!")
		#Dimensionality check
		self.checkDomainRange(model,data)
		adj_ftr = []
		for iop,op in enumerate(self.dask_ops):
			adj_ftr.append(self.client.submit(call_adjoint,op,add,model.vecDask[iop],data.vecDask[iop],pure=False))
		daskD.wait(adj_ftr)
		return
