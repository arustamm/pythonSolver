import pyVector as Vector
import pyOperator as Operator
from typing import List
import types
import numpy as np
import Hypercube
from dask_util import DaskClient
from dask.distributed import wait, as_completed
import copy

class DaskObject:
    """ Remote object based on the Ray runtime """
    
    def __init__(self, dask_client, 
                objCreator,  constructor_pars: List[dict], 
                futures: List=None,
                **kw):
        """
        """
        #  Client to submit tasks
        if not isinstance(dask_client, DaskClient):
            raise TypeError("Passed client is not a Dask Client object!")
        self.dask_client = dask_client
        self.client = client = self.dask_client.getClient()

        # List containing references to ray-remote actors
        self.obj = []
        self.client = dask_client.getClient()
        # Creating a ray-remote wrapper for object
        if not isinstance(objCreator, type) and not isinstance(objCreator, types.FunctionType):
            raise TypeError("DaskObject can only be created by providing the class name or creator-function!")
        if futures and len(futures) == len(constructor_pars):
            self.obj = futures
        else:
            for par in constructor_pars:
                future = client.submit(objCreator, **par)
                # collect all vectors into the pool
                self.obj.append(future)
        
        self.cls = objCreator

    def __getitem__(self, index):
        return self.obj[index]

    def __len__(self):
        return len(self.obj)

    def __del__(self):
        """Default destructor"""


class DaskVector(DaskObject, Vector.vector):

    def __init__(self, dask_client, 
                vecCls, ns: List[int], os: List[float], ds: List[float], 
                chunks=None, futures=None, **kw):
        """
            veCls -- Vector class that MUST take kwargs as [ns=..., ds=..., os=...]
            chunks -- List corresponding to number of chunks along each dimension
        """
        #

        self.ns = ns
        self.os = os
        self.ds = ds
        self.chunks = chunks

        # spread vectors across ray-workers if chunks is present
        if chunks: 
            nchunks = np.prod(np.array(chunks))
            # size of an individual chunk
            nns = np.array(ns) // np.array(chunks)
            ns_list = np.asarray([nns for i in range(nchunks)], dtype=object)
            ds_list = np.asarray([ds for i in range(nchunks)], dtype=object)
            os_list = np.asarray([os for i in range(nchunks)], dtype=object)
            
            # handle the chunks at the boundaries 
            every_index = np.flip(np.cumprod(chunks))
            before = np.ones(len(chunks))
            before[:-1] = every_index[1:]
            for i in range(len(chunks)):
                # calculate origins
                oos = np.array([os[i] + j*nns[i]*ds[i] for j in range(chunks[i])])
                oos = np.repeat(oos, before[i])
                oos = np.tile(oos, len(os_list)//oos.size)
                os_list[:,i] = oos[:]
                # calculate sizes at the boundaries
                for j in range(every_index[i],len(ns_list)+every_index[i],every_index[i]):
                    sublist = ns_list[int(j-before[i]):j]
                    sublist[:,i] += ns[i] % chunks[i]
        # else scatter the vector across workers
        else:
            nchunks = dask_client.getNworkers()
            ns_list = np.asarray([ns for i in range(nchunks)], dtype=object)
            ds_list = np.asarray([ds for i in range(nchunks)], dtype=object)
            os_list = np.asarray([os for i in range(nchunks)], dtype=object)
        
        # list of parameters for each inividual Vector 
        constructor_pars = [{"fromHyper" : Hypercube.hypercube(ns=ns.tolist(), os=os.tolist(), ds=ds.tolist())} 
                                for (ns, os, ds) in zip(ns_list, os_list, ds_list)]

        DaskObject.__init__(self, dask_client, vecCls, constructor_pars, futures=futures)


    def getNdArray(self):
        # maybe return a dask array instead?
        fut = self.client.map(self.cls.getNdArray, self, pure=False)
        return self.client.gather(fut)

    @property
    def shape(self):
        arrs = self.getNdArray()
        return [arr.shape for arr in arrs]

    @property
    def size(self):
        return np.prod(self.ns)

    def norm(self, N=2):
        norm = 0.
        fut = self.client.map(self.cls.norm, self, N=N, pure=False)
        for future, result in as_completed(fut, with_results=True):
            norm += np.power(np.float64(result), N)
        return np.power(norm, 1. / N)

    def zero(self):
        wait(self.client.map(self.cls.zero, self, pure=False))
        return self

    def max(self):
        """Function to obtain maximum value within a vector"""
        maxs = self.client.gather(self.client.map(self.cls.max, self, pure=False))
        return np.array(maxs).max()

    def min(self):
        """Function to obtain minimum value within a vector"""
        mins = self.client.gather(self.client.map(self.cls.min, self, pure=False))
        return np.array(mins).min()

    def set(self, val):
        """Function to set all values in the vector"""
        wait(self.client.map(self.cls.set, self, val=val, pure=False))
        return self

    def scale(self, sc):
        """Function to scale a vector"""
        wait(self.client.map(self.cls.scale, self, sc=sc, pure=False))
        return self

    def addbias(self, bias):
        """Function to add bias to a vector"""
        wait(self.client.map(self.cls.addbias, self, bias=bias, pure=False))
        return self

    def rand(self):
        """Function to randomize a vector"""
        wait(self.client.map(self.cls.rand, self, pure=False))
        return self

    def abs(self):
        """Return a vector containing the absolute values"""
        wait(self.client.map(self.cls.abs, self, pure=False))
        return self

    def sign(self):
        """Return a vector containing the signs"""
        wait(self.client.map(self.cls.sign, self, pure=False))
        return self

    def reciprocal(self):
        """Return a vector containing the reciprocals of self"""
        wait(self.client.map(self.cls.reciprocal, self, pure=False))
        return self

    def conj(self):
        """Compute conjugate transpose of the vector"""
        wait(self.client.map(self.cls.conj, self, pure=False))
        return self

    def real(self):
        """Return the real part of the vector"""
        wait(self.client.map(self.cls.real, self, pure=False))
        return self

    def imag(self):
        """Return the imaginary part of the vector"""
        wait(self.client.map(self.cls.real, self, pure=False))
        return self

    def pow(self, power):
        """Compute element-wise power of the vector"""
        wait(self.client.map(self.cls.pow, self, power=power, pure=False))
        return self

    # Methods combinaning different vectors

    def clone(self):
        """Function to clone (deep copy) a vector from a vector or a Space"""
        fut = self.client.map(self.cls.clone, self, pure=False)
        return DaskVector(self.dask_client, self.cls, self.ns, self.os, self.ds, 
                            chunks=self.chunks, futures=fut)

    def cloneSpace(self):
        """Function to clone vector space"""
        fut = self.client.map(self.cls.cloneSpace, self, pure=False)
        return DaskVector(self.dask_client, self.cls, self.ns, self.os, self.ds, 
                            chunks=self.chunks, futures=fut)

    def check(self, vec):
        # check if number of chunks is the same
        if not isinstance(vec, DaskVector):
            raise TypeError("Vector is not a DaskVector!")
        if len(self) != len(vec):
            raise ValueError(
            "Number of chunks is different! (self chunks %s; vec2 chunks %s)" % (
                len(self), len(vec)))

    def checkSame(self, vec):
        """Function to check to make sure the vectors exist in the same space"""
        self.check(vec)
        fut = self.client.map(self.cls.checkSame, self, vec, pure=False)
        return all(self.client.gather(fut))
        
    def maximum(self, vec2):
        """Return a new vector of element-wise maximum of self and vec2"""
        self.check(vec2)
        wait(self.client.map(self.cls.maximum, self, vec2, pure=False))
        return self

    def copy(self, vec2):
        """Function to copy vector"""
        self.check(vec2)
        wait(self.client.map(self.cls.copy, self, vec2, pure=False))
        return self

    def scaleAdd(self, vec2, sc1=1.0, sc2=1.0):
        """Function to scale two vectors and add them to the first one"""
        self.check(vec2)
        wait(self.client.map(self.cls.scaleAdd, self, vec2, [sc1]*len(self), [sc2]*len(self), pure=False))
        return self

    def dot(self, vec2):
        """Function to compute dot product between two vectors"""
        self.check(vec2)
        dots = self.client.map(self.cls.dot, self, vec2, pure=False)
        # Adding all the results together
        dot = 0.0
        for future, result in as_completed(dots, with_results=True):
            dot += result
        return dot

    def multiply(self, vec2):
        """Function to multiply element-wise two vectors"""
        self.check(vec2)
        wait(self.client.map(self.cls.multiply, self, vec2, pure=False))
        return self

    def isDifferent(self, vec2):
        """Function to check if two vectors are identical"""
        self.check(vec2)
        fut = self.client.map(self.cls.isDifferent, self, vec2, pure=False)
        results = self.client.gather(fut)
        return any(results)

    def clipVector(self, low, high):
        """Function to bound vector values based on input vectors min and max"""
        self.check(low)  # Checking low-bound vector
        self.check(high)  # Checking high-bound vector
        wait(self.client.map(self.cls.clipVector, self, low, high, pure=False))
        return self


def scatter(vector):
    pass

# class RayOperator(RayObject, Operator.Operator):

#     def __init__(self, domain, range, opCls, constructor_par, **kw):
#         RayObject.__init__(opCls, constructor_par, kw)

#     def forward(self, add, model, data):
#         self.objRemote.map(...)
             

#     def adjoint(self, add, model, data):
#         pass 


    