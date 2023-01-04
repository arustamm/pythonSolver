import ray
import pyVector as Vector
import pyOperator as Operator
from ray.util.actor_pool import ActorPool
from typing import List
import types
import numpy as np

class RayObject:
    """ Remote object based on the Ray runtime """
    
    def __init__(self, objCls,  constructor_pars: List[dict], **kw):
        """
        """

        # List containing references to ray-remote actors
        self.actors = []
        # Creating a ray-remote wrapper for object
        if isinstance(objCls, type) or isinstance(objCls, types.FunctionType):
            objRemote = ray.remote(objCls)
        else:
            raise RuntimeError("RayObject can only be created by providing the class name or the function creating the object!")
        
        for par in constructor_pars:
            # creating actual instances of remote vectors
            print(objRemote)
            vecHandle = objRemote.remote(**par)
            print(vecHandle)
            # collect all vectors into the pool
            self.actors.append(vecHandle)
        
        # TODO
        # wait for all the actors to be created?

    def __del__(self):
        """Default destructor"""


class RayVector(RayObject, Vector.vector):

    def __init__(self, vecCls, ns: List[int], os: List[float], ds: List[float], **kw):
        """
            veCls -- Vector class that MUST take kwargs as [ns=..., ds=..., os=...]
            chunks -- List corresponding to number of chunks along each dimension
        """
        if not ray.is_initialized():
            raise RuntimeError("Need to initialize Ray cluster before creating RayVector!")
        # spread vectors across ray-workers if chunks is present
        if "chunks" in kw: 
            chunks = kw.get('chunks')
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
            # TODO 
            # can i store in the Plasma object store ??
            # BEWARE that this causes serialization unless the data is a numpy array 
            nchunks = len(ray.nodes())
            ns_list = np.asarray([ns for i in range(nchunks)], dtype=object)
            ds_list = np.asarray([ds for i in range(nchunks)], dtype=object)
            os_list = np.asarray([os for i in range(nchunks)], dtype=object)
        
        # list of parameters for each inividual Vector 
        constructor_pars = [{"ns" : ns.tolist(), "os": os.tolist(), "ds": ds.tolist()} 
                                for (ns, os, ds) in zip(ns_list, os_list, ds_list)]

        RayObject.__init__(self, vecCls, constructor_pars)


    def getNdArray(self):
        return ray.get([v.getNdArray.remote() for v in self.actors])

    @property
    def shape(self):
        return list(self.vecRay.submit(lambda v: v.shape.remote()))

    def norm(self, N=2):
        norm = 0.
        norms = list(self.actors.submit(lambda v, n: v.norm.remote(N=n), N))
        for n in norms:
            norm += np.power(np.float64(n), N)
        return np.power(norm, 1/N)

    def zero(self):
        self.vecRay.submit(lambda v: v.zero.remote())
        return self



# class RayOperator(RayObject, Operator.Operator):

#     def __init__(self, domain, range, opCls, constructor_par, **kw):
#         RayObject.__init__(opCls, constructor_par, kw)

#     def forward(self, add, model, data):
#         self.objRemote.map(...)
             

#     def adjoint(self, add, model, data):
#         pass 


    