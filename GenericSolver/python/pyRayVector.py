import ray
import pyVector as Vector
import pyOperator as Operator
from ray.util.actor_pool import ActorPool

class RayObject:
    """ Remote object based on the Ray runtime """
    
    def __init__(self, objCls,  *constructor_args, **constructor_kwargs, **kw):
        """
        """

        # List containing references to ray-remote actors
        self.actors = ActorPool([])
        # Creating a ray-remote wrapper for object
        objRemote = ray.remote(objCls)
        
        

                # creating actual instances of remote vectors
                vecHandle = vecRemote.remote()

                # collect all vectors into the pool
                self.actors.push(vecHandle)
        


    def __del__(self):
        """Default destructor"""


class RayVector(RayObject, Vector.vector):

    def __init__(self, vecCls, ns, os=None, ds=None, **kw):
        """
            chunks -- List corresponding to number of chunks along each dimension
        """
        # spread vectors across ray-workers if chunks is present
        if "chunks" in kw:  
            nchunks = np.prod(np.array(kw.get('chunks')))
            
            for i in range(nchunks):
                # calculate shapes
                nns = 

        # else scatter the vector across workers
        else:
            # TODO how to scatter across workers?
            # store in the Plasma object store
            # BEWARE that this causes serialization unless the data is a numpy array 
            ray.put(...)

        RayObject.__init__(vecCls, constructor_par, kw)


    def getNdArray(self):
        return list(self.vecRay.submit(lambda v: v.getNdArray.remote()))

    @property
    def shape(self):
        return list(self.vecRay.submit(lambda v: v.shape.remote()))

    def norm(self, N=2):
        norm = 0.
        norms = list(self.vecRay.submit(lambda v: v.norm.remote(), N=N))
        for n in norms:
            norm += np.power(np.float64(n), N)
        return np.power(norm, 1/N)

    def zero(self):
        self.vecRay.submit(lambda v: v.zero.remote())
        return self



class RayOperator(RayObject, Operator.Operator):

    def __init__(self, domain, range, opCls, constructor_par, **kw):
        RayObject.__init__(opCls, constructor_par, kw)

    def forward(self, add, model, data):
        self.objRemote.map(...)
             

    def adjoint(self, add, model, data):
        pass 


    