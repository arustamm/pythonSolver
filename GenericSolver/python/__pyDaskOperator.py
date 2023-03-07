import pyVector as Vector
import pyOperator as Operator
from typing import List, Tuple
import types
import numpy as np
import Hypercube
from dask_util import DaskClient
from dask.distributed import wait, as_completed
from dask import delayed
import dask.array as da
from __pyDaskVector import DaskObject, DaskVector

class DaskOperator(DaskObject, Operator.Operator):
    
    def __init__(self, dask_client, domain, range, broadcast_what='domain', **kw):
        if not isinstance(domain, DaskVector):
            raise TypeError("Domain vector must be a DaskVector!")
        if not isinstance(range, DaskVector):
            raise TypeError("Range vector must be a DaskVector!")

        if kw.get("opCls"):
            opCls = kw.get("opCls")
            if not "from_subspace" in dir(opCls):
                raise ValueError("To generate DaskOperator from %s, it should contain from_subspace function!" % opCls)
            op_params = []

            dom, ran = self._prepare_spaces_(domain.get_futures(), range.get_futures())
            for d,r in zip(dom, ran) :
                param = []
                param.append(d)
                param.append(r)
                if kw.get("op_args"):
                    for p in kw.get("op_args"):
                        param.append(p)
                op_params.append(tuple(param))
            
            DaskObject.__init__(self, dask_client, objCreator=opCls.from_subspace, constructor_args=op_params, from_object=opCls)
            Operator.Operator.__init__(self, domain, range)

    def _prepare_spaces_(self, domain, range):
        arr = np.array(np.meshgrid(domain, range)).reshape(2,-1)
        return arr[0,:], arr[1,:]

    def check(self, model, data):
        if not isinstance(model, DaskVector):
            raise TypeError("Model vector must be a DaskVector!")
        if not isinstance(data, DaskVector):
            raise TypeError("Data vector must be a DaskVector!")

    def forward(self, add, model, data):
        """ This is not blocking """

        self.check(model, data)
        # self.checkDomainRange(model, data)
        if not add:
            data.scale(0)
        
        mod = model.get_futures()
        dat = data.get_futures()
        ops = np.array(self.fut).reshape(data.nchunks, model.nchunks)
        # submit all tasks
        res = []
        # loop across model chunks 
        for i, m in enumerate(mod):
            fut = self.client.map(fwd, ops[:,i], [m]*len(dat), dat, pure=False)
            res.append(fut)
        # accumulate 
        for d in res:
            dat = self.client.map(data.cls.__add__, dat, d, pure=False)
        # copy the futures
        wait(dat)
        data.set_futures(dat)

    def adjoint(self, add, model, data):
        """ This is not blocking """

        self.check(model, data)
        # self.checkDomainRange(model, data)
        if not add:
            model.scale(0)
        
        mod = model.get_futures()
        dat = data.get_futures()
        ops = np.array(self.fut).reshape(data.nchunks, model.nchunks).T
        # submit all tasks
        res = []
        for i, d in enumerate(dat):
            fut = self.client.map(adj, ops[:,i], mod, [d]*len(mod), pure=False)
            res.append(fut)
        # accumulate 
        for m in res:
            mod = self.client.map(model.cls.__add__, mod, m, pure=False)
        # copy the futures
        wait(mod)
        model.set_futures(mod)

# Need helper functions because DaskOperator 
# is potentially a heterogeneous object (contains different types of Operators)
import time
def fwd(op, model, data):
    """ makes a copy to avoid race condition in reduction"""
    if not isinstance(op, Operator.DummyOp):
        d = data.clone()
        op.forward(False, model, d)
        return d
    else:
        # TODO not the best solution need to fix
        d = data.clone()
        d.zero()
        return d

def adj(op, model, data):
    """ makes a copy to avoid race condition in reduction"""
    if not isinstance(op, Operator.DummyOp):
        m = model.clone()
        op.adjoint(False, m, data)
        return m
    else:
        # TODO not the best solution need to fix
        m = model.clone()
        m.zero()
        return m