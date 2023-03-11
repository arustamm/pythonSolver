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
    
    def __init__(self, dask_client, operator_cls, domain, range, *args, **kw):
        if not isinstance(domain, DaskVector):
            raise TypeError("Domain vector must be a DaskVector!")
        if not isinstance(range, DaskVector):
            raise TypeError("Range vector must be a DaskVector!")
        
        client = dask_client.getClient()
        opCls = operator_cls
        if not "from_subspace" in dir(opCls):
            raise ValueError("To generate DaskOperator from %s, it should contain from_subspace function!" % opCls)
        op_args = []
        op_kwargs = []

        dom, ran = self._prepare_spaces_(domain.get_futures(), range.get_futures())
        for d,r in zip(dom, ran) :
            param = []
            param.append(client.submit(lambda x: x, d, pure=False))
            param.append(client.submit(lambda x: x, r, pure=False))
            for p in list(args):
                param.append(p)
            op_args.append(tuple(param))
            op_kwargs.append(kw)

        DaskObject.__init__(self, dask_client, objCreator=opCls.from_subspace, 
                            constructor_args=op_args, constructor_kw=op_kwargs, from_object=opCls)
        self.setDomainRange(domain, range)


    def _prepare_spaces_(self, domain, range):
        arr = np.array(np.meshgrid(domain, range)).reshape(2,-1)
        return arr[0,:], arr[1,:]

    def check(self, model, data):
        if not isinstance(model, DaskVector):
            raise TypeError("Model vector must be a DaskVector!")
        if not isinstance(data, DaskVector):
            raise TypeError("Data vector must be a DaskVector!")
        
    def as_matrix(self):
        return np.array(self.fut).reshape(self.range.nchunks, self.domain.nchunks)

    def forward(self, add, model, data):

        self.check(model, data)
        self.checkDomainRange(model, data)
        if not add: data.zero()
        
        mod = model.get_futures()
        dat = data.get_futures()
        ops = self.as_matrix()
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
        data.set_futures(dat)

    def adjoint(self, add, model, data):

        self.check(model, data)
        self.checkDomainRange(model, data)
        if not add: model.zero()
        
        mod = model.get_futures()
        dat = data.get_futures()
        ops = self.as_matrix().T
        # submit all tasks
        res = []
        for i, d in enumerate(dat):
            fut = self.client.map(adj, ops[:,i], mod, [d]*len(mod), pure=False)
            res.append(fut)
        # accumulate 
        for m in res:
            mod = self.client.map(model.cls.__add__, mod, m, pure=False)
        # copy the futures
        model.set_futures(mod)

    def set_background(self, model):
        mod = model.get_futures()
        ops = self.as_matrix()
        # submit all tasks
        res = []
        # loop across model chunks 
        for i, m in enumerate(mod):
            fut = self.client.map(set_bg, ops[:,i],[m]*ops.shape[0], pure=False)
            res.append(fut)
        wait(*res)

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
        # TODO not the best solution 
        m = model.clone()
        m.zero()
        return m

def set_bg(op, model):
    op.set_background(model)
    return True