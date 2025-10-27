import pyVector as Vector
import pyOperator as Operator
from typing import List, Tuple
import types
import numpy as np
import Hypercube
from dask.distributed import wait, as_completed
from dask import delayed
import dask.array as da
from __pyDaskVector import DaskVector
from __pyDaskObject import DaskObject
from pyVector import superVector
import functools as ft
import gc

class DaskOperator(DaskObject, Operator.Operator):
    
    def __init__(self, dask_client, operator_cls, domain, range, *args, **kw):
        if not isinstance(domain, DaskVector) and not isinstance(domain, superVector):
            raise TypeError("Domain vector must be a DaskVector or superVector!")
        if not isinstance(range, DaskVector) and not isinstance(range, superVector):
            raise TypeError("Range vector must be a DaskVector or superVector!")
        
        client = dask_client.getClient()
        opCls = operator_cls
        if not "from_subspace" in dir(opCls):
            raise ValueError("To generate DaskOperator from %s, it should contain from_subspace function!" % opCls)
        op_args = []
        op_kwargs = []
        
        dom, ran = self._prepare_spaces_(domain.get_futures(), range.get_futures())
        for d,r in zip(dom, ran) :
            param = []
            param.append(d)
            param.append(r)
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
        if not isinstance(model, DaskVector) and not isinstance(model, superVector):
            raise TypeError("Model vector must be a DaskVector or superVector!")
        if not isinstance(data, DaskVector) and not isinstance(data, superVector):
            raise TypeError("Data vector must be a DaskVector or superVector!")
        
    # def as_matrix(self):
    #     return np.array(self.fut).reshape(self.range.nchunks, self.domain.nchunks)

    def _tree_reduce(self, futures_list: list):
        """
        Performs a memory-efficient parallel tree reduction on a list of futures.

        This function avoids memory bottlenecks by never gathering all results
        on a single worker. It works like a tournament bracket:
        1. Pairs up items in the list and submits tasks to add them.
        2. The list of results becomes the input for the next round.
        3. This continues until only one future remains.

        Args:
            futures_list (list): A list of Dask futures to be summed.

        Returns:
            A single Dask future representing the final sum.
        """
        q = list(futures_list)
        while len(q) > 1:
            # Pop two futures from the end of the list
            fut1 = q.pop()
            fut2 = q.pop()
            
            # Submit a new task to add the results of the two futures
            sum_fut = self.client.submit(lambda m1, m2: m1 + m2, fut1, fut2, pure=False, retries=3)
            
            # Add the new future representing the sum back to the front of the list
            q.insert(0, sum_fut)
            
        # The last remaining future is the final result
        return q[0]

    def forward(self, add, model, data):

        # self.check(model, data)
        # self.checkDomainRange(model, data)
        if not add: data.zero()
        mod = model.get_futures()
        # if isinstance(model, DaskVector):
        #     self.client.replicate(mod)
        # else:
        mod = self.client.scatter(mod, broadcast=True)
        dat = data.get_futures()
        # ops = self.as_matrix()
        # submit all tasks
        res = [dat]
        # loop across model chunks 
        for i, m in enumerate(mod):
            fut = self.client.map(fwd, self.fut, [m]*len(dat), dat, [False]*len(dat), pure=False, retries=3)
            res.append(fut)
        # waitable = [f for sublist in res for f in sublist]
        # wait(waitable)
        # accumulate 
        fin = ft.reduce(lambda d1, d2: self.client.map(data.cls.__add__, d1, d2, pure=False, retries=3), res)
        # copy the futures
        # dd = self.client.map(data.cls.scaleAdd, dat, fin, pure=False)
        data.set_futures(fin)
        # del dat, res, waitable, fut
        gc.collect()  # Clean up memory

    def adjoint(self, add, model, data):

        # self.check(model, data)
        # self.checkDomainRange(model, data)
        if not add: model.zero()
        
        mod = model.get_futures()
        # if isinstance(model, DaskVector):
        #     self.client.replicate(mod)
        # else:
        mod = self.client.scatter(mod, broadcast=True)
        dat = data.get_futures()
        # ops = self.as_matrix()
        # submit all tasks
        res = []
        for i, m in enumerate(mod):
            fut = self.client.map(adj, self.fut, [m]*len(dat), dat, [False]*len(dat), pure=False, retries=3)
            res.append(fut)
        # waitable = [f for sublist in res for f in sublist]
        # wait(waitable)
        #accumulate 
        # fin = []
        # for m in res:
        #     mm = self.client.submit(ft.reduce, lambda m1, m2: m1+m2, m, pure=False)
        #     fin.append(mm)
        fin = [self._tree_reduce(m) for m in res]
        # copy the futures
        mm = self.client.map(model.cls.scaleAdd, mod, fin, pure=False, retries=3)
        model.set_futures(mm)
        # del mod, res, waitable, fin, fut
        gc.collect()  # Clean up memory

    def set_background(self, model):
        self.domain.checkSame(model)
        mod = model.get_futures()
        # if isinstance(model, DaskVector):
        #     self.client.replicate(mod)
        # else:
        mod = self.client.scatter(mod, broadcast=True)
        # ops = self.as_matrix()
        # submit all tasks
        res = []
        # loop across model chunks 
        for i, m in enumerate(mod):
            fut = self.client.map(set_bg, self.fut,[m]*len(self), pure=False, retries=3)
            res.extend(fut)
        self.set_futures(res)
        gc.collect()  # Clean up memory

class DaskDiagonalOperator(DaskOperator):
    """
    A DaskOperator for diagonal operators where each data chunk depends only on
    the corresponding model chunk (e.g., TraceNormalization, Mask).
    """

    # def __init__(self, dask_client, operator_cls, domain, range, *args, **kw):
    #     super().__init__(dask_client, operator_cls, domain, range, args, kw)

    def _prepare_spaces_(self, domain_chunks, range_chunks):
        """
        Overrides the base method to create a 1-to-1 mapping (zip)
        instead of a Cartesian product (meshgrid).
        """
        if len(domain_chunks) != len(range_chunks):
            raise ValueError("For a diagonal operator, domain and range must have the same number of chunks.")
        return domain_chunks, range_chunks

    def forward(self, add, model, data):
        """
        Applies the forward operator chunk by chunk in a 1-to-1 fashion.
        """
        # self.check(model, data) # Basic checks
        # self.checkDomainRange(model, data)

        if not add: data.scale(0)

        mod = model.get_futures()
        dat = data.get_futures()
        ops = self.get_futures()

        # The 'add' flag is passed down to the helper function
        new_dat = self.client.map(fwd, ops, mod, dat, [add] * len(ops), pure=False, retries=3)
        
        data.set_futures(new_dat)
        gc.collect()

    def adjoint(self, add, model, data):
        """
        Applies the adjoint operator chunk by chunk in a 1-to-1 fashion.
        """
        # self.check(model, data)
        # self.checkDomainRange(model, data)

        if not add: model.scale(0)

        mod = model.get_futures()
        dat = data.get_futures()
        ops = self.get_futures()

        new_mod = self.client.map(adj, ops, mod, dat, [add] * len(ops), pure=False, retries=3)

        model.set_futures(new_mod)
        gc.collect()

    def set_background(self, model):
        """
        Sets the background for each operator chunk from the corresponding
        model chunk.
        """
        # self.domain.checkSame(model)
        mod = model.get_futures()
        ops = self.get_futures()

        new_ops = self.client.map(set_bg, ops, mod, pure=False, retries=3)
        
        self.set_futures(new_ops)
        gc.collect()

# Need helper functions because DaskOperator 
# is potentially a heterogeneous object (contains different types of Operators)
import time
def fwd(op, model, data, add):
    """ makes a copy to avoid race condition in reduction"""
    if not isinstance(op, Operator.DummyOp):
        d = data.clone()
        op.forward(add, model, d)
        return d
    else:
        # TODO not the best solution need to fix
        d = data.clone()
        d.zero()
        return d

def adj(op, model, data, add):
    """ makes a copy to avoid race condition in reduction"""
    if not isinstance(op, Operator.DummyOp):
        m = model.clone()
        op.adjoint(add, m, data)
        return m
    else:
        # TODO not the best solution 
        m = model.clone()
        m.zero()
        return m

def set_bg(op, model):
    op.set_background(model)
    return op

def set_domain(op, domain):
    op.setDomain(domain)
    return op

def set_range(op, range):
    op.setRange(range)
    return op

