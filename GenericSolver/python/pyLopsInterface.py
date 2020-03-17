import numpy as np
import pyOperator as pyOp
import pyVector as pyVec
try:
    import pylops
except ImportError:
    import os
    os.system("pip install --user pylops")
    import pylops


class ToPylops(pylops.LinearOperator):
    
    def __init__(self, op):
        """
        Cast an Operator to pylops.LinearOperator
        :param op: `pyOperator.Operator` object (or child)
        """
        assert isinstance(op, pyOp.Operator), 'op has to be a pyOperator.Operator'
        self.shape = (op.range.size, op.domain.size)
        self.dtype = op.domain.getNdArray().dtype
        self.explicit = False
        
        self.forfunc = op.forward
        self.adjfunc = op.adjoint
        self.range_shape = op.range.shape
        self.domain_shape = op.domain.shape
    
    def _matvec(self, x):
        model = pyVec.vectorIC(x.reshape(self.domain_shape).astype(self.dtype))
        data = pyVec.vectorIC(np.empty(self.range_shape, dtype=self.dtype))
        self.forfunc(False, model, data)
        return data.getNdArray().copy()
    
    def _rmatvec(self, y):
        model = pyVec.vectorIC(np.empty(self.domain_shape, dtype=self.dtype))
        data = pyVec.vectorIC(y.reshape(self.range_shape).astype(self.dtype))
        self.adjfunc(False, model, data)
        return model.getNdArray().copy()
