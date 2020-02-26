# Module containing the definition of the operator necessary for the solver class
# It takes vector objects from the pyVector class

from __future__ import division, print_function, absolute_import
import numpy as np
from pyVector import vector
from pyOperator import Operator
import sep_util
try:
    import cupy as cp
except ModuleNotFoundError:
    import subprocess
    import sys
    subprocess.call([sys.executable, "-m", "pip", "install", "--user", "cupy==7.2.0"])
    import cupy as cp


class MatrixOp(Operator):
    """Operator built upon a matrix"""

    def __init__(self, matrix, domain, range, outcore=False):
        """Class constructor
        :param matrix   : matrix to use
        :param domain   : domain vector
        :param range    : range vector
        :param outcore  : use outcore sep operators
        """
        if not isinstance(domain, vector):
            raise TypeError("ERROR! Domain vector not a vector object")
        if not isinstance(range, vector):
            raise TypeError("ERROR! Range vector not a vector object")
        # Setting domain and range of operator and matrix to use during application of the operator
        self.setDomainRange(domain, range)
        if not (isinstance(matrix, np.ndarray) or isinstance(matrix, cp.ndarray)):
            raise ValueError("ERROR! matrix has to be a numpy/cupy ndarray")
        self.M = matrix
        self.outcore = outcore
        self.arr_module = cp.get_array_module(matrix)
        
    def __str__(self):
        return "MatrixOp"
    
    def forward(self, add, model, data):
        """d = A * m"""
        self.checkDomainRange(model, data)
        if not add:
            data.zero()
        model_arr = model.getNdArray()
        if self.outcore:
            [data_arr, data_axis] = sep_util.read_file(data.vecfile)
            data_arr += self.arr_module.matmul(self.M, model_arr.ravel()).reshape(data_arr.shape)
            sep_util.write_file(data.vecfile, data_arr, data_axis)
        else:
            data_arr = data.getNdArray()
            data_arr += self.arr_module.matmul(self.M, model_arr.ravel()).reshape(data_arr.shape)
        return

    def adjoint(self, add, model, data):
        """m = A' * d"""
        self.checkDomainRange(model, data)
        if not add:
            model.zero()
        data_arr = data.getNdArray()
        if self.outcore:
            [model_arr, model_axis] = sep_util.read_file(model.vecfile)
            model_arr += self.arr_module.matmul(self.M.T.conj(), data_arr.ravel()).reshape(model_arr.shape)
            sep_util.write_file(model.vecfile, model_arr, model_axis)
        else:
            model_arr = model.getNdArray()
            model_arr += self.arr_module.matmul(self.M.T.conj(), data_arr.ravel()).reshape(model_arr.shape)
        return
    
    def getNdArray(self):
        return self.arr_module.array(self.M)


class FirstDerivative(Operator):
    def __init__(self, model, sampling=1., axis=0):
        r"""
        Compute 2nd order centered first derivative

        .. math::
            y[i] = 0.5 (x[i+1] - x[i-1]) / dx

        :param model    : vector class; domain vector
        :param sampling : scalar; sampling step [1.]
        :param axis     : int; axis along which to compute the derivative [0]
        """
        self.sampling = sampling
        self.data_tmp = model.clone().zero()
        self.dims = model.getNdArray().shape
        self.axis = axis if axis >= 0 else len(self.dims) + axis
        super(FirstDerivative, self).__init__(model, model)
    
    def __str__(self):
        return "1stDer_%d" % self.axis
    
    def forward(self, add, model, data):
        """Forward operator"""
        self.checkDomainRange(model, data)
        if add:
            self.data_tmp.copy(data)
        data.zero()
        # Getting Ndarrays
        x = model.clone().getNdArray()
        if self.axis > 0:  # need to bring the dim. to derive to first dim
            x = cp.get_array_module(x).swapaxes(x, self.axis, 0)
        y = cp.get_array_module(x).zeros(x.shape)
        
        y[:-1] = (x[1:] - x[:-1]) / self.sampling / 2
        
        if self.axis > 0:  # reset axis order
            y = cp.get_array_module(y).swapaxes(y, 0, self.axis)
        data.getNdArray()[:] = y
        if add:
            data.scaleAdd(self.data_tmp)
        return
    
    def adjoint(self, add, model, data):
        """Adjoint operator"""
        self.checkDomainRange(model, data)
        if add:
            self.data_tmp.copy(model)
        model.zero()
        # Getting Ndarrays
        y = data.clone().getNdArray().reshape(self.dims)
        if self.axis > 0:  # need to bring the dim. to derive to first dim
            y = cp.get_array_module(y).swapaxes(y, self.axis, 0)
        x = cp.get_array_module(y).zeros(y.shape)
        
        x[0] = -y[0] / self.sampling / 2
        x[1:-1] = (-y[1:-1] + y[:-2]) / self.sampling / 2
        x[-1] = y[-2] / self.sampling / 2
        
        if self.axis > 0:
            x = cp.get_array_module(x).swapaxes(x, 0, self.axis)
        model.getNdArray()[:] = x
        if add:
            model.scaleAdd(self.data_tmp)
        return


class SecondDerivative(Operator):
    def __init__(self, model, sampling=1., axis=0):
        r"""
        Compute 2nd order second derivative

        .. math::
            y[i] = (x[i+1] - 2x[i] + x[i-1]) / dx^2

        :param model    : vector class; domain vector
        :param sampling : scalar; sampling step [1.]
        :param axis     : int; axis along which to compute the derivative [0]
        """
        self.sampling = sampling
        self.data_tmp = model.clone().zero()
        self.dims = model.getNdArray().shape
        self.axis = axis if axis >= 0 else len(self.dims) + axis
        super(SecondDerivative, self).__init__(model, model)
    
    def __str__(self):
        return "2ndDer_%d" % self.axis
    
    def forward(self, add, model, data):
        """Forward operator"""
        self.checkDomainRange(model, data)
        if add:
            self.data_tmp.copy(data)
        data.zero()
        
        # Getting Ndarrays
        x = model.clone().getNdArray()
        if self.axis > 0:  # need to bring the dim. to derive to first dim
            x = cp.get_array_module(x).swapaxes(x, self.axis, 0)
        y = cp.get_array_module(x).zeros_like(x)
        
        y[1:-1] = (x[0:-2] - 2 * x[1:-1] + x[2:]) / self.sampling ** 2
        
        if self.axis > 0:  # reset axis order
            y = cp.get_array_module(y).swapaxes(y, 0, self.axis)
        data.getNdArray()[:] = y
        if add:
            data.scaleAdd(self.data_tmp)
        return
    
    def adjoint(self, add, model, data):
        """Adjoint operator"""
        self.checkDomainRange(model, data)
        if add:
            self.data_tmp.copy(model)
        model.zero()
        
        # Getting numpy arrays
        y = data.clone().getNdArray()
        if self.axis > 0:  # need to bring the dim. to derive to first dim
            y = cp.get_array_module(y).swapaxes(y, self.axis, 0)
        x = cp.get_array_module(y).zeros_like(y)
        
        x[0:-2] += y[1:-1] / self.sampling ** 2
        x[1:-1] -= 2 * y[1:-1] / self.sampling ** 2
        x[2:] += y[1:-1] / self.sampling ** 2
        
        if self.axis > 0:
            x = cp.get_array_module(x).swapaxes(x, 0, self.axis)
        model.getNdArray()[:] = x
        if add:
            model.scaleAdd(self.data_tmp)
        return


class TotalVariation(Operator):
    def __init__(self, model, axis=None, weights=None, sampling=None, iso=False):
        r"""
        (An)isotropic Total Variation operator.
        The input parameters are tailored for >2D, but it works also for 1D.

        :param model    : vector class; domain vector
        :param axis     : tuple; axis along which to compute the derivative [0, 1]
        :param weights  : tuple; scalar weight for the axis [1, 1]
        :param sampling : tuple; sampling step [1, 1]
        :param iso      : bool; compute isotropic operator [False]
        """
        self.dims = model.getNdArray().shape
        self.axis = axis if axis is not None else tuple(range(len(self.dims)))
        self.sampling = sampling if sampling is not None else tuple([1] * len(self.dims))
        self.weights = weights if weights is not None else tuple([1] * len(self.dims))
        
        assert len(self.axis) == len(self.weights) == len(self.sampling) != 0, \
            "There is something wrong with the dimensions"
        
        self.isotropic = iso
        
        if self.isotropic:  # self.op is a list of operators
            self.op = [self.weights[d] * FirstDerivative(model, sampling=self.sampling[d], axis=self.axis[d]) for d in
                       range(len(self.axis))]
        else:  # self.op is itself the final operator
            self.op = self.weights[0] * FirstDerivative(model, sampling=self.sampling[0], axis=self.axis[0])
            for d in range(1, len(self.axis)):
                self.op += self.weights[d] * FirstDerivative(model, sampling=self.sampling[d], axis=self.axis[d])
        
        super(TotalVariation, self).__init__(model, model)
    
    def __str__(self):
        return "TotalVar"
    
    def forward(self, add, model, data):
        if self.isotropic:
            self.checkDomainRange(model, data)
            if add:
                data_tmp = data.clone()
            data.zero()
            for op in self.op:
                temp = data.clone().zero()
                op.forward(False, model, temp)
                data.scaleAdd(temp.pow(2))
            data.pow(.5)
            if add:
                data.scaleAdd(data_tmp)
            return
        else:
            return self.op.forward(add, model, data)
    
    def adjoint(self, add, model, data):
        if self.isotropic:
            raise ValueError("ERROR! The Isotropic Total Variation is nonlinear.")
        else:
            return self.op.adjoint(add, model, data)


class Laplacian(Operator):
    def __init__(self, model, axis=None, weights=None, sampling=None):
        r"""
        Laplacian operator.
        The input parameters are tailored for >2D, but it works also for 1D.

        :param model    : vector class; domain vector
        :param axis     : tuple; axis along which to compute the derivative [all]
        :param weights  : tuple; scalar weights for the axis [1 for each model axis]
        :param sampling : tuple; sampling step [1 for each model axis]
        """
        self.dims = model.getNdArray().shape
        self.axis = axis if axis is not None else tuple(range(len(self.dims)))
        self.sampling = sampling if sampling is not None else tuple([1] * len(self.dims))
        self.weights = weights if weights is not None else tuple([1] * len(self.dims))
        
        assert len(self.axis) == len(self.weights) == len(self.sampling) != 0, \
            "There is something wrong with the dimensions"
        
        self.data_tmp = model.clone().zero()
        
        self.op = self.weights[0] * SecondDerivative(model, sampling=self.sampling[0], axis=self.axis[0])
        for d in range(1, len(self.axis)):
            self.op += self.weights[d] * SecondDerivative(model, sampling=self.sampling[d], axis=self.axis[d])
        super(Laplacian, self).__init__(model, model)
    
    def __str__(self):
        return "Laplace "
    
    def forward(self, add, model, data):
        return self.op.forward(add, model, data)
    
    def adjoint(self, add, model, data):
        return self.op.adjoint(add, model, data)


# TODO not finished yet
class Convolution(Operator):
    """ND Convolution through Cupy"""
    
    def __init__(self, domain, kernel):
        """Class constructor
        :param domain   : domain vector
        :param kernel   : kernel vector
        """
        try:
            import chainer.functions as F
        except ModuleNotFoundError:
            import subprocess
            import sys
            subprocess.call([sys.executable, "-m", "pip", "install", "--user", "chainer==7.2.0"])
            import chainer.functions as F
        
        if not isinstance(domain, vector):
            raise TypeError("ERROR! Domain vector not a vector object")
        # Setting domain and range of operator and matrix to use during application of the operator
        self.setDomainRange(domain, domain)
        if not (isinstance(kernel, np.ndarray) or isinstance(kernel, cp.ndarray)):
            raise ValueError("ERROR! matrix has to be a numpy/cupy ndarray")
        
        self.kernel = kernel.getNdArray()
       
        self.pad = tuple([(domain.getNdArray().shape[d] - kernel.shape[d])//2
                          for d in range(len(domain.getNdArray().shape))])
    
    def __str__(self):
        return " ConvOp "
    
    def forward(self, add, model, data):
        """d = A * m"""
        self.checkDomainRange(model, data)
        if not add:
            data.zero()
        model_arr = model.getNdArray()
        if self.outcore:
            [data_arr, data_axis] = sep_util.read_file(data.vecfile)
            data_arr += self.arr_module.matmul(self.M, model_arr.ravel()).reshape(data_arr.shape)
            sep_util.write_file(data.vecfile, data_arr, data_axis)
        else:
            data_arr = data.getNdArray()
            data_arr += F.deconvolution_nd(model_arr, self.kernel)
        return
    
    def adjoint(self, add, model, data):
        """m = A' * d"""
        self.checkDomainRange(model, data)
        if not add:
            model.zero()
        data_arr = data.getNdArray()
        if self.outcore:
            [model_arr, model_axis] = sep_util.read_file(model.vecfile)
            model_arr += self.arr_module.matmul(self.M.T.conj(), data_arr.ravel()).reshape(model_arr.shape)
            sep_util.write_file(model.vecfile, model_arr, model_axis)
        else:
            model_arr = model.getNdArray()
            model_arr += self.arr_module.matmul(self.M.T.conj(), data_arr.ravel()).reshape(model_arr.shape)
        return
    
    def getNdArray(self):
        return self.arr_module.array(self.M)