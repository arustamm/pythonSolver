# Module containing the definition of numpy-based operators

from __future__ import division, print_function, absolute_import
import numpy as np
import pyVector as pyVec
import pyOperator as pyOp

import sep_util
from scipy.signal import convolve, correlate
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import LinearOperator


class MatrixOp(pyOp.Operator):
    """Operator built upon a matrix"""
    
    def __init__(self, matrix, domain, range, outcore=False):
        """Class constructor
        :param matrix   : matrix to use
        :param domain   : domain vector
        :param range    : range vector
        :param outcore  : use outcore sep operators
        """
        if not isinstance(domain, pyVec.vector):
            raise TypeError("ERROR! Domain vector not a vector object")
        if not isinstance(range, pyVec.vector):
            raise TypeError("ERROR! Range vector not a vector object")
        # Setting domain and range of operator and matrix to use during application of the operator
        self.setDomainRange(domain, range)
        if not isinstance(matrix, np.ndarray):
            raise ValueError("ERROR! matrix has to be a numpy ndarray")
        self.M = matrix
        self.outcore = outcore
    
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
            data_arr += np.matmul(self.M, model_arr.ravel()).reshape(data_arr.shape)
            sep_util.write_file(data.vecfile, data_arr, data_axis)
        else:
            data_arr = data.getNdArray()
            data_arr += np.matmul(self.M, model_arr.ravel()).reshape(data_arr.shape)
        return
    
    def adjoint(self, add, model, data):
        """m = A' * d"""
        self.checkDomainRange(model, data)
        if not add:
            model.zero()
        data_arr = data.getNdArray()
        if self.outcore:
            [model_arr, model_axis] = sep_util.read_file(model.vecfile)
            model_arr += np.matmul(self.M.H, data_arr.ravel()).reshape(model_arr.shape)
            sep_util.write_file(model.vecfile, model_arr, model_axis)
        else:
            model_arr = model.getNdArray()
            model_arr += np.matmul(self.M.T.conj(), data_arr.ravel()).reshape(model_arr.shape)
        return
    
    def getNdArray(self):
        return np.array(self.M)


class FirstDerivative(pyOp.Operator):
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
        self.dims = model.getNdArray().shape
        self.axis = axis if axis >= 0 else len(self.dims) + axis
        super(FirstDerivative, self).__init__(model, model)
    
    def __str__(self):
        return "1stDer_%d" % self.axis
    
    def forward(self, add, model, data):
        """Forward operator"""
        self.checkDomainRange(model, data)
        if add:
            data_tmp = data.clone()
        data.zero()
        # Getting Ndarrays
        x = model.clone().getNdArray()
        if self.axis > 0:  # need to bring the dim. to derive to first dim
            x = np.swapaxes(x, self.axis, 0)
        y = np.zeros(x.shape)
        
        y[:-1] = (x[1:] - x[:-1]) / self.sampling
        y[-1] = 0
        if self.axis > 0:  # reset axis order
            y = np.swapaxes(y, 0, self.axis)
        data.getNdArray()[:] = y
        if add:
            data.scaleAdd(data_tmp)
        return
    
    def adjoint(self, add, model, data):
        """Adjoint operator"""
        self.checkDomainRange(model, data)
        if add:
            model_temp = model.clone()
        model.zero()
        # Getting Ndarrays
        y = data.clone().getNdArray()
        if self.axis > 0:  # need to bring the dim. to derive to first dim
            y = np.swapaxes(y, self.axis, 0)
        x = np.zeros(y.shape)
        
        x[0] = -y[0] / self.sampling
        x[1:-1] = (-y[1:-1] + y[:-2]) / self.sampling
        x[-1] = y[-2] / self.sampling
        
        if self.axis > 0:
            x = np.swapaxes(x, 0, self.axis)
        model.getNdArray()[:] = x
        if add:
            model.scaleAdd(model_temp)
        return


class SecondDerivative(pyOp.Operator):
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
            x = np.swapaxes(x, self.axis, 0)
        y = np.zeros(x.shape)
        
        y[1:-1] = (x[0:-2] - 2 * x[1:-1] + x[2:]) / self.sampling ** 2
        
        if self.axis > 0:  # reset axis order
            y = np.swapaxes(y, 0, self.axis)
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
            y = np.swapaxes(y, self.axis, 0)
        x = np.zeros(y.shape)
        
        x[0:-2] += (y[1:-1]) / self.sampling ** 2
        x[1:-1] -= (2 * y[1:-1]) / self.sampling ** 2
        x[2:] += (y[1:-1]) / self.sampling ** 2
        
        if self.axis > 0:
            x = np.swapaxes(x, 0, self.axis)
        model.getNdArray()[:] = x
        if add:
            model.scaleAdd(self.data_tmp)
        return


class TotalVariation(pyOp.Operator):
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


class Laplacian(pyOp.Operator):
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


class GaussianFilter(pyOp.Operator):
    def __init__(self, model, sigma):
        """
        Gaussian smoothing operator using scipy smoothing:
        model    = [no default] - vector class; domain vector
        sigma   = [no default] - scalar or sequence of scalars; standard deviation along the model directions
        """
        self.setDomainRange(model, model)
        self.sigma = sigma
        self.scaling = np.sqrt(np.prod(np.array(self.sigma)/np.pi))  # in order to have the max amplitude 1
        return
    
    def __str__(self):
        return "GausFilt"
    
    def forward(self, add, model, data):
        """Forward operator"""
        self.checkDomainRange(model, data)
        if not add:
            data.zero()
        # Getting Ndarrays
        model_arr = model.getNdArray()
        data_arr = data.getNdArray()
        data_arr[:] += self.scaling * gaussian_filter(model_arr, sigma=self.sigma)
        return
    
    def adjoint(self, add, model, data):
        """Self-adjoint operator"""
        self.forward(add, data, model)
        return


# TODO Fix ConvNDscipy for Matching Filters applications
class ConvNDscipy(pyOp.Operator):
    """
    ND convolution square operator in the domain space

    :param domain   : [no default] - vector class; domain vector
    :param kernel   : [no default] - vector class; kernel vector
    :param method   : [auto] - str; how to compute the convolution [auto, direct, fft]
    :return         : Convolution Operator
    """
    
    def __init__(self, domain, kernel, method='auto'):
        
        if isinstance(kernel, pyVec.vector):
            self.kernel = kernel.clone().getNdArray()
        elif isinstance(kernel, np.ndarray):
            self.kernel = kernel.copy()
        else:
            raise ValueError("kernel has to be either a vector or a numpy.ndarray")

        # Padding array to avoid edge effects
        pad_width = []
        for len_filt in self.kernel.shape:
            half_len = int(len_filt / 2)
            if np.mod(len_filt,2):
                padding = (half_len, half_len)
            else:
                padding = (half_len, half_len -1)
            pad_width.append(padding)
        self.kernel = np.pad(self.kernel, pad_width)
        
        if len(domain.shape()) != len(self.kernel.shape):
            raise ValueError("Domain and kernel number of dimensions mismatch")
        
        assert method in ["auto", "direct", "fft"], "method has to be auto, direct or fft"
        self.method = method
        
        super(ConvNDscipy, self).__init__(domain, domain)
    
    def __str__(self):
        return "ConvScipy"
    
    def forward(self, add, model, data):
        self.checkDomainRange(model, data)
        if not add:
            data.zero()
        modelNd = model.getNdArray()
        dataNd = data.getNdArray()[:]
        dataNd += convolve(modelNd, self.kernel, mode='same', method=self.method)
        return
    
    def adjoint(self, add, model, data):
        self.checkDomainRange(model, data)
        if not add:
            model.zero()
        modelNd = model.getNdArray()
        dataNd = data.getNdArray()[:]
        modelNd += correlate(dataNd, self.kernel, mode='same', method=self.method)
        return


def FourierTransform(domain, dirs=None, nffts=None, sampling=1., dtype=np.complex128):
    if isinstance(domain, pyVec.vectorIC):
        return _FFT_IC(domain, dirs, nffts, sampling, dtype)
    elif isinstance(domain, pyVec.superVector):
        # TODO add the possibility to have different settings for each sub-vector
        return pyOp.Dstack([_FFT_IC(v, dirs, nffts, sampling, dtype) for v in domain.vecs])
    else:
        raise ValueError("ERROR! Provided domain has to be either vector or superVector")


class _FFT_IC(pyOp.Operator):
    
    def __init__(self, domain, dirs=None, nffts=None, sampling=1., dtype=np.complex128):
        """
        :param domain   : vector class; domain vector
        :param dirs     : scalar of sequence of scalars
                            directions along which to compute the FFT
        :param nffts    : scalar or sequence of scalars
                            number of FFT points
        :param sampling : scalar or sequence of scalars
                            sampling step for FFT computation
        :return         : FFT Operator
        """
        assert isinstance(domain, pyVec.vectorIC), "Domain has to be a vectorIC object (for now)"
        self.dims = domain.shape
        if dirs is None:
            self.dirs = list(range(len(self.dims)))
        else:
            self.dirs = [int(dirs[_]) for _ in range(len(dirs))]
            assert len(self.dirs) <= len(
                self.dims), "The number of FFT directions is greater than the domain dimensionality"
        
        if nffts is None:
            self.nffts = self.dims
        else:
            self.nffts = [int(nffts[_]) for _ in range(len(nffts))]
            assert len(self.nffts) <= len(
                self.dims), "The number of FFT points is greater than the domain dimensionality"
        
        if np.isscalar(sampling):
            self.sampling = [sampling] * len(self.nffts)
        else:
            self.sampling = list(sampling)
            assert len(self.sampling) <= len(
                self.nffts), "The number of FFT samplings is greater than the domain dimensionality"
        
        assert len(self.nffts) == len(self.sampling) == len(self.dirs), "Provided parameters dimensions mismatch"
        
        super(_FFT_IC, self).__init__(domain, pyVec.vectorIC(np.zeros(self.nffts, dtype=dtype)))
        
        # store vectors of frequency bins
        self.weight = np.sqrt(np.prod(self.nffts))
        self.sampling_frequencies = [np.fft.fftfreq(nfft, d=samp) for nfft, samp in zip(self.nffts, self.sampling)]
    
    def __str__(self):
        return "Fourier "
    
    def forward(self, add, model, data):
        """Compute FFT on model and save to data"""
        self.checkDomainRange(model, data)
        if not add:
            data.zero()
        data.arr += 1. / self.weight * np.fft.fftn(model.getNdArray(), s=self.nffts, axes=self.dirs)
        return
    
    def adjoint(self, add, model, data):
        """Compute IFFT on data and save to model"""
        self.checkDomainRange(model, data)
        if add:
            temp = model.clone()
        model_arr = self.weight * np.fft.ifftn(data.getNdArray(), s=self.nffts, axes=self.dirs)
        for direction in self.dirs:
            model_arr = np.take(model_arr, range(self.dims[direction]), axis=direction)
        
        model.arr = model_arr
        if add:
            model.scaleAdd(temp, 1., 1.)
        return


def ZeroPad(domain, pad):
    if isinstance(domain, pyVec.vectorIC):
        return _ZeroPadIC(domain, pad)
    elif isinstance(domain, pyVec.superVector):
        # TODO add the possibility to have different padding for each sub-vector
        return pyOp.Dstack([_ZeroPadIC(v, pad) for v in domain.vecs])
    else:
        raise ValueError("ERROR! Provided domain has to be either vector or superVector")


def _pad_vectorIC(vec, pad):
    if not isinstance(vec, pyVec.vectorIC):
        raise ValueError("ERROR! Provided vector must be of vectorIC type")
    assert len(vec.shape) == len(pad), "Dimensions of vector and padding mismatch!"
    
    vec_new_shape = tuple(np.asarray(vec.shape) + [sum(pad[_]) for _ in range(len(pad))])
    return pyVec.vectorIC(np.empty(vec_new_shape, dtype=vec.getNdArray().dtype))


class _ZeroPadIC(pyOp.Operator):
    
    def __init__(self, domain, pad):
        """ Zero Pad operator.
        
        To pad 2 values to each side of the first dim, and 3 values to each side of the second dim, use:
            pad=((2,2), (3,3))
        :param domain: vectorIC class
        :param pad: scalar or sequence of scalars
            Number of samples to pad in each dimension.
            If a single scalar is provided, it is assigned to every dimension.
        """
        if isinstance(domain, pyVec.vectorIC):
            self.dims = domain.shape
            pad = [(pad, pad)] * len(self.dims) if pad is np.isscalar else list(pad)
            if (np.array(pad) < 0).any():
                raise ValueError('Padding must be positive or zero')
            self.pad = pad
            super(_ZeroPadIC, self).__init__(domain, _pad_vectorIC(domain, self.pad))
    
    def __str__(self):
        return "ZeroPad "
    
    def forward(self, add, model, data):
        """Zero padding"""
        self.checkDomainRange(model, data)
        if add:
            temp = data.clone()
        y = np.pad(model.arr, self.pad, mode='constant')
        data.arr = y
        if add:
            data.scaleAdd(temp, 1., 1.)
        return
    
    def adjoint(self, add, model, data):
        """Extract non-zero subsequence"""
        self.checkDomainRange(model, data)
        if add:
            temp = model.clone()
        x = data.clone().arr
        for ax, pad in enumerate(self.pad):
            x = np.take(x, pad[0] + np.arange(self.dims[ax]), axis=ax)
        model.arr = x
        if add:
            model.scaleAdd(temp, 1., 1.)
        return


class FromScipy(pyOp.Operator):
    
    def __init__(self, op):
        """
        Cast a scipy LinearOperator to Operator
        :param op: `scipy.sparse.linalg.LinearOperator` class (or child, such as pylops.LinearOperator)
        """
        assert isinstance(op, LinearOperator), "op has to be a scipy LinearOperator"
        self.matvec = op.matvec
        self.rmatvec = op.rmatvec
        self.name = op.__str__()
        super(FromScipy, self).__init__(pyVec.vectorIC(np.empty(op.shape[1])),
                                        pyVec.vectorIC(np.empty(op.shape[0])))
    
    def __str__(self):
        return self.name.replace('<', '').replace('>', '')
        
    def forward(self, add, model, data):
        self.checkDomainRange(model, data)
        if add:
            temp = data.clone()
        data.getNdArray()[:] = self.matvec(model.getNdArray())
        if add:
            data.scaleAdd(temp, 1., 1.)
            
    def adjoint(self, add, model, data):
        self.checkDomainRange(model, data)
        if add:
            temp = model.clone()
        model.getNdArray()[:] = self.rmatvec(data.getNdArray())
        if add:
            model.scaleAdd(temp, 1., 1.)
    

class ToScipy(LinearOperator):
    
    def __init__(self, op):
        """
        Cast an Operator to scipy LinearOperator or to pylops if available
        :param op: `pyOperator.Operator` object (or child)
        """
        assert isinstance(op, pyOp.Operator), 'op has to be a pyOperator.Operator'
        super(ToScipy, self).__init__(shape=(op.range.size, op.domain.size),
                                      dtype=op.domain.getNdArray().dtype)
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

import matplotlib.pyplot as plt
if __name__ == '__main__':
    x = pyVec.vectorIC(np.load('../testdata/monarch.npy'))
    kernel = np.array([[0,1,0], [1,-4,1], [0,1,0]])
    nh = [5, 10]
    hz = np.exp(-0.1 * np.linspace(-(nh[0] // 2), nh[0] // 2, nh[0]) ** 2)
    hx = np.exp(-0.03 * np.linspace(-(nh[1] // 2), nh[1] // 2, nh[1]) ** 2)
    hz /= np.trapz(hz)  # normalize the integral to 1
    hx /= np.trapz(hx)  # normalize the integral to 1
    kernel = hz[:, np.newaxis] * hx[np.newaxis, :]

    C = ConvNDscipy(x, kernel)

    C.dotTest(True)
    # x = pyVec.vectorIC(np.arange(9).reshape((3, 3)))
    # pad = ((2,2), (3,3))
    # P = ZeroPad(x, pad)
    # P.dotTest()
    # xx = pyVec.superVector(x, x)
    # PP = ZeroPad(xx, pad)
    # PP.dotTest()
    
    # np.random.seed(1)
    # y = pyVec.vectorIC(np.random.rand(301, 601))
    # F = FourierTransform(y, nffts=[512, 1024])
    # yfft = F * y
    # F.dotTest(True)
    #
    # yy = pyVec.superVector(y, y)
    # FF = FourierTransform(yy, nffts=[512, 1024])
    # yyfft = FF * yy
    # FF.dotTest(True)
    #
    # print(0)
