from __future__ import division, print_function, absolute_import
from math import ceil, log
import numpy as np
from pyNpOperator import ZeroPad
from pyOperator import Operator
from pyVector import *

try:
    import pywt
except ModuleNotFoundError:
    import subprocess
    import sys
    subprocess.call([sys.executable, "-m", "pip", "install", "PyWavelets"])
    import pywt


class DWT2D(Operator):
    """Two dimensional Wavelet operator.
    Apply 2D-Wavelet Transform along two directions ``dirs`` of a
    multi-dimensional array of size ``dims``.
    Note that the Wavelet operator is an overload of the ``pywt``
    implementation of the wavelet transform. Refer to
    https://pywavelets.readthedocs.io for a detailed description of the
    input parameters.
    Parameters
    ----------
    dims : :obj:`tuple`
        Number of samples for each dimension
    dirs : :obj:`tuple`, optional
        Direction along which FFT is applied.
    wavelet : :obj:`str`, optional
        Name of wavelet type. Use :func:`pywt.wavelist(kind='discrete')` for
        a list of
        available wavelets.
    level : :obj:`int`, optional
        Number of scaling levels (must be >=0).
    dtype : :obj:`str`, optional
        Type of elements in input array.
    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly
        (True) or not (False)
    Raises
    ------
    ModuleNotFoundError
        If ``pywt`` is not installed
    ValueError
        If ``wavelet`` does not belong to ``pywt.families``
    Notes
    -----
    The Wavelet operator applies the 2-dimensional multilevel Discrete
    Wavelet Transform (DWT2) in forward mode and the 2-dimensional multilevel
    Inverse Discrete Wavelet Transform (IDWT2) in adjoint mode.
    """
    def __init__(self, dims, dirs=(0, 1), wavelet='haar',
                 level=1, dtype='float64'):
        if pywt is None:
            raise ModuleNotFoundError('The wavelet operator requires '
                                      'the pywt package t be installed. '
                                      'Run "pip install PyWavelets" or '
                                      '"conda install pywavelets".')
        _checkwavelet(wavelet)

        # define padding for length to be power of 2
        ndimpow2 = [max(2 ** ceil(log(dims[dir], 2)), 2 ** level)
                    for dir in dirs]
        pad = [(0, 0)] * len(dims)
        for i, d in enumerate(dirs):
            pad[d] = (0, ndimpow2[i] - dims[d])
        self.pad = ZeroPad(dims, pad)
        self.dims = dims
        self.dirs = dirs
        self.dimsd = list(dims)
        for i, d in enumerate(dirs):
            self.dimsd[d] = ndimpow2[i]

        # apply transform once again to find out slices
        _, self.sl = \
            pywt.coeffs_to_array(pywt.wavedec2(np.ones(self.dimsd),
                                               wavelet=wavelet,
                                               level=level,
                                               mode='periodization',
                                               axes=self.dirs),
                                 axes=self.dirs)
        self.wavelet = wavelet
        self.waveletadj = _adjointwavelet(wavelet)
        self.level = level
        self.shape = (int(np.prod(self.dimsd)), int(np.prod(self.dims)))
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        x = self.pad.matvec(x)
        x = np.reshape(x, self.dimsd)
        y = pywt.coeffs_to_array(pywt.wavedec2(x, wavelet=self.wavelet,
                                               level=self.level,
                                               mode='periodization',
                                               axes=self.dirs),
                                 axes=(self.dirs))[0]
        return y.ravel()

    def _rmatvec(self, x):
        x = np.reshape(x, self.dimsd)
        x = pywt.array_to_coeffs(x, self.sl, output_format='wavedec2')
        y = pywt.waverec2(x, wavelet=self.waveletadj, mode='periodization',
                          axes=self.dirs)
        y = self.pad.rmatvec(y.ravel())
        return y


def _checkwavelet(wavelet):
    """Check that wavelet belongs to pywt.wavelist
    """
    wavelist = pywt.wavelist(kind='discrete')
    if wavelet not in wavelist:
        raise ValueError("'%s' not in family set = %s" % (wavelet, wavelist))


def _adjointwavelet(wavelet):
    """Define adjoint wavelet
    """
    waveletadj = wavelet
    if 'rbio' in wavelet:
        waveletadj = 'bior' + wavelet[-3:]
    elif 'bior' in wavelet:
        waveletadj = 'rbio' + wavelet[-3:]
    return waveletadj


if __name__ == '__main__':
    
    x = vectorIC(np.random.rand(301, 601))
    W = DWT2D(x, dirs=(0, 1), wavelet='haar', level=1)
    
    print(0)
