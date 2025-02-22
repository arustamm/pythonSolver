import pyVector
import numpy as np
import pyarrow.dataset as ds
import pyarrow.compute as pc
import pyarrow as pa

import dask.dataframe as dd
import dask.bag as db
import pandas as pd
import random
import string
import os
import dask
import shutil
import hashlib

# Vector class using Parquet files 
# Operations are performed out-of-core in a streaming fashion
class ParquetVector(pyVector.vector):

    def __init__(self, path=None, df=None, data_key="data", filter=None):
        if df is not None and isinstance(df, dd.DataFrame):
            self.df = df.copy()
        elif path is not None:
            self.df = dd.read_parquet(path, 
                        dtype_backend="pyarrow",
                        split_row_groups=True,
                        ignore_metadata_file=True,
                        parquet_file_extension=None,
                        arrow_to_pandas={
                            "split_blocks" : True,
                            "self_destruct" : True,
                            "ignore_metadata" : True,
                        }
                    )
        else:
            raise RuntimeError("Must provide either path to parquet dataset or DaskDataframe!")
        self.path = path
        self.key = data_key
        self.filter = filter
        

    # def __del__(self):
    #     """Default destructor"""

    def hash(self):
        """
            Return the combined hash across partitions
        """        
        # Compute hashes for each partition
        hashes = self.df[self.key].map_partitions(
            lambda p: hashlib.sha1(pd.util.hash_pandas_object(p).values).hexdigest(),
            meta=pd.Series()
            ).compute()
        
        # Combine hashes of all partitions
        combined_hash = hashlib.sha256(''.join(hashes).encode()).hexdigest()
        
        return combined_hash

    def isDifferent(self, vec2):
        pass

    def __add__(self, other):  # self + other
        # self.checkSame(other)
        res = self.clone()
        def _add(series1, series2):
            arr1 = series_to_pyarrow(series1)
            arr2 = series_to_pyarrow(series2)
            arr3 = pc.add(arr1, arr2)
            list_arr = to_pyarrow_list(arr3, len(series1))
            return pd.Series(list_arr, name=series1.name, dtype=pd.ArrowDtype(pa.list_(pa.float32())))
        
        res.df[self.key] = self.df[self.key].map_partitions(
            _add, other.df[self.key],
            meta=pd.Series([], dtype=pd.ArrowDtype(pa.list_(pa.float32())))
        )

        return res

    def __iadd__(self, other):  # self + other
        # self.checkSame(other)
        def _add(series1, series2):
            arr1 = series_to_pyarrow(series1)
            arr2 = series_to_pyarrow(series2)
            arr3 = pc.add(arr1, arr2)
            list_arr = to_pyarrow_list(arr3, len(series1))
            return pd.Series(list_arr, name=series1.name, dtype=pd.ArrowDtype(pa.list_(pa.float32())))
        
        self.df[self.key] = self.df[self.key].map_partitions(
            _add, other.df[self.key],
            meta=pd.Series([], dtype=pd.ArrowDtype(pa.list_(pa.float32())))
        )

        return self

    # def __sub__(self, other):  # self - other
    #     self.checkSame(other)
    #     res = self.__add__(-other)
    #     return res

    # def __neg__(self):  # -self
    #     self.scale(-1)
    #     return self

    # def __mul__(self, other):  # self * other
    #     self.checkSame(other)
    #     if type(other) in [int, float]:
    #         self.scale(other)
    #         return self
    #     elif isinstance(other, vector):
    #         self.multiply(other)
    #         return self
    #     else:
    #         raise NotImplementedError

    # def __rmul__(self, other):
    #     self.checkSame(other)
    #     if type(other) in [int, float]:
    #         self.scale(other)
    #         return self
    #     elif isinstance(other, vector):
    #         self.multiply(other)
    #         return self
    #     else:
    #         raise NotImplementedError

    # def __pow__(self, power, modulo=None):
    #     if type(power) in [int, float]:
    #         self.pow(power)
    #     else:
    #         raise TypeError('power has to be a scalar')

    # def __abs__(self):
    #     self.abs()

    # def __truediv__(self, other):  # self / other
    #     if type(other) in [int, float]:
    #         self.scale(1 / other)
    #     elif isinstance(other, vector):
    #         self.multiply(other.clone().reciprocal())
    #     else:
    #         raise TypeError('other has to be either a scalar or a vector')

    # # these were needed for dask
    # def __getitem__(self, it):
    #     arr = self.getNdArray()
    #     return arr[it]

    # def __setitem__(self, it, val):
    #     arr = self.getNdArray()
    #     arr[it] = val

    # Class vector operations
    def getNdArray(self):
        """Function to return Ndarray of the vector"""
        return self.df[self.key]

    @property
    def shape(self):
        """Property to get the vector shape (number of samples for each axis)"""
        shape = (self.df.shape[0].compute(), )
        return shape
    
    @property
    def size(self):
        """Property to compute the vector size (number of samples)"""
        return self.df.shape[0].compute()
    
    @property
    def ndim(self):
        return 2

    def norm(self, N=2):
        """Function to compute vector N-norm"""
        def _norm(series, N):
            arr = series_to_pyarrow(series)
            val = pc.sum(pc.power(arr, N))
            return pd.Series(val, dtype=pd.ArrowDtype(pa.float64()))
        
        partial_sums = self.df[self.key].map_partitions(
            _norm, N, 
            meta=pd.Series([], dtype=pd.ArrowDtype(pa.float64()))
        ).compute()
        return np.power(np.sum(partial_sums), 1./N)


    def zero(self):
        """Function to zero out a vector"""
        self.scale(0.)
        return self

    def max(self):
        """Function to obtain maximum value within a vector"""
        def _max(series):
            arr = series_to_pyarrow(series)
            val = pc.max(arr)
            return pd.Series(val, dtype=pd.ArrowDtype(pa.float32()))
        
        partial_maxs = self.df[self.key].map_partitions(
            _max, 
            meta=pd.Series([], dtype=pd.ArrowDtype(pa.float32()))
        ).compute()
        return np.amax(partial_maxs)

    def min(self):
        """Function to obtain minimum value within a vector"""
        def _min(series):
            arr = series_to_pyarrow(series)
            val = pc.min(arr)
            return pd.Series(val, dtype=pd.ArrowDtype(pa.float32()))
        
        partial_mins = self.df[self.key].map_partitions(
            _min, 
            meta=pd.Series([], dtype=pd.ArrowDtype(pa.float32()))
        ).compute()
        return np.amin(partial_mins)

    def set(self, val):
        """Function to set all values in the vector"""
        def _set(series, val):
            arr = series_to_pyarrow(series)
            arr = pa.array(val * np.ones(len(arr), dtype=np.float32()))
            list_array = to_pyarrow_list(arr, len(series))
            return pd.Series(list_array, name=series.name, dtype=pd.ArrowDtype(pa.list_(pa.float32())))
        
        self.df[self.key] = self.df[self.key].map_partitions(
            _set, val,
            meta=pd.Series([], dtype=pd.ArrowDtype(pa.list_(pa.float32())))
        )

        return self

    def scale(self, sc):
        """Function to scale a vector"""
        def _scale(series):
            arr = series_to_pyarrow(series)
            arr = pc.multiply(arr, np.float32(sc))
            list_array = to_pyarrow_list(arr, len(series))
            return pd.Series(list_array, name=series.name, dtype=pd.ArrowDtype(pa.list_(pa.float32())))
        
        self.df[self.key] = self.df[self.key].map_partitions(
            _scale,
            meta=pd.Series([], dtype=pd.ArrowDtype(pa.list_(pa.float32())))
        )

        return self

    def addbias(self, bias):
        """Function to add bias to a vector"""
        def _addbias(series, bias):
            arr = series_to_pyarrow(series)
            arr = pc.add(arr, bias)
            list_array = to_pyarrow_list(arr, len(series))
            return pd.Series(list_array, name=series.name, dtype=pd.ArrowDtype(pa.list_(pa.float32())))
        
        self.df[self.key] = self.df[self.key].map_partitions(
            _addbias, bias,
            meta=pd.Series([], dtype=pd.ArrowDtype(pa.list_(pa.float32())))
        )

        return self

    def rand(self):
        """Function to randomize a vector"""
        def _rand(series):
            arr = series_to_pyarrow(series)
            arr = pa.array(np.random.rand(len(arr)), type=pa.float32())
            list_array = to_pyarrow_list(arr, len(series))
            return pd.Series(list_array, name=series.name, dtype=pd.ArrowDtype(pa.list_(pa.float32())))
        
        self.df[self.key] = self.df[self.key].map_partitions(
            _rand,
            meta=pd.Series([], dtype=pd.ArrowDtype(pa.list_(pa.float32())))
        )

        return self

    # def clone(self):
    #     """Function to clone (deep copy) a vector from a vector or a Space"""
    #     random_word = ''.join(random.choices(string.ascii_letters, k=5))
    #     path = self.clone_path + "/" + random_word

    #     if os.path.exists(path):
    #         raise FileExistsError(f"The destination path '{path}' already exists.")
    #     else:
    #         os.makedirs(path, exist_ok=False) 

    #         # Create a bag of file paths
    #     file_paths = db.from_sequence(os.listdir(self.path))
    #     # Apply the copy function to each file
    #     file_paths.map(lambda file: shutil.copy2(os.path.join(self.path, file), path)).compute()

    #     return ParquetVector(path, self.clone_path, data_key=self.key, filter=None)

    def clone(self):
        """Function to clone a vector from a vector or a Space"""
        return ParquetVector(df=self.df, data_key=self.key, filter=self.filter)
        

    # def cloneSpace(self):
    #     """Function to clone vector space"""
    #     raise NotImplementedError("cloneSpace must be overwritten")

    # def checkSame(vec):
    #     """Function to check to make sure the vectors exist in the same space"""
    #     num_cols = vec.num_cols
    #     num_rows = vec.num_rows

    def window(self):
        """ A function to create a chunk of a Vector
            This is needed for creating DaskVector from existing Vector
        """
        raise NotImplementedError("Need to overwrite windowing function!")

    def writeVec(self, path, mode='w'):
        """Function to write vector to file"""
        raise NotImplementedError("writeVec must be overwritten")


    # # TODO implement on seplib
    # def abs(self):
    #     """Return a vector containing the absolute values"""
    #     raise NotImplementedError('abs method must be implemented')

    # # TODO implement on seplib
    # def sign(self):
    #     """Return a vector containing the signs"""
    #     raise NotImplementedError('sign method have to be implemented')

    # # TODO implement on seplib
    # def reciprocal(self):
    #     """Return a vector containing the reciprocals of self"""
    #     raise NotImplementedError('reciprocal method must be implemented')

    # # TODO implement on seplib
    # def maximum(self, vec2):
    #     """Return a new vector of element-wise maximum of self and vec2"""
    #     raise NotImplementedError('maximum method must be implemented')

    # # TODO implement on seplib
    # def conj(self):
    #     """Compute conjugate transpose of the vector"""
    #     raise NotImplementedError('conj method must be implemented')

    # # TODO implement on seplib
    # def pow(self, power):
    #     """Compute element-wise power of the vector"""
    #     raise NotImplementedError('pow method must be implemented')

    # # TODO implement on seplib
    # def real(self):
    #     """Return the real part of the vector"""
    #     raise NotImplementedError('real method must be implemented')

    # # TODO implement on seplib
    # def imag(self):
    #     """Return the imaginary part of the vector"""
    #     raise NotImplementedError('imag method must be implemented')

    # # Combination of different vectors

    def copy(self, vec2):
        """Function to copy vector"""
        self = vec2.clone()
        return self

    def scaleAdd(self, vec2, sc1=1.0, sc2=1.0):
        """Function to scale two vectors and add them to the first one"""
        def _scale_add(series1, series2, sc1, sc2):
            arr1 = series_to_pyarrow(series1)
            arr2 = series_to_pyarrow(series2)
            arr1 = pc.multiply_checked(arr1.values, np.float32(sc1))
            arr2 = pc.multiply_checked(arr2.values, np.float32(sc2))
            arr3 = pc.add(arr1, arr2)
            list_array = to_pyarrow_list(arr3, len(series1))
            return pd.Series(list_array, name=series1.name, dtype=pd.ArrowDtype(pa.list_(pa.float32())))
        
        self.df[self.key] = self.df[self.key].map_partitions(
            _scale_add, vec2.df[self.key], sc1=sc1, sc2=sc2,
            meta=pd.Series([], dtype=pd.ArrowDtype(pa.list_(pa.float32())))
        )
        

    def dot(self, vec2):
        """Function to compute dot product between two vectors"""
        def _dot(series1, series2):
            arr1 = series_to_pyarrow(series1)
            arr2 = series_to_pyarrow(series2)
            val = pc.sum(pc.multiply(arr1.values, arr2.values))
            return pd.Series(val, dtype=pd.ArrowDtype(pa.float64()))
        
        partial_dots = self.df[self.key].map_partitions(
            _dot, vec2.df[self.key], 
            meta=pd.Series([], dtype=pd.ArrowDtype(pa.float64()))
        ).compute()
        return np.sum(partial_dots)

    # def multiply(self, vec2):
    #     """Function to multiply element-wise two vectors"""
    #     raise NotImplementedError("multiply must be overwritten")

    # def isDifferent(self, vec2):
    #     """Function to check if two vectors are identical"""

    #     raise NotImplementedError("isDifferent must be overwritten")

    # def clipVector(self, low, high):
    #     """
    #        Function to bound vector values based on input vectors min and max
    #     """
    #     raise NotImplementedError("clipVector must be overwritten")


# Helper functions for pyarrow conversion
def series_to_pyarrow(series):
    arr = pa.array(series)
    if isinstance(arr, pa.lib.ChunkedArray):
        arr = arr.combine_chunks()
    return arr.values

def to_pyarrow_list(arr, list_len):
    ch = len(arr) // list_len
    offsets = np.arange(0, len(arr) + ch, ch, dtype=int)
    return pa.ListArray.from_arrays(offsets, arr)