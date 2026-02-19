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
import uuid
import atexit

import pysep3d 
from typing import Dict, Any

# Vector class using Parquet files 
# Operations are performed out-of-core in a streaming fashion
class ParquetVector(pyVector.vector):

    # Track all instances to manage cleanup safely
    _temp_instances = set()

    def __init__(self, 
                 vector_reader: pysep3d.PyArrowReader,
                 data_key: str = 'data',
                 temp_dir: str = '/tmp',
                 dtype=np.float32):
        """
        Args:
            vector_reader: An instance of pysep3d.PyArrowReader configured for the source.
            data_key: Column name for the vector data.
            temp_dir: Location for temporary clone files.
        """
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # 1. Read Data using the provided Reader Step
        # This applies any reader-specific logic (like to_object conversion)
        self.df = vector_reader.create()
        self.dtype = dtype
        if dtype == np.complex64 or dtype == np.complex128:
            # Convert float columns to complex
            self.df = pysep3d.FloatToComplex().apply(self.df)
        self.meta = self.df._meta

        # Track the path for cleanup (if it came from a reader with a path)
        self.path = vector_reader.path
        
        # Default cleanup behavior:
        # If the reader points to a file in our temp dir, mark for deletion.
        # Otherwise (e.g., input data), keep it.
        self.remove_file = False
        if self.path and os.path.abspath(self.path).startswith(os.path.abspath(self.temp_dir)):
            self.remove_file = True

        self.key = data_key
        self.save_cnt = 0 # Counter for mode='a' writes

        sample_trace = self.df.head(1)[self.key].iloc[0]
        self.n_samples = len(sample_trace)
            
        # Register for potential cleanup
        ParquetVector._temp_instances.add(self)

    # def __del__(self):
    #     """Default destructor"""

    def hash(self):

        def _compute_partition_hash(df):
            """Compute hash for a single partition"""
            series = df[self.key]
            
            # Stack all arrays in the partition into one matrix
            matrix = np.stack(np.asarray(series.values))
            flat_data = matrix.flatten()
            # Convert to bytes and hash
            data_bytes = flat_data.tobytes()
            hasher = hashlib.sha256(data_bytes)
            
            return hasher.hexdigest()
        
        # Compute hash for each partition
        partition_hashes = self.df.map_partitions(
            _compute_partition_hash,
            meta=pd.Series([], dtype=str)
        ).compute()
        
        # Combine all partition hashes into a single hash
        combined = ''.join(partition_hashes.values)
        
        final_hash = hashlib.sha256(combined.encode()).hexdigest()
        
        return final_hash

    def isDifferent(self, vec2):
        return self.hash() != vec2.hash()

    def __add__(self, other):  # self + other
        # self.checkSame(other)
        res = self.clone()
        res.scaleAdd(self, sc1=1.0, sc2=1.0)

        return res

    def __iadd__(self, other):  # self + other
        # self.checkSame(other)        
        self.scaleAdd(other, sc1=1.0, sc2=1.0)
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

    def __getitem__(self, it):
        arr = self.getNdArray()
        return arr[it]

    def __setitem__(self, it, val):
        raise NotImplementedError("Setting individual items is not supported in ParquetVector")

    # Class vector operations
    def getNdArray(self):
        """Function to return Ndarray of the vector"""
        return self.df[self.key].compute().to_numpy()

    @property
    def shape(self):
        """Property to get the vector shape (number of samples for each axis)"""
        shape = (self.df.shape[0].compute(), self.n_samples)
        return shape
    
    @property
    def size(self):
        """Property to compute the vector size (number of samples)"""
        return self.shape[0] * self.shape[1]
    
    @property
    def ndim(self):
        return 2

    def zero(self):
        """Function to zero out a vector"""
        self.scale(0.)
        return self

    def max(self):
        """Function to obtain maximum value within a vector"""
        return self.df[self.key].max().compute()

    def min(self):
        """Function to obtain minimum value within a vector"""
        return self.df[self.key].min().compute()

    def rand(self):
        def _rand(df, ns):
            n_traces = len(df)
            # Generate one big block of random numbers
            matrix = np.random.rand(n_traces, ns).astype(self.dtype)
            df[self.key] = list(matrix)
            return df
        
        self.df = self.df.map_partitions(
            _rand, self.n_samples,
            meta=self.meta
        )
        return self

    def set(self, val):
        def _set(df, ns):
            n_traces = len(df)
            matrix = np.full((n_traces, ns), val, dtype=self.dtype)
            df[self.key] = list(matrix)
            return df

        self.df = self.df.map_partitions(
            _set, self.n_samples,
            meta=self.meta
        )
        return self

    def scale(self, sc):
        """Scale using matrix multiplication"""
        def _scale(df):
            matrix = np.stack(np.asarray(df[self.key].values))
            matrix *= sc
            df[self.key] = list(matrix)
            return df

        self.df = self.df.map_partitions(
            _scale,
            meta=self.meta
        )
        return self

    def scaleAdd(self, vec2, sc1=1.0, sc2=1.0):
        """Vectorized ScaleAdd"""
        def _scale_add(df1, df2):
            m1 = np.stack(np.asarray(df1[self.key].values))
            m2 = np.stack(np.asarray(df2[self.key].values))
            res = m1 * sc1 + m2 * sc2
            df1[self.key] = list(res)
            return df1
        
        self.df = self.df.map_partitions(
            _scale_add, 
            vec2.df,
            meta=self.meta,
        )
        return self

    def addbias(self, bias):
        def _bias(df):
            matrix = np.stack(np.asarray(df[self.key].values))
            matrix += bias
            df[self.key] = list(matrix)
            return df

        self.df = self.df.map_partitions(
            _bias,
            meta=self.meta
        )
        return self

    def clone(self):
        """
        Creates a physical copy (checkpoint) using the library's Writer/Reader.
        """
        # 1. Generate Temp Path
        unique_id = str(uuid.uuid4())
        new_path = os.path.join(self.temp_dir, f"vec_{unique_id}.parquet")
        
        # 2. Write to disk (Checkpointing)
        # We use pysep3d.PyArrowWriter to ensure the temp file is valid Parquet
        writer = pysep3d.PyArrowWriter(path=new_path)
        if self.dtype == np.complex64 or self.dtype == np.complex128:
            # Convert complex columns to float for storage
            self.df = pysep3d.ComplexToFloat().apply(self.df)
        writer.write(self.df)
        
        # 3. Create a new Reader for the new file
        # This breaks the Dask graph lineage
        new_reader = pysep3d.PyArrowReader(path=new_path)
        
        # 4. Return new Vector
        new_vec = ParquetVector(
            vector_reader=new_reader,
            data_key=self.key,
            temp_dir=self.temp_dir
        )
        
        # Explicitly mark as temporary so it gets deleted on exit
        new_vec.remove_file = True
        return new_vec
        

    # def cloneSpace(self):
    #     """Function to clone vector space"""
    #     raise NotImplementedError("cloneSpace must be overwritten")

    def checkSame(self, vec):
        """Function to check to make sure the vectors exist in the same space"""
        return isinstance(vec, ParquetVector) and self.shape == vec.shape

    def window(self, params: Dict[str, Any]):
        """ A function to create a chunk of a Vector
            This is needed for creating DaskVector from existing Vector
        """
        df = pysep3d.Window(params).apply(self.df)
        self.df = df.copy()
        return self

    def writeVec(self, filename, mode='w'):
        """
        Writes the vector to disk using pysep3d.PyArrowWriter.
        """
        out_path = filename

        if mode == 'a':
            os.makedirs(filename, exist_ok=True)
            sub_name = f"iter_{str(self.save_cnt).zfill(5)}.parquet"
            out_path = os.path.join(filename, sub_name)
            self.save_cnt += 1
        
        # This ensures schema consistency and handles the 'object' -> 'list' conversion
        writer = pysep3d.PyArrowWriter(path=out_path)
        if self.dtype == np.complex64 or self.dtype == np.complex128:
            # Convert complex columns to float for storage
            self.df = pysep3d.ComplexToFloat().apply(self.df)
        writer.write(self.df)

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
        """Function to copy the content of a vector"""
        self.df = vec2.df.copy()
        return self

    def dot(self, vec2):
        def _dot(df1, df2):
            m1 = np.stack(np.asarray(df1[self.key].values)).flatten()
            m2 = np.stack(np.asarray(df2[self.key].values)).flatten()
            return np.vdot(m1, m2).astype(np.float64)

        partial_dots = self.df.map_partitions(
            _dot, 
            vec2.df, 
            meta=pd.Series([], dtype=np.float64)
        ).compute()
        
        return np.sum(partial_dots)

    def norm(self, N=2):
        def _norm(df):
            matrix = np.stack(np.asarray(df[self.key].values)).flatten()
            return np.sum(np.power(np.abs(matrix), N)).astype(np.float64)

        partial_sums = self.df.map_partitions(
            _norm,
            meta=pd.Series([], dtype=np.float64)
        ).compute()
        
        return np.power(np.sum(partial_sums), 1.0/N)

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

    @classmethod
    def cleanup(cls):
        for instance in list(cls._temp_instances):
            if getattr(instance, 'remove_file', False) and instance.path:
                if os.path.exists(instance.path):
                    try:
                        shutil.rmtree(instance.path)
                    except OSError:
                        pass
        cls._temp_instances.clear()

atexit.register(ParquetVector.cleanup)

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