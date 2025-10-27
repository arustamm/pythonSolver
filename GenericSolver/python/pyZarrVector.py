import numpy as np
import dask.array as da
import dask.dataframe as dd
import zarr
from pathlib import Path
from typing import Optional, Union
import hashlib
import pyVector
import pyarrow as pa
import pyarrow.parquet as pq

class ZarrVector(pyVector.vector):
    """
    Vector class that separates headers (Dask DataFrame) and data (Dask Array).
    
    This avoids PyArrow list type issues and uses appropriate data structures:
    - Headers: Dask DataFrame (tabular metadata)
    - Data: Dask Array (numerical traces)
    
    Attributes:
        headers: Dask DataFrame with metadata (trace_id, sx, sy, etc.)
        data: Dask Array with trace amplitudes (n_traces, n_samples)
    """
    
    def __init__(self, 
                 headers_path: Optional[Union[str, Path]] = None,
                 traces_path: Optional[Union[str, Path]] = None,
                 headers: Optional[dd.DataFrame] = None,
                 data: Optional[da.Array] = None):
        """
        Initialize from either paths or existing Dask objects.
        
        Args:
            headers_path: Path to Parquet file with headers
            traces_path: Path to Zarr array with traces
            headers: Existing Dask DataFrame
            data: Existing Dask Array
        """
        if headers is not None and data is not None:
            # Initialize from existing Dask objects
            self.headers = headers
            self.data = data
        elif headers_path is not None and traces_path is not None:
            # Load from disk
            self.headers = dd.read_parquet(headers_path, 
                        dtype_backend="pyarrow",
                        split_row_groups=True,
                        ignore_metadata_file=True,
                        parquet_file_extension=None,
                        arrow_to_pandas={
                            "split_blocks" : True,
                            "self_destruct" : True,
                            "ignore_metadata" : True,
                        },
                    )
            self.data = da.from_zarr(traces_path)
        else:
            raise ValueError("Must provide either (headers_path, traces_path) or (headers, data)")
        
        # Validate shapes match
        if self.headers.shape[0].compute() != self.data.shape[0]:
            raise ValueError(
                f"Shape mismatch: headers has {self.headers.shape[0].compute()} rows "
                f"but data has {self.data.shape[0]} rows"
            )
    
    # ============================================
    # Properties
    # ============================================
    
    @property
    def shape(self):
        """Shape of the data array (n_traces, n_samples)"""
        return self.data.shape
    
    @property
    def size(self):
        """Total number of elements in data"""
        return self.data.size
    
    @property
    def ndim(self):
        """Number of dimensions in data"""
        return self.data.ndim
    
    @property
    def n_traces(self):
        """Number of traces"""
        return self.data.shape[0]
    
    @property
    def n_samples(self):
        """Number of samples per trace"""
        return self.data.shape[1]
    
    # ============================================
    # Array Operations (delegated to Dask Array)
    # ============================================
    
    def __add__(self, other):
        """self + other"""
        if isinstance(other, ZarrVector):
            return ZarrVector(
                headers=self.headers.copy(),
                data=self.data + other.data
            )
        else:
            # Scalar addition
            return ZarrVector(
                headers=self.headers.copy(),
                data=self.data + other
            )
    
    def __iadd__(self, other):
        """self += other"""
        if isinstance(other, ZarrVector):
            self.data = self.data + other.data
        else:
            self.data = self.data + other
        return self
    
    def __sub__(self, other):
        """self - other"""
        if isinstance(other, ZarrVector):
            return ZarrVector(
                headers=self.headers.copy(),
                data=self.data - other.data
            )
        else:
            return ZarrVector(
                headers=self.headers.copy(),
                data=self.data - other
            )
    
    def __mul__(self, other):
        """self * other"""
        if isinstance(other, ZarrVector):
            return ZarrVector(
                headers=self.headers.copy(),
                data=self.data * other.data
            )
        else:
            return ZarrVector(
                headers=self.headers.copy(),
                data=self.data * other
            )
    
    def __truediv__(self, other):
        """self / other"""
        if isinstance(other, ZarrVector):
            return ZarrVector(
                headers=self.headers.copy(),
                data=self.data / other.data
            )
        else:
            return ZarrVector(
                headers=self.headers.copy(),
                data=self.data / other
            )
    
    def __neg__(self):
        """- self"""
        return ZarrVector(
            headers=self.headers.copy(),
            data=-self.data
        )
    
    def __getitem__(self, key):
        """Support indexing"""
        # This is complex - need to slice both headers and data consistently
        if isinstance(key, (int, slice)):
            return ZarrVector(
                headers=self.headers.iloc[key],
                data=self.data[key]
            )
        else:
            raise NotImplementedError("Advanced indexing not yet supported")
    
    # ============================================
    # Vector Operations
    # ============================================
    
    def norm(self, ord=2):
        """Compute vector norm"""
        return da.linalg.norm(self.data.flatten(), ord=ord).compute()
    
    def dot(self, other):
        """Dot product with another vector"""
        if not isinstance(other, ZarrVector):
            raise TypeError("Can only dot with another ZarrVector")
        
        # Flatten and compute dot product
        return da.dot(self.data.flatten(), other.data.flatten()).compute()
    
    def max(self):
        """Maximum value in data"""
        return self.data.max().compute()
    
    def min(self):
        """Minimum value in data"""
        return self.data.min().compute()
    
    def mean(self):
        """Mean value in data"""
        return self.data.mean().compute()
    
    def std(self):
        """Standard deviation of data"""
        return self.data.std().compute()
    
    # ============================================
    # In-place Modifications
    # ============================================
    
    def zero(self):
        """Zero out the data"""
        self.data = da.zeros_like(self.data)
        return self
    
    def set(self, value):
        """Set all values to a constant"""
        self.data = da.full_like(self.data, value, dtype=self.data.dtype)
        return self
    
    def scale(self, factor):
        """Scale data by a factor"""
        self.data = self.data * factor
        return self
    
    def rand(self, seed=None):
        """Fill with random values"""
        if seed is not None:
            da.random.seed(seed)
        self.data = da.random.random(self.data.shape, chunks=self.data.chunks)
        return self
    
    def abs(self):
        """Absolute value"""
        self.data = da.abs(self.data)
        return self
    
    def clip(self, min_val, max_val):
        """Clip values to range [min_val, max_val]"""
        self.data = da.clip(self.data, min_val, max_val)
        return self
    
    # ============================================
    # Advanced Operations
    # ============================================
    
    def scaleAdd(self, other, sc1=1.0, sc2=1.0):
        """self = sc1 * self + sc2 * other"""
        if not isinstance(other, ZarrVector):
            raise TypeError("other must be ZarrVector")
        
        self.data = sc1 * self.data + sc2 * other.data
        return self
    
    def multiply(self, other):
        """Element-wise multiplication"""
        if isinstance(other, ZarrVector):
            self.data = self.data * other.data
        else:
            self.data = self.data * other
        return self
    
    # ============================================
    # Cloning and Copying
    # ============================================
    
    def clone(self):
        """Create a deep copy"""
        return ZarrVector(
            headers=self.headers.copy(),
            data=self.data.copy()
        )
    
    def copy(self, other):
        """Copy data from another vector"""
        if not isinstance(other, ZarrVector):
            raise TypeError("Can only copy from another ZarrVector")
        
        self.headers = other.headers.copy()
        self.data = other.data.copy()
        return self
    
    # ============================================
    # I/O Operations
    # ============================================
    
    def writeVec(self, output_path: Union[str, Path], overwrite: bool = True):
        """
        Write ZarrVector to disk using efficient partition-by-partition writing.
        
        Args:
            output_path: Base directory path
            overwrite: Whether to overwrite existing data
        """
        import shutil
        import gc
        
        output_path = Path(output_path)
        
        # Handle existing data
        if output_path.exists():
            if overwrite:
                print(f"Removing existing output: {output_path}")
                shutil.rmtree(output_path)
            else:
                raise FileExistsError(f"Output path {output_path} exists. Set overwrite=True to replace.")
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # ============================================
        # Write Headers Partition-by-Partition
        # ============================================
        headers_path = output_path / 'headers.parquet'
        headers_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare metadata for first partition
        metadata = {
            b'traces_path': str((output_path / 'data.zarr').absolute()).encode('utf-8'),
            b'n_samples': str(self.n_samples).encode('utf-8'),
            b'n_traces': str(self.n_traces).encode('utf-8'),
        }
        
        def write_header_partition(partition_df, partition_info=None):
            """Write a single header partition to parquet file."""
            partition_idx = partition_info['number'] if partition_info else 0
            filename = headers_path / f"part-{partition_idx:05d}.parquet"
            
            try:
                # Convert to PyArrow table
                table = pa.Table.from_pandas(partition_df, preserve_index=False)
                
                # Add metadata to first partition only
                if partition_idx == 0:
                    schema = table.schema.with_metadata(metadata)
                    table = pa.Table.from_pandas(partition_df, schema=schema, preserve_index=False)
                
                # Write to parquet file
                pq.write_table(table, filename, compression='snappy')
                
                if (partition_idx + 1) % 10 == 0 or partition_idx == 0:
                    print(f"  ✓ Wrote partition {partition_idx + 1}: {len(partition_df)} rows")
                
            except Exception as e:
                print(f"  ✗ Error writing partition {partition_idx}: {e}")
                raise
            
            # Return partition unchanged (required by map_partitions)
            return partition_df
        
        # Write all header partitions
        self.headers.map_partitions(
            write_header_partition,
            meta=self.headers._meta
        ).compute()
        
        print(f"✓ Headers written: {self.headers.npartitions} files in {headers_path}")
        
        # Clean up
        gc.collect()
        
        # ============================================
        # Write Data to Zarr
        # ============================================
        data_path = output_path / 'data.zarr'
        
        # Write using Dask's to_zarr (efficient chunked writing)
        self.data.to_zarr(
            str(data_path),
            component='data',  # Creates data.zarr/data structure
            overwrite=overwrite
        )
        
        print(f"✓ Data written to {data_path}")
        
        # Clean up
        gc.collect()
        
        print(f"✅ ZarrVector successfully written to {output_path}")
        print(f"   Total: {self.n_traces:,} traces × {self.n_samples} samples")
    
    @classmethod
    def read(cls, path: Union[str, Path]):
        """
        Read from disk.
        
        Args:
            path: Base directory path
            
        Returns:
            ZarrVector instance
        """
        path = Path(path)
        return cls(
            headers_path=path / 'headers.parquet',
            traces_path=path / 'data.zarr' / 'data'
        )
    
    # ============================================
    # Utility Methods
    # ============================================
    
    def hash(self):
        """Compute hash of the data (for verification)"""
        # Hash the data array
        data_hash = hashlib.sha256(
            self.data.compute().tobytes()
        ).hexdigest()
        
        return data_hash
    
    def compute(self):
        """
        Trigger computation and return in-memory version.
        
        Returns:
            Tuple of (pandas DataFrame, numpy array)
        """
        return self.headers.compute(), self.data.compute()
    
    def filter_headers(self, query: str):
        """
        Filter based on header values.
        
        Args:
            query: Pandas query string (e.g., "sx > 1000 and sy < 2000")
            
        Returns:
            New ZarrVector with filtered data
        """
        # Filter headers
        filtered_headers = self.headers.query(query)
        
        # Get trace indices
        if 'trace_id' not in filtered_headers.columns:
            raise ValueError("Headers must have 'trace_id' column for filtering")
        
        trace_indices = filtered_headers['trace_id'].values
        
        # Filter data
        filtered_data = self.data[trace_indices, :]
        
        return ZarrVector(
            headers=filtered_headers.reset_index(drop=True),
            data=filtered_data
        )
    
    def persist(self):
        """
        Persist data in memory (useful for iterative algorithms).
        """
        self.headers = self.headers.persist()
        self.data = self.data.persist()
        return self
    
    def __repr__(self):
        return (
            f"ZarrVector(\n"
            f"  n_traces={self.n_traces},\n"
            f"  n_samples={self.n_samples},\n"
            f"  headers={list(self.headers.columns)},\n"
            f"  dtype={self.data.dtype}\n"
            f")"
        )