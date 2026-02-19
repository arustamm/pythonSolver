import pyVector
import numpy as np
import zarr
import dask.array as da
import os
import shutil
import uuid
import weakref
import atexit
from typing import Tuple, List, Union
import hashlib

class ZarrVector(pyVector.vector):
    _temp_instances = weakref.WeakSet()

    def __init__(self, 
                 path: str = None, 
                 ns_list: List[int] = None,
                 ds_list: List[float] = None,
                 os_list: List[float] = None, 
                 shape: Tuple[int] = None, 
                 chunks: Tuple[int] = None,
                 shards: Tuple[int] = None,  
                 dtype: np.dtype = np.float32, 
                 dask_array = None, # Renamed/Updated for Dask
                 temp_dir: str = '/tmp/',
                 remove_file: bool = False):
        
        self.remove_file = remove_file
        self._temp_dir = temp_dir
        self.shards = shards
        if not os.path.exists(self._temp_dir):
            os.makedirs(self._temp_dir, exist_ok=True)

        # --- 1. Metadata Setup ---
        # We cache metadata in the object because Dask operations (like + or *) 
        # often drop custom .attrs from the resulting array.
        self._ns = ns_list
        self._ds = ds_list
        self._os = os_list

        # Determine Shape from ns_list if not provided
        if shape is None and ns_list is not None:
            shape = tuple(reversed(ns_list))

        # --- 2. Dask Array Setup ---
        
        # CASE A: Wrapped Dask Array (e.g., resulting from a slice or math op)
        if dask_array is not None:
            self.da = dask_array
            # If path is provided, we might be backed by it, otherwise it's a graph
            self.path = path 
        
        # CASE B: Existing Zarr Path
        elif path is not None and os.path.exists(path):
            self.path = path
            # Lazy load from Zarr
            self.da = da.from_zarr(self.path)
            
            # Try to load metadata if missing from init args
            if self._ns is None:
                # We need to open the zarr handle briefly to read attrs if dask didn't catch them
                # Dask usually exposes attrs via self.da.attrs, but let's be safe
                za = zarr.open_array(self.path, mode='r')
                self._ns = za.attrs.get('ns')
                self._ds = za.attrs.get('ds')
                self._os = za.attrs.get('os')

        # CASE C: New Vector (Create backing storage)
        else:
            if path is None:
                temp_id = str(uuid.uuid4())
                path = os.path.join(self._temp_dir, f"zarr_vec_{temp_id}.zarr")
                if not self.remove_file: self.remove_file = True
            
            self.path = path
            
            # Create the physical Zarr file to define shape/chunks
            za = zarr.create_array(
                store=path, shape=shape, chunks=chunks, dtype=dtype, shards=shards
            )
            # Initialize with zeros
            za[:] = 0
            
            # Write metadata to disk
            if self._ns is not None:
                za.attrs['ns'] = self._ns
                za.attrs['ds'] = self._ds if self._ds else [1.0]*len(self._ns)
                za.attrs['os'] = self._os if self._os else [0.0]*len(self._ns)
            
            # Load as Dask Array
            self.da = da.from_zarr(self.path)

        # Register cleanup if we own the file
        if self.remove_file and self.path:
            self._finalizer = weakref.finalize(self, self._cleanup_path, self.path)
            ZarrVector._temp_instances.add(self)

    # --- Properties ---
    # We implement shape/dtype/ndim forwarding to the underlying dask array
    def hash(self):
        """
        Computes SHA1 hash of the array content.
        """
        # Define function to run on each chunk
        def compute_chunk_hash(chunk):
            sha = hashlib.sha1()
            
            # Ensure consistent byte order and contiguous memory
            # (Important for accurate hashing)
            if hasattr(chunk, 'astype'):
                data = np.ascontiguousarray(chunk)
                sha.update(data.tobytes())
            
            hash_str = sha.hexdigest()
            
            # --- THE FIX ---
            # Create a 1D array containing the hash
            res = np.array([hash_str], dtype=object)
            
            # Reshape it to match the input chunk's dimensionality.
            # If chunk is 3D, this makes res shape (1, 1, 1)
            # If chunk is 2D, this makes res shape (1, 1)
            return res.reshape((1,) * chunk.ndim)

        # We promise Dask that every input chunk results in a 1x1x... output chunk
        output_chunks = tuple([(1,)*len(c) for c in self.da.chunks])
        
        lazy_hashes = self.da.map_blocks(
            compute_chunk_hash,
            dtype=object,
            chunks=output_chunks
        )
        
        # Trigger Compute: Reads all data and returns flat array of hashes
        chunk_hashes = lazy_hashes.compute().flatten()
        
        # Combine all chunk hashes into one final hash
        final_sha = hashlib.sha1()
        
        # Sort to ensure deterministic order (Dask might return chunks out of order)
        chunk_hashes.sort() 
        
        for h in chunk_hashes: 
            final_sha.update(h.encode('utf-8'))
            
        return final_sha.hexdigest()

    def isDifferent(self, other):
        return self.hash() != other.hash()

    @property
    def shape(self): return self.da.shape

    @property
    def dtype(self): return self.da.dtype

    @property
    def ndim(self): return self.da.ndim

    @property
    def size(self): return self.da.size
    
    @property
    def chunks(self): return self.da.chunksize

    @property
    def ns(self): return self._ns

    @property
    def ds(self): return self._ds

    @property
    def os(self): return self._os

    # --- Math Operations (Lazy Dask) ---
    # These update the computational graph (self.da) but do not compute immediately.

    def zero(self):
        self.da = da.zeros_like(self.da)
        return self

    def set(self, val):
        self.da = da.ones_like(self.da) * val
        return self

    def scale(self, sc):
        self.da = self.da * sc
        return self

    def scaleAdd(self, vec2, sc1=1.0, sc2=1.0):
        self.checkSame(vec2)
        # Lazy graph construction
        self.da = (self.da * sc1) + (vec2.da * sc2)
        return self

    def addbias(self, bias):
        self.da = self.da + bias
        return self
    
    def multiply(self, vec2):
        self.checkSame(vec2)
        self.da = self.da * vec2.da
        return self

    def rand(self):
        # Dask has its own random state generators
        rng = da.random.default_rng()
        if np.iscomplexobj(self.dtype):
            r = rng.random(self.shape, chunks=self.chunks).astype(self.dtype.real.dtype)
            i = rng.random(self.shape, chunks=self.chunks).astype(self.dtype.real.dtype)
            self.da = r + 1j*i
        else:
            self.da = rng.random(self.shape, chunks=self.chunks).astype(self.dtype)
        return self

    # --- Reductions (Trigger Compute) ---
    # Solvers usually need scalar values immediately, so we call .compute()

    def dot(self, vec2):
        self.checkSame(vec2)
        # da.dot or vdot handles the reduction tree efficiently
        # flatten() is cheap in Dask (just metadata manip)
        d1 = self.da.flatten()
        d2 = vec2.da.flatten()
        # vdot conjugates the first argument for complex inputs
        res = da.vdot(d1, d2)
        return res.compute()

    def norm(self, N=2):
        # da.linalg.norm is optimized
        res = da.linalg.norm(self.da.flatten(), ord=N)
        return float(res.compute())

    def min(self):
        return float(self.da.min().compute())

    def max(self):
        return float(self.da.max().compute())

    # --- IO & Accessors ---

    def __getitem__(self, key):
        return self.da[key]
    
    # Note: __setitem__ in Dask is limited. 
    # self.da[key] = val is NOT supported for general dask arrays because they are immutable graphs.
    # However, for full assignment we can sometimes hack it, but generally in Dask 
    # you construct NEW arrays rather than modify old ones.
    # For pyVector compatibility, if users do vec[:] = val, we can map to set()
    def __setitem__(self, key, value):
        self.da[key] = value  

    def copy(self, vec2):
        self.da = vec2.da.copy()
        # Copy metadata as well
        self._ns = vec2.ns
        self._os = vec2.os
        self._ds = vec2.ds
        return self

    def clone(self):
        # Return a new object with the same graph
        # This is lightweight.
        new_vec = ZarrVector(
            dask_array=self.da.copy(),
            ns_list=self._ns, ds_list=self._ds, os_list=self._os, shards=self.shards,
            remove_file=True # Clones usually ephemeral unless written
        )
        return new_vec

    def cloneSpace(self):
        # Create a zero-filled vector with same structure
        zeros = da.zeros_like(self.da)
        return ZarrVector(
            dask_array=zeros,
            ns_list=self._ns, ds_list=self._ds, os_list=self._os,
            remove_file=True
        )

    def writeVec(self, filename, mode='w'):
        """Triggers computation and writes the current graph to disk."""
        if mode == 'a':
            if not os.path.exists(filename): os.makedirs(filename)
            idx = len([n for n in os.listdir(filename) if "iter" in n])
            out_path = os.path.join(filename, f"iter_{idx:05d}.zarr")
        else:
            out_path = filename
        
        # We use da.to_zarr (or da.store) to compute and write in parallel
        # This is the "Eager" trigger for Dask.
        
        # 1. Write Data
        # compute=True makes it block until finished
        self.da.to_zarr(out_path, overwrite=True, compute=True)
        
        # 2. Write Metadata
        # We have to open the file we just wrote to add the custom SEP attributes
        # (da.to_zarr doesn't easily let us inject custom attrs during write)
        za = zarr.open_array(out_path, mode='r+')
        if self._ns:
            za.attrs['ns'] = self._ns
            za.attrs['ds'] = self._ds if self._ds else [1.0]*len(self._ns)
            za.attrs['os'] = self._os if self._os else [0.0]*len(self._ns)

    def window(self, slices):
        """
        Returns a new ZarrVector representing the window (Lazy).
        Does NOT trigger compute.
        """
        # 1. Lazy Slice
        window_da = self.da[slices]
        
        # 2. Calculate New Metadata (Origin Shift)
        # Using the logic derived for "Robust In-Memory Windowing"
        new_ns = list(reversed(window_da.shape))
        new_ds = list(self.ds) if self.ds else None
        new_os = list(self.os) if self.os else None
        
        if self.os and self.ds:
            ndim = self.ndim
            for i, sl in enumerate(slices):
                sep_idx = ndim - 1 - i
                
                # Handle standard slices
                if isinstance(sl, slice):
                    start = sl.start if sl.start is not None else 0
                    offset_samples = int(start)
                    new_os[sep_idx] = self.os[sep_idx] + (offset_samples * self.ds[sep_idx])
                
                # Handle integer indexing (reducing a dimension)
                elif isinstance(sl, (int, np.integer)):
                    offset_samples = int(sl)
                    new_os[sep_idx] = self.os[sep_idx] + (offset_samples * self.ds[sep_idx])

        return ZarrVector(
            dask_array=window_da,
            ns_list=new_ns, ds_list=new_ds, os_list=new_os,
            remove_file=False
        )
    
    def to_numpy(self, slices=None):
        """
        Reads a window directly into RAM for the Worker.
        """
        # 1. Get the Window Vector (Lazy)
        if slices is not None:
            win_vec = self.window(slices)
            data = win_vec.da.compute()
            return data, win_vec.ns, win_vec.os, win_vec.ds
        else:
            data = self.da.compute()
            return data, self.ns, self.os, self.ds

    # --- Utils ---

    def _copy_metadata_to(self, target_zarr):
        # Helper used if manually touching zarr arrays
        if self._ns: target_zarr.attrs['ns'] = self._ns
        if self._ds: target_zarr.attrs['ds'] = self._ds
        if self._os: target_zarr.attrs['os'] = self._os

    def checkSame(self, other):
        if self.shape != other.shape:
            return False
        return True
    
    @staticmethod
    def _cleanup_path(path):
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
            except OSError:
                pass

    @classmethod
    def cleanup(cls):
        for instance in list(cls._temp_instances):
            if getattr(instance, 'remove_file', False) and instance.path:
                ZarrVector._cleanup_path(instance.path)
        cls._temp_instances.clear()

    # Pickling
    # Dask arrays pickle efficiently (graph + metadata). 
    # We do NOT need the complex __setstate__ logic anymore because 
    # we aren't passing raw open file handles (self.za) around.
    def __getstate__(self):
        state = self.__dict__.copy()
        if '_finalizer' in state: del state['_finalizer']
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        # On restore, we are a worker copy. We do not own the file.
        self.remove_file = False 

atexit.register(ZarrVector.cleanup)