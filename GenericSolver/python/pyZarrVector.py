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
                 existing_zarr_array = None,
                 temp_dir: str = '/tmp/',
                 overwrite: bool = False,
                 remove_file: bool = False):
        
        self.remove_file = remove_file
        self._temp_dir = temp_dir
        if not os.path.exists(self._temp_dir):
            os.makedirs(self._temp_dir, exist_ok=True)

        # 1. Determine Shape
        if existing_zarr_array is None and shape is None and ns_list is not None:
            shape = tuple(reversed(ns_list))
        
        # 2. Setup Zarr Storage
        if existing_zarr_array is not None:
            self.za = existing_zarr_array
            if path is not None:
                self.path = path
            else:
                self.path = getattr(self.za.store, 'path', None)
        else:
            if path is None:
                # FIX: Respect the remove_file flag if passed
                temp_id = str(uuid.uuid4())
                path = os.path.join(self._temp_dir, f"vector_{temp_id}.zarr")
                if not self.remove_file: self.remove_file = True
            
            self.path = path
            needs_new = not os.path.exists(self.path) or overwrite
            mode = 'w' if needs_new else 'r+'
            # Create/Open Zarr Array
            self.za = zarr.open_array(
                store=path,
                shape=shape if needs_new else None,
                chunks=chunks if needs_new else None,
                dtype=dtype,
                mode=mode,
            )
            
            if needs_new and ns_list is not None:
                self.za.attrs['ns'] = ns_list
                self.za.attrs['ds'] = ds_list if ds_list else [1.0] * len(ns_list)
                self.za.attrs['os'] = os_list if os_list else [0.0] * len(ns_list)

        self.shape = self.za.shape
        self.dtype = self.za.dtype
        self.size = self.za.size
        self.ndim = len(self.shape)
        self.chunks = self.za.chunks
        self.shards = getattr(self.za, 'shards', None)
        
        # Cleanup Registration
        self._register_finalizer()

    def _register_finalizer(self):
        if self.path and not hasattr(self, '_finalizer'):
            self._finalizer = weakref.finalize(self, self._cleanup_path, self.path)
            if self.remove_file:
                ZarrVector._temp_instances.add(self)
            else:
                self._finalizer.detach()

    # --- HELPER: Dask Interface ---

    def _as_dask(self):
        """Returns a lazy dask array representation of this vector."""
        # We explicitly pass chunks to ensure Dask tasks align with Zarr chunks
        return da.from_zarr(self.za, chunks=self.chunks)

    def _write_dask(self, dask_arr):
        meta = dict(self.za.attrs)
        temp_store = self.path + "_write_temp"
        if os.path.exists(temp_store):
            shutil.rmtree(temp_store)
        dask_arr.to_zarr(temp_store, overwrite=True, compute=True)
        shutil.rmtree(self.path)
        shutil.move(temp_store, self.path)
        self.za = zarr.open_array(self.path, mode='r+')
        self.za.attrs.update(meta)

    # --- Math Operations (Parallelized) ---

    def zero(self):
        # Create a dask array of zeros with correct chunking
        zeros = da.zeros(self.shape, chunks=self.chunks, dtype=self.dtype)
        self._write_dask(zeros)
        return self

    def set(self, val):
        # Broadcast value to full array
        d = self._as_dask()
        d[:] = val
        self._write_dask(d)
        return self

    def scale(self, sc):
        d = self._as_dask()
        res = d * sc
        self._write_dask(res)
        return self

    def scaleAdd(self, vec2, sc1=1.0, sc2=1.0):
        self.checkSame(vec2)
        
        d1 = self._as_dask()
        d2 = vec2._as_dask()
        
        # This builds a graph to read d1, read d2, multiply, add
        res = d1 * sc1 + d2 * sc2
        
        # Triggers the computation and parallel write
        self._write_dask(res)
        return self

    def addbias(self, bias):
        d = self._as_dask()
        res = d + bias
        self._write_dask(res)
        return self
    
    def multiply(self, vec2):
        self.checkSame(vec2)
        d1 = self._as_dask()
        d2 = vec2._as_dask()
        res = d1 * d2
        self._write_dask(res)
        return self

    def rand(self):
        # Dask has its own parallel random generator
        rng = da.random.default_rng()
        
        if np.iscomplexobj(self.dtype):
            r = rng.random(self.shape, chunks=self.chunks).astype(self.dtype.real.dtype)
            i = rng.random(self.shape, chunks=self.chunks).astype(self.dtype.real.dtype)
            data = r + 1j*i
        else:
            data = rng.random(self.shape, chunks=self.chunks).astype(self.dtype)
            
        self._write_dask(data)
        return self

    # --- Reductions (Parallelized) ---

    def dot(self, vec2):
        self.checkSame(vec2)
        d1 = self._as_dask().flatten()
        d2 = vec2._as_dask().flatten()
        
        # vdot handles complex conjugation: sum(x * conj(y))
        # Dask handles the reduction tree (summing chunks -> summing sums)
        return da.vdot(d1, d2).compute()

    def norm(self, N=2):
        d = self._as_dask()
        # linalg.norm is generally optimized in Dask
        return float(da.linalg.norm(d.flatten(), ord=N).compute())

    def min(self):
        return float(self._as_dask().min().compute())

    def max(self):
        return float(self._as_dask().max().compute())

    # --- IO & Accessors ---

    def __getitem__(self, key):
        return self.za[key]
    
    def __setitem__(self, key, value):
        self.za[key] = value

    def copy(self, vec2):
        self.checkSame(vec2)
        # Directly pipe one dask array into the other's storage
        self._write_dask(vec2._as_dask())
        return self

    def clone(self):
        new_vec = self.cloneSpace()
        new_vec.copy(self)
        return new_vec

    def cloneSpace(self):
        unique_id = str(uuid.uuid4())
        new_path = os.path.join(self._temp_dir, f"clone_{unique_id}.zarr")
        
        # Just create empty array structure
        new_za = zarr.open_array(
            store=new_path, shape=self.shape, 
            chunks=self.chunks, dtype=self.dtype, mode='w'
        )
        
        # Clone metadata
        self._copy_metadata_to(new_za)
        
        return ZarrVector(existing_zarr_array=new_za, 
                          path=new_path,
                          temp_dir=self._temp_dir, 
                          overwrite=False, # Already created above
                          remove_file=True)

    def writeVec(self, filename, mode='w'):
        if mode == 'a':
            if not os.path.exists(filename): os.makedirs(filename)
            idx = len([n for n in os.listdir(filename) if "iter" in n])
            out_path = os.path.join(filename, f"iter_{idx:05d}.zarr")
        else:
            out_path = filename
            
        # Dask's to_zarr is extremely efficient for this
        d = self._as_dask()
        
        # to_zarr creates the file and writes data in parallel
        # We pass arguments to ensure it matches our specs
        d.to_zarr(out_path, 
                  overwrite=True, 
                  compute=True,
                  storage_options={'chunks': self.chunks}) 
        
        # Re-open to write SEP metadata which Dask ignores
        # (Alternatively, Dask creates .zattrs, we just update it)
        target_za = zarr.open_array(out_path, mode='r+')
        self._copy_metadata_to(target_za)

    def window(self, slices):
        temp_id = str(uuid.uuid4())
        path = os.path.join(self._temp_dir, f"win_{temp_id}.zarr")
        
        # Use Dask slicing (lazy)
        d_window = self._as_dask()[slices]
        
        # Write window to new Zarr
        d_window.to_zarr(path, overwrite=True, compute=True)
        
        new_za = zarr.open_array(path, mode='r+')

        if self.ns and self.os and self.ds:
            # Metadata update logic (Keep your existing logic, it's fine)
            new_ns = list(self.ns)
            new_ds = list(self.ds) 
            new_os = list(self.os) 
            
            ndim = len(self.shape)
            for i, sl in enumerate(slices):
                start = sl.start if isinstance(sl, slice) else sl
                if start is None: start = 0
                sep_idx = ndim - 1 - i
                new_os[sep_idx] = self.os[sep_idx] + start * self.ds[sep_idx]

            new_za.attrs['ns'] = new_ns
            new_za.attrs['ds'] = new_ds
            new_za.attrs['os'] = new_os
        
        return ZarrVector(existing_zarr_array=new_za, 
                          path=path,
                          temp_dir=self._temp_dir, 
                          overwrite=True,
                          remove_file=True)
    
    def to_numpy(self, slices=None):
        """
        Reads a slice directly into memory without creating a temp Zarr file.
        Returns: Tuple (numpy_data, ns, os, ds)
        """
        if slices is None:
            return self.za[:], self.ns, self.os, self.ds
        # 1. Read directly from Global Zarr to RAM
        data = self.za[slices] 
        
        # 2. Calculate Metadata (same logic as your window function)
        new_ns = list(reversed(data.shape))
        new_ds = list(self.ds)
        new_os = list(self.os)
        
        ndim = len(self.shape)
        for i, sl in enumerate(slices):
            start = sl.start if isinstance(sl, slice) else sl
            if start is None: start = 0
            sep_idx = ndim - 1 - i
            new_os[sep_idx] = self.os[sep_idx] + start * self.ds[sep_idx]
            
        return data, new_ns, new_os, new_ds

    # --- Utils ---

    def _copy_metadata_to(self, target_zarr):
        target_zarr.attrs.update(self.za.attrs)

    def checkSame(self, other):
        if self.shape != other.shape:
            return False
        if self.za.chunks != other.za.chunks:
            return False
        return True
    
    def hash(self):
        """
        Computes SHA1 hash of the array content.
        """
        d = self._as_dask()
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
        output_chunks = tuple([(1,)*len(c) for c in d.chunks])
        
        lazy_hashes = d.map_blocks(
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
    def ns(self):
        return self.za.attrs.get('ns')

    @property
    def ds(self):
        return self.za.attrs.get('ds')

    @property
    def os(self):
        return self.za.attrs.get('os')

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
    def __getstate__(self):
        state = self.__dict__.copy()
        self._register_finalizer()
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self._register_finalizer()
        
    @staticmethod
    def sum(vectors: List['ZarrVector']) -> 'ZarrVector':
        """
        Optimized Parallel Reduction using Dask.
        1. Converts all input vectors to lazy Dask arrays.
        2. Sums them symbolically (in memory).
        3. Writes the result to disk in parallel (one pass).
        """
        if not vectors:
            return None
        
        # 1. Use the first vector as a template for shape/chunks/path generation
        template = vectors[0]
        
        # 2. Create the Result Vector container
        # cloneSpace creates a new unique path and registers cleanup handles
        result = template.cloneSpace()
        
        # 3. Build the Dask Graph
        # We convert all ZarrVectors to lazy dask arrays
        dask_vecs = [v._as_dask() for v in vectors]
        
        # This creates a summation graph: sum(chunk_i_vec1, chunk_i_vec2, ...)
        # It does NOT compute yet.
        total_lazy = sum(dask_vecs)
        
        # 4. Compute and Write
        # to_zarr(overwrite=True) ensures we replace the empty array created by cloneSpace
        # compute=True triggers the actual parallel execution
        total_lazy.to_zarr(result.path, overwrite=True, compute=True)
        
        # 5. Restore Metadata
        # Since overwrite=True wipes the directory (including attributes),
        # we must re-open the array and re-apply the SEP metadata.
        result.za = zarr.open_array(result.path, mode='r+')
        
        if template.ns: result.za.attrs['ns'] = template.ns
        if template.ds: result.za.attrs['ds'] = template.ds
        if template.os: result.za.attrs['os'] = template.os
            
        return result

atexit.register(ZarrVector.cleanup)