import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pathlib
import os
import dask
from dask.distributed import wait
import segyio
import math
import logging

 # Configure logging
logging.basicConfig(filename='segy_to_parquet.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

def segy_to_parquet(file:pathlib.PosixPath, path_to_save:str, headers: dict, batch: int):
    with segyio.open(file, "r", ignore_geometry=True) as f:
        # Create the base Parquet dataset
        base = pathlib.Path(path_to_save)
        base.mkdir(exist_ok=True)
        name, _ = os.path.splitext(file.name)
        path = base / (name + ".pq")

        num_samples = f.samples.size
        metadata = {
            "nt" : str(num_samples),
            "dt" : str(segyio.tools.dt(f) / 1_000_000),
            "ot" : str(0),
         }
        
        writer = None
        num_traces = len(f.trace)
        if batch is None:
            batch = num_traces

        num_full_batches = int(num_traces // batch)
        remaining = num_traces % batch
        if remaining > 0: num_full_batches += 1

        for i in range(num_full_batches):
            start = int(i * batch)
            end = int(min(start + batch, num_traces))
            traces = f.trace.raw[start:end]
            trace_arr = pa.array([trace for trace in traces], type=pa.list_(pa.float32()))

            data = [trace_arr]
            fields = [pa.field("data", pa.list_(pa.float32()))]

            for k, v in headers.items():
                h_arr = f.attributes(v)[start:end]
                fields.append(pa.field(k, type=pa.from_numpy_dtype(h_arr.dtype)))
                data.append(pa.array(h_arr))

            schema = pa.schema(fields, metadata=metadata)
            table = pa.Table.from_arrays(data, schema=schema)

            if writer is None:
                writer = pq.ParquetWriter(path, schema, compression='snappy')  # Initialize writer

            writer.write_table(table)

            logging.info(f"Processed {name} traces batch {i+1}/{num_full_batches}")

        if writer:
            writer.close()

def segy_to_parquet_dask(client, files: list, path_to_save: str, headers: dict, trace_batch=None, file_batch:int=None):

    if file_batch is None:
        file_batch = len(files)
    
    num_full_batches = int(len(files) // file_batch)
    remaining_files = len(files) % file_batch
    if remaining_files > 0: num_full_batches += 1

    # Process full batches
    for i in range(num_full_batches):
        start = i * file_batch
        end = min(start + file_batch, len(files))
        batch = files[start:end]
        results = client.map(segy_to_parquet, batch, [path_to_save]*len(batch), [headers]*len(batch), [trace_batch]*len(batch))
        wait(results)
        print(results[0])
        logging.info(f"Processed file batch {i+1}/{num_full_batches}")
    

def get_files_with_extension(directory: str, extension: str) -> list:
  path = pathlib.Path(directory)
  return list(path.glob(f"*.{extension}"))

