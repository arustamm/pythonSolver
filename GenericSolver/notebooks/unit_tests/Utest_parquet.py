
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import os

def create_test_parquet(output_dir, num_rows=10000, vector_size=5000, num_partitions=10):
    """
    Create a test Parquet file with multiple partitions.
    
    :param output_dir: Directory to save the Parquet file
    :param num_rows: Total number of rows in the table
    :param vector_size: Size of each vector in the "data" column
    :param num_partitions: Number of partitions to create
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate rows per partition
    rows_per_partition = num_rows // num_partitions
    
    for i in range(num_partitions):
        # Generate random data for this partition
        data = [np.random.rand(vector_size).astype(np.float32) for _ in range(rows_per_partition)]
        
        # Create a PyArrow array from the data
        arr = pa.array(data)
        
        # Create a PyArrow table
        table = pa.table({'data': arr})
        
        # Write the partition
        pq.write_table(table, os.path.join(output_dir, f'part-{i}.parquet'))

    print(f"Created {num_partitions} partitions in {output_dir}")

# Usage
output_directory = f'{os.getenv("DATAPATH")}/test_parquet_data'
create_test_parquet(output_directory)