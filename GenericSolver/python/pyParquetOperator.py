import numpy as np
import pyarrow.dataset as ds
import pyarrow.compute as pc
import pyarrow as pa

from pyOperator import Operator
from pyParquetVector import ParquetVector
from pyVector import vectorIC

import dask.dataframe as dd
import dask.bag as db
import pandas as pd
import random
import string
import os
import dask
import shutil
import hashlib

class ParquetOperator(Operator):

    def __init__(self, operator_cls, domain: vectorIC, range: ParquetVector, *args, **kwargs):
        self.operator_cls = operator_cls
        

    def forward(self, add, model, data):
        """
            Should be not blocking
        """
        if not add: data.zero()

        data.df = data.df.map_partitions(
            fwd, model, self.operator_cls,
        )

    def adjoint(self, add, model, data):
        pass

def fwd(data: pd.Series, model: vectorIC, op_cls) -> pd.Series:
    # initialize the operator
    op = op_cls(model, data)
    return op.forward(model)
    