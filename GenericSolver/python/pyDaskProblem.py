import pandas as pd
import numpy as np
import dask
import pyVector as Vector
import pyOperator as Operator
from pyProblem import Problem  # Assuming base class is in pyProblem.py
from pyParquetVector import pyParquetVector  # Assuming this is your Dask-compatible vector

# This helper function must be defined at the top level for Dask serialization
def _compute_obj_grad_partition(
    data_partition_pd: pd.DataFrame,
    model: Vector.vector,
    nlop, # Your non-linear operator class/builder
    gop,  # Your gradient operator class/builder
    data_loader, # Function to convert pandas DataFrame to your custom pyVector
) -> pd.DataFrame:
    """
    Computes objective and gradient for a single data partition.
    
    Args:
        data_partition_pd: A pandas DataFrame representing one partition of the data.
        model: The current model vector (fits in memory).
        nlop: The non-linear operator (or a function to create it).
        gop: The gradient operator (or a function to create it).
        data_cols: List of column names in data_partition_pd to be treated as data.
    
    Returns:
        A single-row pandas DataFrame with 'norm_sq' and 'grad'.
    """
    
    # 1. Convert pandas partition to a pyVector
    # This is specific to your data. Example:
    # data_vec = Vector.from_numpy(data_partition_pd[data_cols].values)
    # You might need to instantiate a custom pyVector object here.
    # Let's assume you have a function for this:
    data = data_loader(data_partition_pd)
    
    # 2. Instantiate operators for this partition
    # (This logic depends on your nlop/gop setup)
    nl_op = nlop.from_subspace(model, data, ...)
    g_op = gop.from_operator(model, data, nl_op, ...)

    # Example for a standard L2 problem: r = f(m) - d
    res = data.clone()
    grad = model.clone().zero()
    
    # 3. Forward: r = f(m) - d
    nl_op.forward(False, model, res)  # res = f(m)
    res.scaleAdd(data, 1.0, -1.0)   # res = f(m) - d
    
    # 4. Objective: J = ||r||^2
    norm_sq = res.dot(res)
    
    # 5. Adjoint: g = f'(m)* r
    g_op.adjoint(False, grad, res)
    
    # 6. Return as a single-row DataFrame
    # The 'grad' column contains the full gradient vector object for this partition.
    return pd.DataFrame({"norm_sq": [norm_sq], "grad": [grad]})


class DaskParquetProblem(Problem):
    """
    A pyProblem class that operates on a Dask DataFrame (from Parquet)
    and computes the objective and gradient in a fused, out-of-core manner.
    
    Assumes an L2 non-linear problem: J = 1/2 * ||f(m) - d||^2
    """

    def __init__(self, 
                 model_template: Vector.vector, 
                 data: pyParquetVector,
                 nlop_builder, # Function/class to build the non-linear operator
                 gop_builder,  # Function/class to build the gradient operator
                 data_loader,   # Function to convert pandas DataFrame to custom pyVector
                 resource_constraints: dict = None,
                 **kwargs):
        """
        Constructor
        
        Args:
            model_template: A pyVector of the correct shape for the model.
            data_ddf: The Dask DataFrame containing the observed data.
            nlop_builder: A class or function to build the non-linear operator.
            gop_builder: A class or function to build the gradient/adjoint operator.
            data_cols: List of column names in data_ddf to use as data.
            kwargs: Any other static arguments needed by your operator builders.
        """
        # Initialize base class
        super(DaskParquetProblem, self).__init__()

        # Store Dask-related items
        self.data = data  # The base class uses 'self.data'
        
        # Store operator builders and static args
        self.nlop_builder = nlop_builder
        self.gop_builder = gop_builder
        self.data_loader = data_loader
        self.op_kwargs = kwargs # Store other static args
        # Specify resource constraints for Dask jobs, i.e. {'GPU' : 1}
        self.resource_constraints = resource_constraints or {}
        dask.config.set({"optimization.fuse.active": False})
        
        # Set up model, gradient, and residual templates
        # These are essential for the base class and for cloning
        self.model = model_template.clone()
        self.grad = model_template.clone().zero()
        
        # We explicitly DO NOT create self.res, as it's too large.
        # self.res = data_template.clone() # <-- DO NOT DO THIS
        
        self.setDefaults()

    def objgradf(self, model):
        """
        Computes objective and gradient in a single, fused Dask operation.
        This is the core of the class.
        """
        # Define the output structure for map_partitions
        meta_df = pd.DataFrame({'norm_sq': pd.Series(dtype='float64'),
                                'grad': pd.Series(dtype='object')})

        # Run the map-partitions (map) and sum (reduce)
        with dask.annotate(resources=self.resource_constraints):
            res_df = self.data.map_partitions(
                _compute_obj_grad_partition,
                model,
                self.nlop_builder,
                self.gop_builder,
                self.data_loader,
                **self.op_kwargs,
                meta=meta_df
            )

        # compute() triggers the Dask computation and returns a pandas DataFrame
        # The .sum() performs:
        # 1. A float sum on the 'norm_sq' column.
        # 2. A vector-sum (obj + obj) on the 'grad' column.
        summed_df = res_df.sum().compute()

        # Extract results
        self.obj = 0.5 * summed_df["norm_sq"]
        self.grad.copy(summed_df["grad"]) # Copy the summed gradient vector

        # Set flags
        self.obj_updated = True
        self.grad_updated = True
        self.fevals += 1
        self.gevals += 1

        return self.obj, self.grad

    # --- Override essential base class methods ---

    def get_obj(self, model):
        """Overrides base class 'get_obj' to use our fused Dask method."""
        self.set_model(model)
        if not self.obj_updated:
            # This calls objgradf and sets self.obj and self.grad
            self.get_obj_grad(model)
        return self.obj

    def get_grad(self, model):
        """Overrides base class 'get_grad' to use our fused Dask method."""
        self.set_model(model)
        if not self.grad_updated:
            # This calls objgradf and sets self.obj and self.grad
            self.get_obj_grad(model)
        return self.grad
        
    def get_obj_grad(self, model):
        """
        Accessor for objective function and gradient vector.
        This overrides the base method to call our specific objgradf.
        """
        self.set_model(model)
        if not self.obj_updated:
            # This is the only place the computation is triggered
            self.obj, self.grad = self.objgradf(model)
        return self.obj, self.grad

    # --- Ban methods that materialize the full residual ---

    def resf(self, model):
        """Residual vector is too large to compute and store."""
        raise NotImplementedError("DaskParquetProblem does not support materializing the full residual vector!")

    def dresf(self, model, dmodel):
        """dres is also too large to materialize."""
        raise NotImplementedError("DaskParquetProblem does not support dres computation!")

    def get_res(self, model):
        """Accessor for residual vector."""
        raise NotImplementedError("DaskParquetProblem does not support get_res!")