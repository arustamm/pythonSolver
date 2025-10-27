# Module containing generic Solver and Restart definitions

from sys import path
path.insert(0, '.')
import pyProblem
import pyVector as Vec
import atexit
import os
# Functions and modules necessary for writing on disk
import pickle
import re
import numpy as np
from math import isnan

import sep_util as sepu
from sys_util import mkdir

from shutil import rmtree
from copy import deepcopy
import datetime


class Solver:
    """Solver parent object"""

    # Default class methods/functions
    def __init__(self, stopper, stepper, proxOp=None, logger=None):
        """Default class constructor for Solver"""
        # Parameter for saving results
        self.save_obj = False
        self.save_res = False
        self.save_grad = False
        self.save_model = False
        self.flush_memory = False

        self.prefix = None

        # Iteration axis-sampling parameters
        self.iter_buffer_size = None
        self.iter_sampling = 1
        self.iter = 0

        # Lists of the results (list and vector Sets)
        self.obj = list()
        self.last_obj_value = None
        self.obj_terms = list()
        self.model = list()
        self.res = list()
        self.grad = list()
        self.modelSet = Vec.vectorSet()
        self.resSet = Vec.vectorSet()
        self.gradSet = Vec.vectorSet()
        self.inv_model = None
        self.iter_written = 0
        self.overwrite = True

        # Set Restart object
        self.restart = Restart()
        self.create_msg = False
        
        self.stopper = stopper
        self.stepper = stepper
        self.stepper.proxOp = proxOp
        self.logger = logger
        self.stopper.logger = self.logger
        self.iter_msg = "\niter = %s, obj = %.5e, gradnorm = %.2e, feval = %d, geval = %d"
        return
    
    def log_message(self, msg, verbose=False):
        if verbose:
            print(msg)
        if self.logger:
            self.logger.addToLog(msg)

    def initialize_solver(self, problem, verbose=False, restart_path: str=None):
        self.stopper.reset()
        problem.reset()
        if hasattr(problem, 'objgradf') and self.save_res:
            self.log_message("WARNING: using objgradf. Residuals are not cached! Saving will call a forward operator!", verbose)
        
        if not restart_path:
            msg = self.get_initial_message()
            self.log_message(msg, verbose)
            # Setting internal vectors (model, search direction, and previous gradient vectors)
            prblm_mdl = problem.get_model()
            self.inv_model = prblm_mdl.clone()
            self.dmodl = prblm_mdl.clone().zero()
            self.grad0 = self.dmodl.clone()
            self.init_obj = None
            self.iter = 0
        else:
            msg = f"Restarting previous solver run from: {restart_path}"
            self.log_message(msg, verbose)

            restart = Restart(restart_path)
            
            restart.read_restart()
            self.iter = restart.retrieve_parameter("iter")
            self.stepper.alpha = restart.retrieve_parameter("alpha")
            self.init_obj = restart.retrieve_parameter("obj_initial")
            self.inv_model = restart.retrieve_vector("solver_mdl")
            self.dmodl = restart.retrieve_vector("solver_dmodl")
            self.grad0 = restart.retrieve_vector("solver_grad0")
            problem.set_model(self.inv_model)
            if not hasattr(self, 'objgradf'):
                problem.set_residual(problem.get_res(problem.get_model()))

        self.prev_model = problem.get_model().clone().zero()

    def log_iteration_info(self, problem, verbose):
        msg = self.iter_msg % (
            str(self.iter).zfill(self.stopper.zfill),
            self.last_obj_value,
            problem.get_gnorm(self.inv_model),
            problem.get_fevals(),
            problem.get_gevals()
        )
        self.log_message(msg, verbose)

    def check_values(self, obj, grad, verbose):
        if isnan(obj) or isnan(grad.norm()):
            self.log_message("Either gradient norm or objective function value NaN!", verbose)
            return False
        if grad.norm() == 0:
            self.log_message("Gradient vanishes identically", verbose)
            return False
        return True

    def save_restart_info(self, alpha):
        self.restart.save_parameter("iter", self.iter)
        self.restart.save_parameter("alpha", alpha)
        self.restart.save_vector("solver_mdl", self.inv_model)
        self.restart.save_vector("solver_dmodl", self.dmodl)
        self.restart.save_vector("solver_grad0", self.grad0)
        # self.restart.save_vector("prblm_res", prblm_res)

    def run(self, problem, verbose=False, restart_path=None):
        self.initialize_solver(problem, verbose, restart_path)

        success = True

        while success:
            success = self.perform_iteration(problem, verbose)
        
        self.save_results(problem, force_save=True, force_write=True)  
        self.log_message(self.get_final_message(), verbose)
        self.restart.clear_restart()

    def log_final_message(self, verbose):
        raise NotImplementedError("Subclasses must implement log_final_message")
    
    def get_initial_message(self):
        raise NotImplementedError("Subclasses must implement get_initial_message")

    def perform_iteration(self, problem, verbose):
        raise NotImplementedError("Subclasses must implement perform_iteration")

    def __del__(self):
        """Default destructor"""
        return

    def setPrefix(self, prefix):
        """Mutator to change prefix and file names for saving inversion results"""
        self.prefix = prefix
        return

    def setDefaults(self, save_obj=False, save_res=False, save_grad=False, save_model=False, prefix=None,
                    iter_buffer_size=None, iter_sampling=1, flush_memory=False):
        """
        Function to set parameters for result saving.

        :param save_obj         : [False] - boolean; Flag to save objective function values into the list self.obj
        :param save_res         : [False] - boolean; Flag to save residual vectors into the list self.res
        :param save_grad        : [False] - boolean; Flag to save gradient vectors into the list self.grad
        :param save_model       : [False] - boolean; Flag to save model vectors into the list self.model.
                                    It will also say the last inverted model vector into self.inv_model
        :param prefix           : [None] - string; Prefix of the files in which requested results will be saved;
                                    If prefix is None, then nothing is going to be saved on disk
        :param iter_buffer_size : [None] - int; Number of steps to save before flushing results to disk
                                    (by default the solver waits until all iterations are done)
        :param iter_sampling    : [1] - int; Sampling of the iteration axis
        :param flush_memory     : [False] - boolean; Whether to keep results into the object lists or clean those
                                    once inversion is completed or results have been written on disk
        """

        # Parameter for saving results
        self.save_obj = save_obj            # Flag to save objective function value
        self.save_res = save_res            # Flag to save residual vector
        self.save_grad = save_grad          # Flag to save gradient vector
        self.save_model = save_model        # Flag to save model vector
        self.flush_memory = flush_memory    # Keep results in RAM or flush memory every time results are written on disk

        # Prefix of the saved files (if provided the results will be written on disk)
        self.prefix = prefix                # Prefix for saving inversion results on disk

        # Iteration axis-sampling parameters
        self.iter_buffer_size = iter_buffer_size  # Number of steps to save before flushing results to disk
        self.iter_sampling = iter_sampling  # Sampling of the iteration axis

        # Lists of the results (list and vector Sets)
        self.obj = np.array([])  # Array for objective function values
        self.obj_terms = np.array([])       # Array for objective function values for each terms
        self.model = list()                 # List for model vectors (to save results in-core)
        self.res = list()                   # List for residual vectors (to save results in-core)
        self.grad = list()                  # List for gradient vectors (to save results in-core)
        self.modelSet = Vec.vectorSet()     # Set for model vectors
        self.resSet = Vec.vectorSet()       # Set for residual vectors
        self.gradSet = Vec.vectorSet()      # Set for gradient vectors
        self.inv_model = None               # Temporary saved inverted model
        self.prev_model = None              # Previously inverted model
        self.overwrite = True               # Flag to overwrite results if first time writing on disk

    def flush_results(self):
        """Flushing internal memory of the saved results"""
        # Lists of the results (list and vector Sets)
        self.obj = np.array([])  # Array for objective function values
        self.obj_terms = np.array([])       # Array for objective function values for each terms
        self.model = list()  # List for model vectors (to save results in-core)
        self.res = list()  # List for residual vectors (to save results in-core)
        self.grad = list()  # List for gradient vectors (to save results in-core)
        self.modelSet = Vec.vectorSet()  # Set for model vectors
        self.resSet = Vec.vectorSet()  # Set for residual vectors
        self.gradSet = Vec.vectorSet()  # Set for gradient vectors
        self.inv_model = None  # Temporary saved inverted model

    def get_restart(self, log_file):
        """
        Function to retrieve restart folder from log file. It enables the user to use restart flag on self.run().
        :param log_file: [None] - string;
        """
        restart_folder = None
        # Obtaining restart folder path
        reg_prog = re.compile("Restart folder: ([^\s]+)")
        if not os.path.isfile(log_file):
            raise OSError("ERROR! No %s file found!" % log_file)
        for line in reversed(open(log_file).readlines()):
            if restart_folder is None:
                find = reg_prog.search(line)
                if find:
                    restart_folder = find.group(1)
        # Setting restart folder if user needs to do so
        if restart_folder is not None:
            self.restart.restart_folder = restart_folder
        else:
            print("WARNING! No restart folder's path was found in %s" % log_file)
        return

    def save_results(self, problem, **kwargs):
        """
        Method to save results
        :param iiter        : Iteration index
        :param problem      : Problem that is being solved
        :param kwargs       :
        - force_save   : [False]; Flag to ignore iteration sampling
        - force_write  : [False]; Force writing on disk if necessary (used to handle last iteration)
        - model : [problem.model] model to be saved and/or written
        - obj : [problem.obj] objective function to be saved
        """
        if not isinstance(problem, pyProblem.Problem):
            raise TypeError("Input variable is not a Problem object")
        force_save = kwargs.get("force_save", False)
        force_write = kwargs.get("force_write", False)
        # Getting a model from arguments if provided (necessary to remove preconditioning)
        mod_save = kwargs.get("model", problem.get_model())
        # Obtaining objective function value
        objf_value = kwargs.get("obj", problem.get_obj(problem.get_model()))
        obj_terms = kwargs.get("obj_terms", problem.obj_terms) if "obj_terms" in dir(problem) else None
        # Save if it is forced to or if the solver hits a sampled iteration number
        # The objective function is saved every iteration if requested
        if self.save_obj:
            self.obj = np.append(self.obj, deepcopy(objf_value))
            # Checking if the objective function has multiple terms
            if obj_terms is not None:
                if len(self.obj_terms) == 0:
                    # First time obj_terms are saved
                    self.obj_terms = np.expand_dims(np.append(self.obj_terms, deepcopy(obj_terms)), axis=0)
                else:
                    self.obj_terms = np.append(self.obj_terms,
                                               np.expand_dims(np.array(deepcopy(obj_terms)), axis=0),
                                               axis=0)
        if self.iter % self.iter_sampling == 0 or force_save:
            if self.save_model:
                self.modelSet.append(mod_save)
                # Storing model vector into a temporary vector
                del self.inv_model  # Deallocating previous saved model
                self.inv_model = mod_save.clone()
            if self.save_res:
                res_vec = problem.get_res(problem.get_model())
                self.resSet.append(res_vec)
            if self.save_grad:
                grad = problem.get_grad(problem.get_model())
                self.gradSet.append(grad)
        # Write on disk if necessary or requested
        self._write_steps(force_write)
        return

    def _write_steps(self, force_write=False):
        """Method to write inversion results on disk if forced to or if buffer is filled"""
        # Save results if buffer size is hit
        save = True if force_write or (self.iter_buffer_size is not None and max(len(self.modelSet.vecSet),
                                                                                 len(self.resSet.vecSet),
                                                                                 len(self.gradSet.vecSet))
                                                                             >= self.iter_buffer_size) \
            else False

        mode = "w" if self.overwrite else "a"

        if save:
            self.overwrite = False  # Written at least once; do not overwrite files
            # Getting current saved results into an in-core list
            if not self.flush_memory:
                self.model += self.modelSet.vecSet
                self.res += self.resSet.vecSet
                self.grad += self.gradSet.vecSet
            # Writing objective function value on disk if requested
            if self.save_obj and self.prefix is not None:
                obj_file = self.prefix + "_obj.H"  # File name in which the objective function is saved
                sepu.write_file(obj_file, self.obj)
                # Writing each term of the objective function
                if len(self.obj_terms) != 0:
                    for iterm in range(self.obj_terms.shape[1]):
                        # File name in which the objective function is saved
                        obj_file = self.prefix + "_obj_comp%s.H" % (iterm + 1)
                        sepu.write_file(obj_file, self.obj_terms[:, iterm])
            # Writing current inverted model and model vectors on disk if requested
            if self.save_model and self.prefix is not None:
                inv_mod_file = self.prefix + "_inv_mod.H"  # File name in which the current inverted model is saved
                model_file = self.prefix + "_model.H"  # File name in which the model vector is saved
                self.modelSet.writeSet(model_file, mode=mode)
                self.inv_model.writeVec(inv_mod_file, mode="w") # Writing inverted model file
            # Writing gradient vectors on disk if requested
            if self.save_grad and self.prefix is not None:
                grad_file = self.prefix + "_gradient.H"  # File name in which the gradient vector is saved
                self.gradSet.writeSet(grad_file, mode=mode)
            # Writing residual vectors on disk if requested
            if self.save_res and self.prefix is not None:
                res_file = self.prefix + "_residual.H"  # File name in which the residual vector is saved
                self.resSet.writeSet(res_file, mode=mode)


class Restart:
    """Class for restarting a solver run"""

    def __init__(self, path: str=None):
        """Restart constructor"""
        self.par_dict = dict()
        self.vec_dict = dict()
        if path:
            restart_folder = path
        else:
            # Restart folder in case it is necessary to write restart
            now = datetime.datetime.now()
            restart_folder = sepu.datapath + "restart_" + now.isoformat() + "/"
            restart_folder = restart_folder.replace(":", "-")
        self.restart_folder = restart_folder
        # Calling write_restart when python session dies
        atexit.register(self.write_restart)

    def save_vector(self, vec_name, vector_in):
        """Method to save vector for restarting"""
        # Deleting the vector if present in the dictionary
        element = self.vec_dict.pop(vec_name, None)
        if element:
            del element
        self.vec_dict.update({vec_name: vector_in.clone()})

    def retrieve_vector(self, vec_name):
        """Method to retrieve a vector from restart object"""
        return self.vec_dict[vec_name]

    def save_parameter(self, par_name, parameter_in):
        """Method to save vector for restarting"""
        self.par_dict.update({par_name: parameter_in})
        return

    def retrieve_parameter(self, par_name):
        """Method to retrieve a parameter from restart object"""
        return self.par_dict[par_name]

    def write_restart(self):
        """Restart destructor: it will write vectors on disk if the solver breaks"""
        if bool(self.par_dict) or bool(self.vec_dict):
            # Creating restarting directory
            mkdir(self.restart_folder)
            with open(self.restart_folder + 'restart_obj.pkl', 'wb') as out_file:
                pickle.dump(self, out_file, pickle.HIGHEST_PROTOCOL)
            # Checking if a vectorOC was in the restart and preventing the removal of the vector file
            for vec_name, vec in self.vec_dict.items():
                if isinstance(vec, Vec.vectorOC):
                    vec.remove_file = False

    def read_restart(self):
        """Method to read restart object from saved folder"""
        if os.path.isdir(self.restart_folder):
            with open(self.restart_folder + 'restart_obj.pkl', 'rb') as in_file:
                restart = pickle.load(in_file)
            self.par_dict = restart.par_dict
            self.vec_dict = restart.vec_dict
            # Checking if a vectorOC was in the restart and setting the removal of the vector file
            for vec_name, vec in self.vec_dict.items():
                if isinstance(vec, Vec.vectorOC):
                    vec.remove_file = True
            # Removing previous restart and deleting read object
            restart.clear_restart()
            del restart

    def clear_restart(self):
        """Method to clear the restart"""
        self.par_dict = dict()
        self.vec_dict = dict()
        # Removing restart folder if existing
        if os.path.isdir(self.restart_folder):
            # Removing folder
            rmtree(self.restart_folder)
