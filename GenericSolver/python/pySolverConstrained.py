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

import sep_util as sepu
from sys_util import mkdir

from shutil import rmtree
from copy import deepcopy
import datetime


class AugLagrangianSolver:
    """Solver parent object"""

    # Default class methods/functions
    def __init__(self, inner_solver, rho=[0]):
        """Default class constructor for Solver"""
        self.p_solver = inner_solver
        self.rho = rho
        return

    def run(self, problem, verbose=False, restart=False):
        for it in range(len(self.rho)):
            problem.set_rho(self.rho[it])
            self.p_solver.run(problem,verbose,restart)
            # Update dual variable
            problem.update_dual()