# Module containing generic Solver and Restart definitions

from sys import path
path.insert(0, '.')
import pyProblem
import pyVector as Vec
import pySolver
import pyStepper
import atexit
import os
# Functions and modules necessary for writing on disk
import pickle
import re
import numpy as np
import pyProxOperator 
import sep_util as sepu
from sys_util import mkdir

from shutil import rmtree
from copy import deepcopy
import datetime


class AugLagrangianSolver:
    """Solver parent object"""

    # Default class methods/functions
    def __init__(self, inner_solver, rho, p_rho, constraint_tol=0.25, m_rho=1, outer=1, save_dual=False):
        """Default class constructor for Solver"""
        self.p_solver = inner_solver
        self.rho = rho
        self.p_rho = p_rho
        self.m_rho = m_rho
        self.c_tol = constraint_tol
        self.outer = outer
        self.save_dual = save_dual
        return

    def run(self, problem, verbose=False, restart=False):
        c_ratio = 0
        dual_count = 0
        inner_count = 0
        start_iter = 0 
        for it in range(self.outer):
            problem.set_rho(self.rho)
            while True:
                problem.setDefaults()
                # temporary solution for resetting the stepper
                problem.stepper = pyStepper.ParabolicStep()

                if verbose:
                    msg = 90 * "*" + "\n"
                    msg += "\t\t\tAUGMENTED LAGRANGIAN (METHOD OF MULTIPLIERS)\n"
                    msg += "\t Rho value used: %.5f\n" % self.rho
                    msg += "\t Inner problem solved %d times\n" % inner_count
                    msg += "\t Dual variable updated %d times\n" % dual_count
                    msg += 90 * "*" + "\n"
                    print(msg)
                    self.p_solver.logger.addToLog(msg)
                
                # Solver inner problem
                self.p_solver.run(problem,verbose,restart)
                inner_count += 1

                # c_ratio = ||mod_res_final||/||max(mod_res)||
                max = np.amax(self.p_solver.obj_terms[:,1])
                if max > 0:
                    c_ratio = self.p_solver.obj_terms[-1,1] / max
                else:
                    c_ratio = 0
                
                if verbose:
                        msg = "\t\t\tCurrent decrease in the constraint-residual norm: %.5f\n" % c_ratio

                if c_ratio <= self.c_tol:
                    if verbose:
                        msg += "\t\t\tUpdating dual variable and decreasing rho = %.5f\n" % self.rho
                        print(msg)
                        self.p_solver.logger.addToLog(msg)

                    # Update dual variable
                    problem.update_dual()
                    dual_count += 1
                    if self.save_dual and self.p_solver.prefix is not None:
                        dual_file = self.p_solver.prefix + "_dual.H"  
                        problem.dual.writeVec(dual_file, mode="a")
                    
                    # decrease rho
                    self.rho *= self.m_rho
                    break
                else:
                    if verbose:
                        msg += "\t\t\tKeeping the dual variable and increasing rho\n"
                        print(msg)
                        self.p_solver.logger.addToLog(msg)
                    # increase rho
                    self.rho *= self.p_rho
                    break
                
class ADMMsolver(pySolver.Solver):

    """
        Alternating Direction Method of Multipliers (ADMM) solver
        using proximal operators formulation with scaled dual variables
        for solving constrained problems of the form:
        
            min f(x) + epsilon * g(z)
            s.t. x - z = 0
        
        where f and g could be differentiable or non-differentiable functions

        using iterations in the form:
            x^{k+1} = prox_{f}(z^k - u^k)
            z^{k+1} = prox_{g}(x^{k+1} + u^k)
            u^{k+1} = u^k + x^{k+1} - z^{k+1}
        
        where u is the scaled dual variable
    """

    def __init__(self, proxf, proxg, outer, rho=1, mu=10, tdecr=2, tincr=2, logger=None, save_dual=False, model=None):
        self.proxf = proxf
        self.proxg = proxg
        if model is None:
            model = proxf.problem.model

        self.logger = logger
        # tau is equivalent to inverse of rho in Augmented Lagrangian
        self.rho = rho
        self.mu = mu
        self.tdecr = tdecr
        self.tincr = tincr
        # force the same logger for inner solvers
        if hasattr(proxf, 'solver'):
            self.proxf.solver.logger = self.logger
        if hasattr(proxg, 'solver'):
            self.proxg.solver.logger = self.logger
        self.outer = outer
        self.save_dual = save_dual
        
        self.x = model.clone().zero()
        self.z = model.clone().zero()
        self.u = model.clone().zero()

    def get_initial_message(self):
        msg = f"\t\t\tADMM SOLVER log file\n"
        return msg
    
    def get_final_message(self):
        msg = 90 * "#" + "\n"
        msg += "\t\t\tADMM%s SOLVER log file end\n"
        msg += 90 * "#" + "\n"
        return msg
    
    def get_iteration_message(self, it):
        msg = 90 * "#" + "\n"
        msg += "\t\t\tADMM iteration %d\n" % it
        msg += 90 * "#" + "\n"
        return msg
    
        
    def run(self, verbose=False):
        self.log_message(self.get_initial_message(), verbose)
        for it in range(self.outer):
            self.log_message(self.get_iteration_message(it), verbose)
            # x_k+1 = proxf(z_k - u_k)
            self.proxf.prox(self.z - self.u, self.x, 1/self.rho)
            # z_k+1 = proxf(x_k+1 + u_k)
            self.proxg.prox(self.x + self.u, self.z, 1/self.rho)
            # u_k+1 = u_k + x_k+1 - z_k+1
            self.u.scaleAdd(self.x - self.z, 1, 1)
            # adjust rho
        self.log_message(self.get_final_message(), verbose)


            
            
            
            

            
