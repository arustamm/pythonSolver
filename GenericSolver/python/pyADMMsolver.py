# Module containing the definition of inverse problems where the ADMM method is used
import pyOperator
import pyVector
from pyLinearSolver import LCGsolver, LSQRsolver
from pyProblem import Problem, ProblemL1Lasso, ProblemL2LinearReg, ProblemL2Linear
from pySolver import Solver
from pySparseSolver import ISTAsolver
from pyStopper import BasicStopper
from math import isnan, sqrt


# TODO make it accept L2 reg problems
class ProblemLinearReg(Problem):
    def __init__(self, model, data, op, epsL1=None, regsL1=None, epsL2=None, regsL2=None, dataregsL2=None,
                 minBound=None, maxBound=None, boundProj=None):
        """
        Linear Problem with both L1 and L2 regularizers:

        .. math ::
            1 / 2 |Op m - d|_2^2 +
            \sum_i epsL2_i |R2_i m - dr|_2^2 +
            \sum_i epsL1_i |R1_i m|_1

        :param model        : vector; initial model
        :param data         : vector; data
        :param op           : LinearOperator; data fidelity operator
        :param epsL1        : list; weights of L1 regularizers [None]
        :param regsL1       : list; L1 regularizers of class LinearOperator [None]
        :param epsL2        : list; weights of L2 regularizers [None]
        :param regsL2       : list; L2 regularizers of class LinearOperator [None]
        :param dataregsL2   : vector; prior model for L2 regularization term [None]
        :param minBound     : vector; minimum value bounds
        :param maxBound     : vector; maximum value bounds
        :param boundProj    : Bounds; object with a method "apply(x)" to project x onto some convex set
        """
        super(ProblemLinearReg, self).__init__(minBound, maxBound, boundProj)
        self.model = model.clone()
        self.dmodel = model.clone().zero()
        self.grad = self.dmodel.clone()
        self.data = data
        self.op = op

        self.minBound = minBound
        self.maxBound = maxBound
        self.boundProj = boundProj

        # L1 Regularizations
        self.regL1_op = None if regsL1 is None else pyOperator.Vstack(regsL1)
        self.nregsL1 = self.regL1_op.n if self.regL1_op is not None else 0
        self.epsL1 = epsL1 if epsL1 is not None else []
        if type(self.epsL1) in [int, float]:
            self.epsL1 = [self.epsL1]
        assert len(self.epsL1) == self.nregsL1, 'The number of L1 regs and related weights mismatch!'
        
        # L2 Regularizations
        self.regL2_op = None if regsL2 is None else pyOperator.Vstack(regsL2)
        self.nregsL2 = self.regL2_op.n if self.regL2_op is not None else 0
        self.epsL2 = epsL2 if epsL2 is not None else []
        if type(self.epsL2) in [int, float]:
            self.epsL2 = [self.epsL2]
        assert len(self.epsL2) == self.nregsL2, 'The number of L2 regs and related weights mismatch!'
        
        if self.regL2_op is not None:
            self.dataregsL2 = dataregsL2 if dataregsL2 is not None else self.regL2_op.range.clone().zero()
        else:
            self.dataregsL2 = None

        # At this point we should have:
        # - a list of L1 regularizers;
        # - a list of L1 weights (with same length of previous);
        # - a list of L2 regularizers (even empty is ok);
        # - a list of L2 weights (with same length of previous);
        # - a list of L2 dataregs (with same length of previous);

        # Last settings
        self.obj_terms = [None] * (1 + self.nregsL2 + self.nregsL1)
        self.linear = True
        # store the "residuals" (for computing the objective function)
        self.res_data = self.op.range.clone().zero()
        self.res_regsL2 = self.regL2_op.range.clone().zero() if self.nregsL2 != 0 else None
        self.res_regsL1 = self.regL1_op.range.clone().zero() if self.nregsL1 != 0 else None
        # this last superVector is instantiated with pointers to res_data and res_regs!
        self.res = pyVector.superVector(self.res_data, self.res_regsL2, self.res_regsL1)
        
        # flags for avoiding extra computations
        self.res_data_already_computed = False
        self.res_regsL1_already_computed = False
        self.res_regsL2_already_computed = False
        
        # TODO add compatibility with L2 problems and Lasso

    def __del__(self):
        """Default destructor"""
        return
    
    def objf(self, res):
        """
        Compute objective function based on the residual (super)vector
        
        .. math ::
            1 / 2 |Op m - d|_2^2 +
            \sum_i epsL2_i |R2_i m - dr|_2^2 +
            \sum_i epsL1_i |R1_i m|_1
        
        """
        res_data = res.vecs[0]
        res_regsL2 = res.vecs[1] if self.res_regsL2 is not None else None
        if self.res_regsL1 is not None:
            res_regsL1 = res.vecs[2] if self.res_regsL2 is not None else res.vecs[1]
        else:
            res_regsL1 = None
        
        self.obj_terms[0] = .5 * res_data.norm(2)**2  # data fidelity
        
        if res_regsL2 is not None:
            for idx in range(self.nregsL2):
                self.obj_terms[1 + idx] = self.epsL2[idx] * res_regsL2.vecs[idx].norm(2)**2
        if res_regsL1 is not None:
            for idx in range(self.nregsL1):
                self.obj_terms[1 + self.nregsL2 + idx] = self.epsL1[idx] * res_regsL1.vecs[idx].norm(1)
        
        return sum(self.obj_terms)
    
    def resf(self, model):
        """Compute residuals from current model"""
        
        # compute data residual: Op * m - d
        if model.norm() != 0:
            self.op.forward(False, model, self.res_data)  # rd = Op * m
        else:
            self.res_data.zero()
        self.res_data.scaleAdd(self.data, 1., -1.)  # rd = rd - d
        
        # compute L2 reg residuals
        if self.res_regsL2 is not None:
            if model.norm() != 0:
                self.regL2_op.forward(False, model, self.res_regsL2)
            else:
                self.res_regsL2.zero()
            if self.dataregsL2 is not None and self.dataregsL2.norm() != 0.:
                self.res_regsL2.scaleAdd(self.dataregsL2, 1., -1.)
        
        # compute L1 reg residuals
        if self.res_regsL1 is not None:
            if model.norm() != 0. and self.regL1_op is not None:
                self.regL1_op.forward(False, model, self.res_regsL1)
            else:
                self.res_regsL1.zero()
        
        return self.res
    
    
def _shrinkage(x, thresh, eps=1e-10):
    """
    Shrinkage function Gamma
        y = x / (|x| + eps) * maximum(|x| - thresh, 0)
    """
    y = x.clone()
    y / x.clone().abs().addbias(eps)
    return y * x.clone().abs().addbias([-t for t in thresh]).maximum(0.)


class SplitBregmanSolver(Solver):
    """Split-Bregman solver for L1 and L2 regularized problems"""

    # Default class methods/functions
    def __init__(self, stopper, logger=None, niter_inner=3, niter_solver=5, breg_weight=1., linear_solver='CG', use_prev_sol=False):
        """
        Constructor for Split-Bregman Solver
        :param stopper          : stopper object
        :param logger           : logger object
        :param niter_inner      : int; number of iterations for the shrinkage loop [default 3]
        :param niter_solver     : int; number of iterations for the internal linear solver [default 5]
        :param breg_weight      : float; coefficient for the Bregman update b += beta * (R*x - d) [1.]
        :param linear_solver    : str; linear solver to be used [CG, SD, LSQR]
        :param use_prev_sol     : bool; linear solver restarts from previous solution [False]
        """
        # Calling parent construction
        super(SplitBregmanSolver, self).__init__()
        # Defining stopper object
        self.stopper = stopper
        # Logger object to write on log file
        self.logger = logger
        # Overwriting logger of the Stopper object
        self.stopper.logger = self.logger

        self.niter_inner = niter_inner  # number of iterations for the shrinkage
        self.niter_solver = niter_solver  # number of iterations for the internal problem
        self.use_prev_sol = use_prev_sol
        if breg_weight > 1.:
            raise ValueError("ERROR! Bregman update weight has to be <= 1")
        self.breg_weight = float(breg_weight)
        
        if linear_solver == 'CG':
            self.linear_solver = LCGsolver(BasicStopper(niter=self.niter_solver), steepest=False, logger=self.logger)
        elif linear_solver == 'SD':
            self.linear_solver = LCGsolver(BasicStopper(niter=self.niter_solver), steepest=True, logger=self.logger)
        elif linear_solver == 'LSQR':
            self.linear_solver = LSQRsolver(BasicStopper(niter=self.niter_solver), logger=self.logger)
        else:
            raise ValueError('ERROR! Solver has to be CG, SD or LSQR')
        self.linear_solver.setDefaults(iter_sampling=1, flush_memory=True)
        
        # print formatting
        self.iter_msg = "iter = %s, obj = %.5e, df_obj = %.2e, reg_obj = %.2e, resnorm = %.2e"
        
    def __del__(self):
        print('Destructor called, Split-Bregman solver deleted')
        
    def run(self, problem, verbose=False, inner_verbose=False, restart=False, initial_guess=None):
        """Running SplitBregman solver"""
        assert type(problem) == ProblemLinearReg, 'problem has to be a ProblemLinearReg'
        if problem.regL1_op is None:
            raise ValueError("ERROR! Problem has to include at least one L1 Regularizer")
        
        verbose = True if inner_verbose else verbose
        self.create_msg = verbose or self.logger
        
        # reset stopper before running the inversion
        self.stopper.reset()

        # initialize all the vectors and operators for Split-Bregman
        breg_b = problem.regL1_op.range.clone().zero()
        breg_a = breg_b.clone()
        RL1x = breg_b.clone()  # store RegL1 * solution
        
        sb_mdl = problem.model.clone().zero() if initial_guess is None else initial_guess.clone()
        if not problem.op.domain.checkSame(sb_mdl):
            raise ValueError("ERROR! The initial guess and the operator domain mismatch.")
        
        # TODO linear_solver accepts only one regularizer and one epsilon:
        #  we must convert reg_op to a scaled version and epsilon to 1.
        regL2_op_scaled_list = [sqrt(problem.epsL2[i]/2)/sqrt(1/2) * problem.regL2_op.ops[i] for i in range(problem.nregsL2)]
        regL1_op_scaled_list = [sqrt(problem.epsL1[i]/2)/sqrt(1/2) * problem.regL1_op.ops[i] for i in range(problem.nregsL1)]
        reg_op = pyOperator.Vstack(pyOperator.Vstack(regL2_op_scaled_list) if len(regL2_op_scaled_list) != 0 else None,
                                   pyOperator.Vstack(regL1_op_scaled_list) if len(regL1_op_scaled_list) != 0 else None)
        
        if restart:
            self.restart.read_restart()
            outer_iter = self.restart.retrieve_parameter("iter")
            initial_obj_value = self.restart.retrieve_parameter("obj_initial")
            sb_mdl = self.restart.retrieve_vector("sb_mdl")
            if self.create_msg:
                msg = "Restarting previous solver run from: %s" % self.restart.restart_folder
                if verbose:
                    print(msg)
                if self.logger:
                    self.logger.addToLog(msg)

        else:
            outer_iter = 0
            if self.create_msg:
                msg = 90 * '#' + '\n'
                msg += "\t\t\t\t\tSPLIT-BREGMAN ALGORITHM log file\n\n"
                msg += "\tRestart folder: %s\n" % self.restart.restart_folder
                msg += "\tModeling Operator:\t\t%s\n" % problem.op
                if problem.nregsL2 != 0:
                    msg += "\tL2 Regularizer ops:\t\t" + ", ".join(["%s" % op for op in problem.regL2_op.ops]) + "\n"
                    msg += "\tL2 Regularizer weights:\t" + ", ".join(["{:.2e}".format(e) for e in problem.epsL2]) + "\n"
                if problem.nregsL1 != 0:
                    msg += "\tL1 Regularizer ops:\t\t" + ", ".join(["%s" % op for op in problem.regL1_op.ops]) + "\n"
                    msg += "\tL1 Regularizer weights:\t" + ", ".join(["{:.2e}".format(e) for e in problem.epsL1]) + "\n"
                msg += "\tBregman update weight:\t%.2e\n" % self.breg_weight
                msg += 90 * '#' + '\n'
                if verbose:
                    print(msg.replace(" log file", ""))
                if self.logger:
                    self.logger.addToLog(msg)
        
        # Main iteration loop
        while True:
            obj0 = problem.get_obj(sb_mdl)
            
            if outer_iter == 0:
                initial_obj_value = obj0
                self.restart.save_parameter("obj_initial", initial_obj_value)
                if self.create_msg:
                    msg = self.iter_msg % (str(outer_iter).zfill(self.stopper.zfill),
                                           obj0,
                                           problem.obj_terms[0],
                                           obj0 - problem.obj_terms[0],
                                           problem.get_rnorm(sb_mdl))
                    if verbose:
                        print(msg)
                    if self.logger:
                        self.logger.addToLog("\n" + msg)
                
                if isnan(obj0):
                    raise ValueError("Objective function values NaN!")
            
            if obj0 == 0:
                print("Objective function is 0!")
                break
            
            self.save_results(outer_iter, problem, force_save=False)
            
            for iter_inner in range(self.niter_inner):

                # if self.create_msg:
                #     msg = "\t\tstarting inner iter %d with a = %.2e, b = %.2e"\
                #           % (iter_inner, breg_a.norm(), breg_b.norm())
                #     if verbose:
                #         print(msg)
                #     if self.logger:
                #         self.logger.addToLog("\n" + msg)
                
                # solve inner problem
                prior = pyVector.superVector(problem.dataregsL2, breg_a.clone().scaleAdd(breg_b, 1., -1.))

                linear_problem = ProblemL2LinearReg(
                    model=sb_mdl.clone().zero() if not self.use_prev_sol else sb_mdl.clone(),
                    data=problem.data,
                    op=problem.op,
                    epsilon=1.,
                    reg_op=reg_op,
                    prior_model=prior,
                    minBound=problem.minBound, maxBound=problem.maxBound, boundProj=problem.boundProj
                )
                if outer_iter == 0 and initial_guess is not None:
                    linear_problem.model = initial_guess.clone()
                    
                # self.linear_solver.setDefaults()
                self.linear_solver.run(linear_problem, verbose=inner_verbose)

                # sb_mdl = linear_problem.model.clone()
                sb_mdl.copy(linear_problem.model)
                
                # compute RL1*x
                if problem.nregsL1 != 0:
                    problem.regL1_op.forward(False, sb_mdl, RL1x)
                
                # update breg_a
                if problem.nregsL1 != 0:
                    breg_a.copy(_shrinkage(RL1x.clone() + breg_b, thresh=problem.epsL1))
                
                # if self.create_msg:
                #     msg = "\t\tfinished inner iter %d with sb_mdl = %.2e, RL1x = %.2e"\
                #           % (iter_inner, sb_mdl.norm(), RL1x.norm())
                #     if verbose:
                #         print(msg)
                #     if self.logger:
                #         self.logger.addToLog("\n" + msg)
                
            # update breg_b
            if problem.nregsL1 != 0:
                breg_b.scaleAdd(RL1x.clone() - breg_a, 1., self.breg_weight)

            outer_iter += 1
            # check objective function
            # problem.res_regsL1 = RL1x.clone()
            # problem.res_regsL1_already_computed = True
            # problem.res_data_already_computed = False
            obj1 = problem.get_obj(sb_mdl)
            # if obj1 >= obj0:  # TODO check theory for monotonic convergence
            #     if self.create_msg:
            #         msg = "Objective function didn't reduce, will terminate solver:\n\t"\
            #               "obj_new = %.2e\tobj_cur = %.2e" % (obj1, obj0)
            #         if verbose:
            #             print(msg)
            #         if self.logger:
            #             self.logger.addToLog(msg)
            #     break
            
            # iteration info
            if self.create_msg:
                msg = self.iter_msg % (str(outer_iter).zfill(self.stopper.zfill),
                                       obj1,
                                       problem.obj_terms[0],
                                       obj1 - problem.obj_terms[0],
                                       problem.get_rnorm(sb_mdl))
                if verbose:
                    print(msg)
                if self.logger:
                    self.logger.addToLog("\n" + msg)
            
            # saving in case of restart
            self.restart.save_parameter("iter", outer_iter)
            self.restart.save_vector("sb_mdl", sb_mdl)
            
            if self.stopper.run(problem, outer_iter, initial_obj_value, verbose):
                break
            
        # writing last inverted model
        self.save_results(outer_iter, problem, force_save=True, force_write=True)
        
        # ending message and log file
        if self.create_msg:
            msg = 90 * '#' + '\n'
            msg += "\t\t\t\t\tSPLIT-BREGMAN ALGORITHM log file end\n"
            msg += 90 * '#'
            if verbose:
                print(msg.replace(" log file", ""))
            if self.logger:
                self.logger.addToLog("\n" + msg)
        
        # Clear restart object
        self.restart.clear_restart()
        

class ADMMsolver(Solver):
    """Alternate Directions of Multipliers Method (ADMM)"""
    
    # Default class methods/functions
    def __init__(self, stopper, logger=None, niter_linear=5, niter_lasso=15, rho=None, auto_rho=True, mu=10., tau=2., use_prev_sol=False, linear_solver='CG'):
        """
        Constructor for ADMM Solver
        .. math ::
            1/2 |Op x - d|_2^2 + \sum_i epsL2_i |R2_i x - dr|_2^2 + \gamma |y|_1
                subject to Ax + By = c
        Note: for now B=-I, c=0 (i.e., y=Ax)
        :param stopper          : stopper object
        :param logger           : logger object
        :param niter_linear     : int; number of iterations for solving the linear problem [5]
        :param niter_lasso      : int; number of iterations for solving the lasso problem [5]
        :param rho              : float; penalty parameter rho (if None it is initialized as 2*gamma+.1)
        :param auto_rho         : bool; update rho automatically
        :param mu               : float; norm ratio between residuals for updating rho [10]
        :param tau              : float; scaling factor for updating rho [2]
        :param use_prev_sol     : bool; linear solver uses previous solution [False]
        :param linear_solver    : str; linear solver to be used [CG, SD, LSQR]
        """
        # Calling parent construction
        super(ADMMsolver, self).__init__()

        self.stopper = stopper
        self.logger = logger
        self.stopper.logger = self.logger
        self.niter_linear = niter_linear
        self.niter_lasso = niter_lasso
        
        if linear_solver == 'CG':
            self.solver_linear = LCGsolver(BasicStopper(niter=self.niter_linear), steepest=False, logger=self.logger)
        elif linear_solver == 'SD':
            self.solver_linear = LCGsolver(BasicStopper(niter=self.niter_linear), steepest=True, logger=self.logger)
        elif linear_solver == 'LSQR':
            self.solver_linear = LSQRsolver(BasicStopper(niter=self.niter_linear), logger=self.logger)
        else:
            raise ValueError('ERROR! Solver has to be CG, SD or LSQR')
        self.solver_linear.setDefaults(iter_sampling=1, flush_memory=True)
        
        self.solver_lasso = ISTAsolver(BasicStopper(niter=self.niter_lasso), fast=True, logger=self.logger)
        self.solver_lasso.setDefaults(iter_sampling=1, flush_memory=True)
        self.use_prev_sol = use_prev_sol

        self.rho = rho          # ADMM penalty parameter
        self.mu = mu
        self.tau = tau
        self.auto_rho = auto_rho
        self.primal = None      # aka r (in the complete formulation is A x + B z - c)
        self.dual = None        # aka s (in the complete formulation is rho A.H B r)

        # print formatting
        self.iter_msg = "iter = %s, obj = %.5e, df_obj = %.2e, reg_obj = %.2e, resnorm = %.2e"

    def __del__(self):
        print('Destructor called, ADMM deleted')
 
    def update_rho(self):
        """update penalty parameter rho as suggested in boyd2010distributed (3.13)"""
        if self.primal.norm(2) > self.mu * self.dual.norm(2):
            self.rho = self.rho * self.tau
        elif self.dual.norm(2) > self.mu * self.primal.norm(2):
            self.rho = self.rho / self.tau
        else:
            pass

    def init_rho(self, gamma):
        self.rho = 2*gamma + .1

    def run(self, problem, verbose=False, inner_verbose=False, restart=False, initial_guess=None):

        assert type(problem) == ProblemLinearReg, 'problem has to be a ProblemLinearReg'
        if problem.nregsL1 == 0:
            raise ValueError('ERROR! Provide at least one L1 regularizer!')

        self.create_msg = verbose or self.logger
        
        # I want to set dfw=1, so:
        gamma = max(problem.epsL1)

        # A is the Vstack of the L1 reg operators, scaled by their respective eps
        # we divide by gamma as gamma becomes the lambda value for the FISTA problem
        A = problem.regL1_op * [e / gamma for e in problem.epsL1]
        
        # initialize all others variables
        if self.rho is None:
            self.init_rho(gamma)
        
        admm_mdl = problem.model.clone().zero() if initial_guess is None else initial_guess.clone()
        y = A.range.clone().zero()
        u = y.clone()
        self.dual = A.domain.clone().zero()
        
        if restart:
            self.restart.read_restart()
            outer_iter = self.restart.retrieve_parameter("iter")
            initial_obj_value = self.restart.retrieve_parameter("obj_initial")
            admm_mdl = self.restart.retrieve_vector("admm_mdl")
            if self.create_msg:
                msg = "Restarting previous solver run from: %s" % self.restart.restart_folder
                if verbose:
                    print(msg)
                if self.logger:
                    self.logger.addToLog(msg)
        else:
            outer_iter = 0
            if self.create_msg:
                msg = 90 * '#' + '\n'
                msg += "\t\t\t\t\tADMM ALGORITHM log file\n\n"
                msg += "\tRestart folder: %s\n" % self.restart.restart_folder
                msg += "\tModeling Operator:\t\t%s\n" % problem.op
                if problem.nregsL2 != 0:
                    msg += "\tL2 Regularizer ops:\t\t" + ", ".join(["%s" % op for op in problem.regL2_op.ops]) + "\n"
                    msg += "\tL2 Regularizer weights:\t" + ", ".join(["{:.2e}".format(e) for e in problem.epsL2]) + "\n"
                msg += "\tL1 Regularizer ops:\t\t" + ", ".join(["%s" % op for op in problem.regL1_op.ops]) + "\n"
                msg += "\tL1 Regularizer weights:\t" + ", ".join(["{:.2e}".format(e) for e in problem.epsL1]) + "\n"
                msg += "\tPenalty parameter:\t\t%.2e\n" % self.rho
                msg += 90 * '#' + '\n'
                if verbose:
                    print(msg.replace(" log file", ""))
                if self.logger:
                    self.logger.addToLog(msg)

        # Main iteration loop
        while True:
            obj0 = problem.get_obj(admm_mdl)

            if outer_iter == 0:
                initial_obj_value = obj0
                self.restart.save_parameter("obj_initial", initial_obj_value)
                if self.create_msg:
                    msg = self.iter_msg % (str(outer_iter).zfill(self.stopper.zfill),
                                           obj0,
                                           problem.obj_terms[0],
                                           obj0 - problem.obj_terms[0],
                                           problem.get_rnorm(admm_mdl))
                    if verbose:
                        print(msg)
                    if self.logger:
                        self.logger.addToLog("\n" + msg)
    
                if isnan(obj0):
                    raise ValueError("Objective function values NaN!")

            if obj0 == 0:
                print("Objective function is 0!")
                break

            self.save_results(outer_iter, problem, force_save=False)
            
            # 1) update x
            # Linear Problem:       1/2 | Op x - d| + epsL2   | R2 x -  dr  |
            #                                         rho/2   | A  x - (y-u)|
            regL2_op_scaled_list = [problem.epsL2[i] * problem.regL2_op.ops[i] for i in range(problem.nregsL2)]
            regA_op_scaled_list = [self.rho / 2 * A.ops[i] for i in range(A.n)]
            reg_op = pyOperator.Vstack(
                pyOperator.Vstack(regL2_op_scaled_list) if len(regL2_op_scaled_list) != 0 else None,
                pyOperator.Vstack(regA_op_scaled_list) if len(regA_op_scaled_list) != 0 else None,
            )
            prior = pyVector.superVector(problem.dataregsL2, y.clone().scaleAdd(u, 1., -1.))
            linear_problem = ProblemL2LinearReg(
                model=admm_mdl.clone().zero() if not self.use_prev_sol else admm_mdl.clone(),
                data=problem.data,
                op=problem.op,
                reg_op=reg_op,  # pyOperator.Vstack(problem.regL2_op, A),
                epsilon=1.,     # problem.epsL2 + [self.rho / 2] * A.n,
                prior_model=prior,
                minBound=problem.minBound, maxBound=problem.maxBound, boundProj=problem.boundProj
            )
            if outer_iter == 0 and initial_guess is not None:
                linear_problem.model = initial_guess.clone()
            
            self.solver_linear.run(linear_problem, verbose=inner_verbose)
            
            admm_mdl = linear_problem.model.clone()
            
            # 2) update y
            # lasso problem: rho/2 | A x - z + u|_2^2 + gamma | y |_1
            # this means to solve: 1/2 | I y - (Ax + u)| + gamma/rho | y |_1
            Ax = A * admm_mdl
            lasso_problem = ProblemL1Lasso(     # TODO it stops at the second iteration
                model=y.clone().zero(),
                data=Ax.clone() + u,
                op=pyOperator.IdentityOp(y),
                op_norm=1.,
                lambda_value=gamma / self.rho,
                minBound=problem.minBound, maxBound=problem.maxBound, boundProj=problem.boundProj
            )
            self.solver_lasso.setDefaults()
            self.solver_lasso.run(lasso_problem, verbose=inner_verbose)
            y = lasso_problem.model.clone()
            
            # 3) update penalty parameter and scaled dual variable
            self.primal = Ax.clone() - y
            
            A.adjoint(False, self.dual, self.primal)
            self.dual.scale(-self.rho)
            self.update_rho()
            u.__add__(self.primal).scale(1/self.rho)  # u = (u + r)/rho

            outer_iter += 1
            # check objective function
            obj1 = problem.get_obj(admm_mdl)
            # if obj1 >= obj0:
            #     msg = "Objective function didn't reduce, will terminate solver:\n\t" \
            #           "obj_new = %.2e\tobj_cur = %.2e" % (obj1, obj0)
            #     if verbose:
            #         print(msg)
            #     if self.logger:
            #         self.logger.addToLog(msg)
            #     break
            
            # iteration info
            if self.create_msg:
                msg = self.iter_msg % (str(outer_iter).zfill(self.stopper.zfill),
                                       obj1,
                                       problem.obj_terms[0],
                                       obj1 - problem.obj_terms[0],
                                       problem.get_rnorm(admm_mdl))
                if verbose:
                    print(msg)
                if self.logger:
                    self.logger.addToLog("\n" + msg)

            # saving in case of restart
            self.restart.save_parameter("iter", outer_iter)
            self.restart.save_vector("admm_mdl", admm_mdl)

            if self.stopper.run(problem, outer_iter, initial_obj_value, verbose):
                break

        # writing last inverted model
        self.save_results(outer_iter, problem, model=None, force_save=False, force_write=False)
    
        # ending message and log file
        if self.create_msg:
            msg = 90 * '#' + '\n'
            msg += "\t\t\t\t\tADMM ALGORITHM log file end\n"
            msg += 90 * '#'
            if verbose:
                print(msg.replace(" log file", ""))
            if self.logger:
                self.logger.addToLog("\n" + msg)

        # Clear restart object
        self.restart.clear_restart()


def main():
    from sys import path
    path.insert(0, '.')
    import numpy as np
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    import pyNpOperator
    
    PLOT = True
    EXAMPLE = 'noisy'  # must be noisy, gaussian or medical
    
    if EXAMPLE == 'noisy':
        # data examples
        np.random.seed(1)
        nx = 101
        x = pyVector.vectorIC((nx,)).zero()
        x.getNdArray()[:nx // 2] = 10
        x.getNdArray()[nx // 2:3 * nx // 4] = -5
    
        Iop = pyOperator.IdentityOp(x)
        TV = pyNpOperator.FirstDerivative(x)
        L = pyNpOperator.SecondDerivative(x)
        
        n = x.clone()
        n.getNdArray()[:] = np.random.normal(0,  1, nx)
        y = Iop * (x.clone() + n)
        
        derivative = TV * x
    
        # if PLOT:
        #     plt.figure(figsize=(5, 4))
        #     plt.plot(x.getNdArray(), 'k', lw=1, label='x')
        #     plt.plot(y.getNdArray(), '.k', label='y=x+n')
        #     plt.plot(derivative.getNdArray(), '.b', lw=2, label='∂x')
        #     plt.legend()
        #     plt.title('Model, Data and Derivative')
        #     plt.show()
        #
        # # CG solver
        # problemLS = ProblemL2Linear(x.clone().zero(), y, Iop)
        # CG = LCGsolver(BasicStopper(niter=30))
        # CG.run(problemLS, verbose=True)
        # if PLOT:
        #     plt.figure(figsize=(5, 4))
        #     plt.plot(x.getNdArray(), 'k', lw=1, label='x')
        #     plt.plot(y.getNdArray(), '.k', label='y=x+n')
        #     plt.plot(problemLS.model.getNdArray(), 'r', lw=2, label='x_inv')
        #     plt.legend()
        #     plt.title('Least-Squares CG')
        #     plt.show()
        #
        # # LSQR solver
        # problemLSQR = ProblemL2Linear(x.clone().zero(), y, Iop)
        # LSQR = LSQRsolver(BasicStopper(niter=30))
        # LSQR.run(problemLSQR, verbose=True)
        # if PLOT:
        #     plt.figure(figsize=(5, 4))
        #     plt.plot(x.getNdArray(), 'k', lw=1, label='x')
        #     plt.plot(y.getNdArray(), '.k', label='y=x+n')
        #     plt.plot(problemLSQR.model.getNdArray(), 'r', lw=2, label='x_inv')
        #     plt.legend()
        #     plt.title('Least-Squares LSQR')
        #     plt.show()
        #
        # # CG solver with L2 regularization
        # problemLSR = ProblemL2LinearReg(x.clone().zero(), y, Iop, np.sqrt(50), L)
        # CG = LCGsolver(BasicStopper(niter=30))
        # CG.run(problemLSR, verbose=True)
        # if PLOT:
        #     plt.figure(figsize=(5, 4))
        #     plt.plot(x.getNdArray(), 'k', lw=1, label='x')
        #     plt.plot(y.getNdArray(), '.k', label='y=x+n')
        #     plt.plot(problemLSR.model.getNdArray(), 'r', lw=2, label='x_inv')
        #     plt.legend()
        #     plt.title('CG with Laplacian reg')
        #     plt.show()
        #
        # # LSQR solver with L2 regularization
        # problemLSR_1 = ProblemL2LinearReg(x.clone().zero(), y, Iop, np.sqrt(50), L)
        # LSQR = LSQRsolver(BasicStopper(niter=30))
        # LSQR.run(problemLSR_1, verbose=True)
        # if PLOT:
        #     plt.figure(figsize=(5, 4))
        #     plt.plot(x.getNdArray(), 'k', lw=1, label='x')
        #     plt.plot(y.getNdArray(), '.k', label='y=x+n')
        #     plt.plot(problemLSR_1.model.getNdArray(), 'r', lw=2, label='x_inv')
        #     plt.legend()
        #     plt.title('LSQR with Laplacian reg')
        #     plt.show()
        #
        # # FISTA
        # problemFISTA = ProblemL1Lasso(x.clone().zero(), y, Iop, lambda_value=1, op_norm=1)
        # FISTA = ISTAsolver(BasicStopper(niter=300), fast=True)
        # FISTA.run(problemFISTA, verbose=True)
        # if PLOT:
        #     plt.figure(figsize=(5, 4))
        #     plt.plot(x.getNdArray(), 'k', lw=1, label='x')
        #     plt.plot(y.getNdArray(), '.k', label='y=x+n')
        #     plt.plot(problemFISTA.model.getNdArray(), 'r', lw=2, label='x_inv')
        #     plt.legend()
        #     plt.title('FISTA inversion')
        #     plt.show()
        
        # SplitBregman
        problemSB = ProblemLinearReg(x.clone().zero(), y, Iop, regsL1=TV, epsL1=3.)
        SB = SplitBregmanSolver(BasicStopper(niter=50), niter_inner=10, niter_solver=10,
                                linear_solver='LSQR', breg_weight=1., use_prev_sol=False)
        SB.run(problemSB, verbose=True, inner_verbose=False)
        if PLOT:
            plt.figure(figsize=(5, 4))
            plt.plot(x.getNdArray(), 'k', lw=1, label='x')
            plt.plot(y.getNdArray(), '.k', label='y=x+n')
            plt.plot(derivative.getNdArray(), ':k', lw=1, label='∂x')
            plt.plot(problemSB.model.getNdArray(), 'r', lw=2, label='x_inv')
            plt.plot((TV * problemSB.model).getNdArray(), ':r', lw=2, label='∂(x_inv)')
            plt.legend()
            plt.title('SB inversion')
            plt.show()
    
        # ADMM
        problemADMM = ProblemLinearReg(x.clone().zero(), y, Iop, regsL1=TV, epsL1=3.)
        
        ADMM = ADMMsolver(BasicStopper(niter=30), niter_linear=10, niter_lasso=10)
        ADMM.run(problemADMM, verbose=True, inner_verbose=False)
        if PLOT:
            plt.figure(figsize=(5, 4))
            plt.plot(x.getNdArray(), 'k', lw=1, label='x')
            plt.plot(y.getNdArray(), '.k', label='y=x+n')
            plt.plot(derivative.getNdArray(), ':k', lw=1, label='∂x')
            plt.plot(problemADMM.model.getNdArray(), 'r', lw=2, label='x_inv')
            plt.plot((TV * problemADMM.model).getNdArray(), ':r', lw=2, label='∂(x_inv)')
            plt.legend()
            plt.title('ADMM inversion')
            plt.show()
    
    elif EXAMPLE == 'gaussian':
        x = pyVector.vectorIC(np.empty((301, 601))).set(0)
        x.getNdArray()[150, 300] = 1.0
        # x.getNdArray()[100, 200] = -5.0
        # x.getNdArray()[280, 400] = 1.0
        if PLOT:
            plt.figure(figsize=(6, 3))
            plt.imshow(x.getNdArray()), plt.colorbar()
            plt.title('Model')
            plt.show()

        G = pyNpOperator.Gauss_smooth_scipy(x, 25, 15)
        y = G * x
        # y.scale(1./y.norm())
        if PLOT:
            plt.figure(figsize=(6, 3))
            plt.imshow(y.getNdArray()), plt.colorbar()
            plt.title('Data')
            plt.show()

        # CG solver
        problemLS = ProblemL2Linear(x.clone().zero(), y, G)
        CG = LCGsolver(BasicStopper(niter=30))
        CG.setDefaults()
        CG.run(problemLS, verbose=True)
        if PLOT:
            plt.figure(figsize=(6, 3))
            plt.imshow(problemLS.model.getNdArray()), plt.colorbar()
            plt.title('CG, %d its' % CG.stopper.niter)
            plt.show()

        # FISTA
        problemFISTA = ProblemL1Lasso(x.clone().zero(), y, G, lambda_value=1000, op_norm=1.)
        FISTA = ISTAsolver(BasicStopper(niter=1000), fast=True)
        FISTA.setDefaults()
        FISTA.run(problemFISTA, verbose=True)
        if PLOT:
            plt.figure(figsize=(6, 3))
            plt.imshow(problemFISTA.model.getNdArray()), plt.colorbar()
            plt.title(r'FISTA, $\lambda$=%.2e, %d its' % (problemFISTA.lambda_value, FISTA.stopper.niter))
            plt.show()

        # SplitBregman
        I = pyOperator.IdentityOp(x)
        problemSB = ProblemLinearReg(x.clone().zero(), y, G, regsL1=I, epsL1=10.)

        SB = SplitBregmanSolver(BasicStopper(niter=50), niter_inner=3, niter_solver=30,
                                linear_solver='LSQR', breg_weight=1., use_prev_sol=False)
        SB.setDefaults()
        SB.run(problemSB, verbose=True, inner_verbose=False)
        if PLOT:
            plt.figure(figsize=(6, 3))
            plt.imshow(problemSB.model.getNdArray()), plt.colorbar()
            plt.title('SplitBregman')
            plt.show()

        # ADMM
        problemADMM = ProblemLinearReg(x.clone().zero(), y, G, regsL1=I, epsL1=2.)

        ADMM = ADMMsolver(BasicStopper(niter=10), niter_linear=30, niter_lasso=100)
        ADMM.setDefaults()
        ADMM.run(problemADMM, verbose=True, inner_verbose=True)
        if PLOT:
            plt.figure(figsize=(6, 3))
            plt.imshow(problemADMM.model.getNdArray()), plt.colorbar()
            plt.title('ADMM')
            plt.show()
    
    elif EXAMPLE == 'medical':
        x = pyVector.vectorIC(np.load('../testdata/shepp_logan_phantom.npy', allow_pickle=True).astype(np.float32))
        if PLOT:
            plt.figure(figsize=(5, 4))
            plt.imshow(x.getNdArray(), cmap='bone'), plt.colorbar()
            plt.title('Model')
            plt.show()
            
        nh = [5, 10]
        hz = np.exp(-0.1 * np.linspace(-(nh[0] // 2), nh[0] // 2, nh[0]) ** 2)
        hx = np.exp(-0.03 * np.linspace(-(nh[1] // 2), nh[1] // 2, nh[1]) ** 2)
        hz /= np.trapz(hz)  # normalize the integral to 1
        hx /= np.trapz(hx)  # normalize the integral to 1
        h = hz[:, np.newaxis] * hx[np.newaxis, :]
        if PLOT:
            plt.figure(figsize=(5, 4))
            plt.imshow(h, aspect='equal'), plt.colorbar()
            plt.title('Blurring Kernel')
            plt.show()
        Blurring = pyNpOperator.ConvNDscipy(model=x, kernel=pyVector.vectorIC(h))
        
        y = Blurring * x
        if PLOT:
            plt.figure(figsize=(5, 4))
            plt.imshow(y.getNdArray(), cmap='bone'), plt.colorbar()
            plt.title('Data')
            plt.show()
            
        # CG solver
        problemLS = ProblemL2Linear(x.clone().zero(), y, Blurring)
        CG = LCGsolver(BasicStopper(niter=50))
        CG.run(problemLS, verbose=True)
        if PLOT:
            plt.figure(figsize=(5, 4))
            plt.imshow(problemLS.model.getNdArray(), cmap='bone'), plt.colorbar()
            plt.title('CG, %d iter' % CG.stopper.niter)
            plt.show()

        # FISTA
        problemFISTA = ProblemL1Lasso(x.clone().zero(), y, Blurring, lambda_value=1, op_norm=1.25)
        FISTA = ISTAsolver(BasicStopper(niter=100), fast=True)
        FISTA.run(problemFISTA, verbose=True)
        if PLOT:
            plt.figure(figsize=(5, 4))
            plt.imshow(problemFISTA.model.getNdArray(), cmap='bone'), plt.colorbar()
            plt.title(r'FISTA, $\lambda$=%.2e, %d iter'
                      % (problemFISTA.lambda_value, FISTA.stopper.niter))
            plt.show()

        # SplitBregman
        # the gradient of the image is 6e3
        D = pyNpOperator.TotalVariation(x)
        I = pyOperator.IdentityOp(x)

        problemSB = ProblemLinearReg(x.clone().zero(), y, Blurring, regsL1=D, epsL1=.01)
        
        SB = SplitBregmanSolver(BasicStopper(niter=10), niter_inner=10, niter_solver=50,
                                linear_solver='LSQR', breg_weight=1, use_prev_sol=False)
        SB.run(problemSB, verbose=True, inner_verbose=False)
        if PLOT:
            plt.figure(figsize=(5, 4))
            plt.imshow(problemSB.model.getNdArray(), cmap='bone'), plt.colorbar()
            plt.title(r'SB TV, $\varepsilon=%.2e$, %d iter'
                      % (problemSB.epsL1[0], SB.stopper.niter))
            plt.show()

        # ADMM
        problemADMM = ProblemLinearReg(x.clone().zero(), y, Blurring,
                                     regsL1=D, epsL1=.1)

        ADMM = ADMMsolver(BasicStopper(niter=10), niter_linear=30, niter_lasso=10)
        ADMM.setDefaults(save_obj=True, save_model=True)
        ADMM.run(problemADMM, verbose=True, inner_verbose=True)
        if PLOT:
            plt.figure(figsize=(5, 4))
            plt.imshow(problemADMM.model.getNdArray(), cmap='bone'), plt.colorbar()
            plt.title(r'ADMM TV, $\varepsilon=%.2e$, %d iter'
                      % (problemADMM.epsL1[0], ADMM.stopper.niter))
            plt.show()
        
    else:
        raise ValueError("EXAMPLE has to be one of noisy, gaussian, medical")

    return 0


if __name__ == '__main__':
    main()
