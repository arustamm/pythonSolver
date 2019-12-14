# Module containing the definition of inverse problems where the ADMM method is used
import pyOperator
import pyVector
from pyLinearSolver import LCGsolver
from pyProblem import Problem, ProblemL1Lasso, ProblemL2LinearReg
from pySolver import Solver
from pySparseSolver import ISTAsolver
from pyStopper import BasicStopper
from math import isnan


class ProblemLinearReg(Problem):
    def __init__(self, model, data, op, dfw=1., epsL1=None, regsL1=None, epsL2=None, regsL2=None, dataregsL2=None,
                 minBound=None, maxBound=None, boundProj=None):
        """
        Linear Problem with both L1 and L2 regularizers:

        .. math ::
            dfw / 2 |Op m - d|_2^2 +
            \sum_i epsL2_i |R2_i m - dr|_2^2 +
            \sum_i epsL1_i |R1_i m|_1

        :param model        : vector; initial model
        :param data         : vector; data
        :param op           : LinearOperator; data fidelity operator
        :param dfw          : float; weight of the data fidelity term
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
        self.dfw = dfw
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
        
        # L2 Regularizations (not mandatory)
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
        self.res_data = self.op.range.clone().zero()
        self.res_regsL2 = self.regL2_op.range.clone().zero() if self.nregsL2 != 0 else None
        self.res_regsL1 = self.regL1_op.range.clone().zero() if self.nregsL1 != 0 else None
        self.res = pyVector.superVector(self.res_data, self.res_regsL2, self.res_regsL1)
        
        # flags for avoiding extra computations
        self.res_data_already_computed = False
        self.res_regsL1_already_computed = False
        self.res_regsL2_already_computed = False

    def __del__(self):
        """Default destructor"""
        return
    
    def objf(self, res):
        """Compute objective function based on the residuals"""
        res_data = res.vecs[0]
        res_regsL2 = res.vecs[1] if self.res_regsL2 is not None else None
        if self.res_regsL1 is not None:
            res_regsL1 = res.vecs[2] if self.res_regsL2 is not None else res.vecs[1]
        else:
            res_regsL1 = None
        
        self.obj_terms[0] = self.dfw*.5 * res_data.norm()**2  # data fidelity
        
        if res_regsL2 is not None:
            for idx in range(self.nregsL2):
                self.obj_terms[1 + idx] = self.epsL2[idx] * res_regsL2.vecs[idx].norm()**2
        if res_regsL1 is not None:
            for idx in range(self.nregsL1):
                self.obj_terms[1 + self.nregsL2 + idx] = self.epsL1[idx] * res_regsL1.vecs[idx].norm(1)
        
        return sum(self.obj_terms)
    
    def resf(self, model):
        # compute data residual: Op * m - d
        if not self.res_data_already_computed:
            if model.norm() != 0:
                self.op.forward(False, self.model, self.res_data)
            else:
                self.res.vecs[0].zero()
            self.res_data.scaleAdd(self.data, 1., -1.)
            self.res_data_already_computed = True
        
        # compute L2 reg residuals
        if not self.res_regsL2_already_computed and self.res_regsL2 is not None:
            if model.norm() != 0 and self.regL2_op is not None:
                self.regL2_op.forward(False, self.model, self.res_regsL2)
            
            if self.dataregsL2 is not None and self.dataregsL2.norm() != 0.:
                self.res_regsL2.scaleAdd(self.dataregsL2, 1., -1.)
            self.res_regsL2_already_computed = True
        
        # compute L1 reg residuals
        if not self.res_regsL1_already_computed and self.res_regsL1 is not None:
            if model.norm() != 0. and self.regL1_op is not None:
                self.regL1_op.forward(False, self.model, self.res_regsL1)
            else:
                self.res_regsL1.zero()
            self.res_regsL1_already_computed = True
        
        return self.res
    

def _shrinkage(x, thresh, eps=1e-10):
    """
    Shrinkage function Gamma
        y = x / (|x| + eps) * maximum(|x| - alpha, 0)
    """
    y = x.clone()
    y / x.clone().abs().addbias(eps)
    return y * x.clone().abs().addbias([-t for t in thresh]).maximum(x.clone().zero())


class SplitBregmanSolver(Solver):
    """Split-Bregman solver for L1 and L2 regularized problems"""

    # Default class methods/functions
    def __init__(self, stopper, logger=None, niter_inner=3, niter_solver=5, breg_weight=1., steepest=False):
        """
        Constructor for Split-Bregman Solver
        :param stopper      : stopper object
        :param logger       : logger object
        :param niter_inner  : int; number of iterations for the shrinkage loop [default 3]
        :param niter_solver : int; number of iterations for the internal linear solver [default 5]
        :param breg_weight  : float; coefficient for the Bregman update b += beta * (R*x - d) [1.]
        :param steepest     : bool; use Steepest Descent instead of Conjugate Gradient [False]
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
        self.breg_weight = breg_weight

        self.linear_solver = LCGsolver(BasicStopper(niter=self.niter_solver), steepest=steepest, logger=self.logger)
        self.linear_solver.setDefaults(iter_sampling=1, flush_memory=True)

        self.breg_b = None
        self.breg_a = None
        self.dataregsL1 = None
        self.solution = None
        
        # print formatting
        self.iter_msg = "iter = %s, obj = %.2e, df_obj = %.2e, reg_obj = %.2e, resnorm = %.2e"

    def __del__(self):
        print('Destructor called, Split-Bregman deleted')
    
    def _update_dataregsL1(self):
        # self.dataregsL1 = self.breg_a.clone() - self.breg_b
        self.dataregsL1.copy(self.breg_a.clone() - self.breg_b)
        
    def run(self, problem, verbose=False, inner_verbose=False, restart=False, initial_guess=None):
        """Running SplitBregman solver"""
        assert type(problem) == ProblemLinearReg, 'problem has to be a ProblemLinearReg'
        if problem.nregsL1 == 0:
            raise ValueError('ERROR! Provide at least one L1 regularizer!')
        
        # reset stopper before running the inversion
        self.stopper.reset()

        # initialize all the vectors and operators for Split-Bregman
        self.breg_b = problem.regL1_op.range.clone()
        self.breg_a = self.breg_b.clone()
        self.dataregsL1 = self.breg_b.clone()
        RL1x = problem.regL1_op.range.clone()  # store RegL1 * solution
        
        sb_mdl = problem.model.clone().zero() if initial_guess is None else initial_guess.clone()
        
        # reweight the eps according to dfw
        eps = [(e / problem.dfw)**.5 for e in problem.epsL2 + problem.epsL1]
        
        # # inner L2 reg problem
        # linear_problem = ProblemL2LinearReg(
        #     model=sb_mdl,
        #     data=problem.data,
        #     op=problem.op,
        #     epsilon=eps,
        #     reg_op=pyOperator.Vstack(problem.regL2_op, problem.regL1_op),
        #     prior_model=pyVector.superVector(problem.dataregsL2, self.dataregsL1),
        #     minBound=problem.minBound,
        #     maxBound=problem.maxBound,
        #     boundProj=problem.boundProj
        # )

        # TODO add restart and merge with initial guess
        if restart:
            self.restart.read_restart()
            outer_iter = self.restart.retrieve_parameter("iter")
            initial_obj_value = self.restart.retrieve_parameter("obj_initial")
            sb_mdl = self.restart.retrieve_vector("sb_mdl")
            
            msg = "Restarting previous solver run from: %s" % self.restart.restart_folder
            if verbose:
                print(msg)
            if self.logger:
                self.logger.addToLog(msg)

        else:
            outer_iter = 0
            
            msg = 90 * '#' + '\n'
            msg += "\t\t\t\t\tSPLIT-BREGMAN ALGORITHM log file\n\n"
            msg += "\tRestart folder: %s\n" % self.restart.restart_folder
            msg += "\tModeling Operator:\t\t%s\n" % problem.op
            msg += "\tData Fidelity weight:\t%.2e\n" % problem.dfw
            if problem.nregsL2 != 0:
                msg += "\tL2 Regularizer ops:\t\t" + ", ".join(["%s" % op for op in problem.regL2_op.ops]) + "\n"
                msg += "\tL2 Regularizer weights:\t" + ", ".join(["{:.2e}".format(e) for e in problem.epsL2]) + "\n"
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
            print("\tInitial SB model norm = %.2e" % sb_mdl.norm())
            print("\tInitial breg_a norm = %.2e" % self.breg_a.norm())
            print("\tInitial breg_b norm = %.2e" % self.breg_b.norm())

            for iter_inner in range(self.niter_inner):
                
                # update L1 regularizer data for the inner problem
                # self._update_dataregsL1()
                
                # solve inner problem
                # inner L2 reg problem
                linear_problem = ProblemL2LinearReg(
                    model=sb_mdl,
                    data=problem.data,
                    op=problem.op,
                    epsilon=eps,
                    reg_op=pyOperator.Vstack(problem.regL2_op, problem.regL1_op),
                    prior_model=pyVector.superVector(problem.dataregsL2, self.breg_a.clone() - self.breg_b),
                    minBound=problem.minBound,
                    maxBound=problem.maxBound,
                    boundProj=problem.boundProj
                )
                linear_problem.setDefaults()
                self.linear_solver.run(linear_problem, verbose=inner_verbose)
                sb_mdl = linear_problem.model
                
                # compute RL1*x
                problem.regL1_op.forward(False, sb_mdl, RL1x)
                
                # update breg_a
                self.breg_a = _shrinkage(RL1x.clone() + self.breg_b, thresh=eps[-problem.nregsL1:])
                
                msg = "\t\tInner iter %d, mdl = %.2e, breg_a = %.2e" % (iter_inner, sb_mdl.norm(), self.breg_a.norm())
                if verbose:
                    print(msg)
                if self.logger:
                    self.logger.addToLog("\n" + msg)
                    
            # update breg_b
            self.breg_b.scaleAdd(RL1x.clone() - self.breg_a, 1., self.breg_weight)
            msg = "\t Bregman loop completed, breg_b = %.2e" % self.breg_b.norm()
            if verbose:
                print(msg)
            if self.logger:
                self.logger.addToLog("\n" + msg)
                
            outer_iter += 1
            # check objective function
            problem.res_regsL1 = RL1x.clone()
            problem.res_regsL1_already_computed = True
            obj1 = problem.get_obj(sb_mdl)
            # if obj1 >= obj0:
            #     msg = "Objective function didn't reduce, will terminate solver:\n\t"\
            #           "obj_new = %.2e\tobj_cur = %.2e" % (obj1, obj0)
            #     if verbose:
            #         print(msg)
            #     if self.logger:
            #         self.logger.addToLog(msg)
            #     break
            
            # TODO save cost data, logger and all the stuff
            # iteration info
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
    def __init__(self, stopper, logger=None, niter_LCG=5, niter_ISTA=15, rho=None, auto_rho=True, mu=10., tau=2.):
        """
        Constructor for ADMM Solver
        .. math ::
            dfw/2 |Op x - d|_2^2 + \sum_i epsL2_i |R2_i x - dr|_2^2 + \gamma |y|_1
                subject to Ax + By = c
        Note: for now B=-I, c=0 (i.e., y=Ax)
        :param stopper      : stopper object
        :param logger       : logger object
        :param niter_LCG    : int; number of iterations for solving the linear problem [5]
        :param niter_ISTA   : int; number of iterations for solving the lasso problem [5]
        :param rho          : float; penalty parameter rho (if None it is initialized as 2*gamma+.1)
        :param auto_rho     : bool; update rho automatically
        :param mu           : float; norm ratio between residuals for updating rho [10]
        :param tau          : float; scaling factor for updating rho [2]
        """
        # Calling parent construction
        super(ADMMsolver, self).__init__()

        self.stopper = stopper
        self.logger = logger
        self.stopper.logger = self.logger
        self.niter_LCG = niter_LCG
        self.niter_ISTA = niter_ISTA
        self.solver_LCG = LCGsolver(BasicStopper(niter=self.niter_LCG), steepest=False, logger=self.logger)
        self.solver_LCG.setDefaults(iter_sampling=1, flush_memory=True)
        self.solver_ISTA = ISTAsolver(BasicStopper(niter=self.niter_ISTA), fast=True, logger=self.logger)
        self.solver_ISTA.setDefaults(iter_sampling=1, flush_memory=True)

        self.rho = rho      # ADMM penalty parameter
        self.mu = mu
        self.tau = tau
        self.auto_rho = auto_rho
        self.primal = None      # aka r (in the complete formulation is A x + B z - c)
        self.dual = None        # aka s (in the complete formulation is rho A.H B r)
        self.x = None           # solution vector
        self.y = None           # lagrangian vector
        self.u = None           # scaled dual variable
        self.y_minus_u = None   # store y-u
        self.A = None           # constraint matrix A
        self.Ax = None          # store A*x
        self.Ax_plus_u = None   # storee A*x+u for ISTA data

        # print formatting
        self.iter_msg = "iter = %s, obj = %.2e, df_obj = %.2e, reg_obj = %.2e, resnorm = %.2e"

    def __del__(self):
        print('Destructor called, ADMM deleted')

    def compute_primal(self):
        """r = Ax - y"""
        self.primal = self.Ax.clone() - self.y

    def compute_dual(self):
        """s = -rho A.H r"""
        self.A.adjoint(False, self.dual, self.primal)
        self.dual.scale(-self.rho)
    
    def compute_y_minus_u(self):
        self.y_minus_u = self.y.clone() - self.u
        
    def compute_Ax(self):
        self.A.forward(False, self.x, self.Ax)
        
    def compute_Ax_plus_u(self):
        self.Ax_plus_u = self.Ax.clone() + self.u
    
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
        # I want to set dfw=1, so:
        epsL2 = [e / problem.dfw for e in problem.epsL2]
        epsL1 = [e / problem.dfw for e in problem.epsL1]
        gamma = max(epsL1)

        # A is the Vstack of the L1 reg operators, scaled by their respective eps
        self.A = problem.regL1_op * [e / gamma for e in epsL1]
        
        # initialize all others variables
        if self.rho is None:
            self.init_rho(gamma)
        self.x = initial_guess.clone() if initial_guess is not None else problem.model.clone().zero()
        self.y = self.A.range.clone().zero()
        self.u = self.y.clone()
        self.compute_y_minus_u()
        self.dual = self.A.domain.clone()
        
        # Linear Problem:       1/2 | Op x - d| + epsL2   | R2 x - dr   |
        #                                         rho/2   | A x  - (y-u)|
        problem_linear = ProblemL2LinearReg(
            model=self.x,
            data=problem.data,
            op=problem.op,
            epsilon=epsL2 + [self.rho/2]*self.A.n,
            reg_op=pyOperator.Vstack(problem.regL2_op, self.A),
            prior_model=pyVector.superVector(problem.dataregsL2, self.y_minus_u),
            minBound=problem.minBound,
            maxBound=problem.maxBound,
            boundProj=problem.boundProj
        )

        # lasso problem: rho/2 | A x - z + u|_2^2 + gamma | y |_1
        # this means to solve: 1/2 | I y - (Ax + u)| + gamma/rho | y |_1
        self.Ax = self.A.range.clone()
        self.compute_Ax_plus_u()
        problem_lasso = ProblemL1Lasso(
            model=self.y,
            data=self.Ax_plus_u,
            op=pyOperator.IdentityOp(self.y),
            op_norm=1.,
            lambda_value=gamma/self.rho,
            minBound=problem.minBound,
            maxBound=problem.maxBound,
            boundProj=problem.boundProj
        )
        
        if restart:
            self.restart.read_restart()
            outer_iter = self.restart.retrieve_parameter("iter")
            initial_obj_value = self.restart.retrieve_parameter("obj_initial")
            self.x = self.restart.retrieve_vector("solution")
    
            msg = "Restarting previous solver run from: %s" % self.restart.restart_folder
            if verbose:
                print(msg)
            if self.logger:
                self.logger.addToLog(msg)
        else:
            outer_iter = 0
    
            msg = 90 * '#' + '\n'
            msg += "\t\t\t\t\tADMM ALGORITHM log file\n\n"
            msg += "\tRestart folder: %s\n" % self.restart.restart_folder
            msg += "\tModeling Operator:\t\t%s\n" % problem.op
            msg += "\tData Fidelity weight:\t%.2e\n" % problem.dfw
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
            obj0 = problem.get_obj(self.x)

            if outer_iter == 0:
                initial_obj_value = obj0
                self.restart.save_parameter("obj_initial", initial_obj_value)
                msg = self.iter_msg % (str(outer_iter).zfill(self.stopper.zfill),
                                       obj0,
                                       problem.obj_terms[0],
                                       obj0 - problem.obj_terms[0],
                                       problem.get_rnorm(self.x))
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
            
            # update x
            self.solver_LCG.setDefaults()
            self.solver_LCG.run(problem_linear, verbose=inner_verbose)
            self.x = problem_linear.model
            
            # update y TODO FISTA Gradient vanishes identically: why?
            self.compute_Ax()
            self.compute_Ax_plus_u()  # i.e., data for the LASSO problem
            problem_lasso.data = self.Ax_plus_u  # TODO why the pointer is not working?
            self.solver_ISTA.setDefaults()
            problem_lasso.set_lambda(gamma/self.rho)
            self.solver_ISTA.run(problem_lasso, verbose=inner_verbose)  # TODO the iteration is not updating
            self.y = problem_lasso.model
            
            # update penalty parameter and scaled dual variable
            self.compute_primal()  # r = Ax - y
            self.compute_dual()    # s = rho A.H r
            self.update_rho()
            self.u.__add__(self.primal).scale(1/self.rho)  # u = (u + r)/rho

            outer_iter += 1
            # check objective function
            obj1 = problem.get_obj(self.x)
            if obj1 >= obj0:
                msg = "Objective function didn't reduce, will terminate solver:\n\t" \
                      "obj_new = %.2e\tobj_cur = %.2e" % (obj1, obj0)
                if verbose:
                    print(msg)
                if self.logger:
                    self.logger.addToLog(msg)
                break
            
            # iteration info
            msg = self.iter_msg % (str(outer_iter).zfill(self.stopper.zfill),
                                   obj1,
                                   problem.obj_terms[0],
                                   obj1 - problem.obj_terms[0],
                                   problem.get_rnorm(self.x))
            if verbose:
                print(msg)
            if self.logger:
                self.logger.addToLog("\n" + msg)

            # saving in case of restart
            self.restart.save_parameter("iter", outer_iter)
            self.restart.save_vector("solution", self.x)

            if self.stopper.run(problem, outer_iter, initial_obj_value, verbose):
                break

        # writing last inverted model
        self.save_results(outer_iter, problem, model=None, force_save=False, force_write=False)
    
        # ending message and log file
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
    from scipy.ndimage.filters import gaussian_filter
    import pyVector
    import pyOperator
    import matplotlib.pyplot as plt
    from pyProblem import ProblemL2Linear

    class Gauss_smooth_scipy(pyOperator.Operator):
        def __init__(self, model, sigmax, sigmaz):
            """
            Gaussian 2D smoothing operator using scipy smoothing:
            model    = [no default] - vector class; domain vector
            sigmax   = [no default] - float; standard deviation along the x direction
            sigmaz   = [no default] - float; standard deviation along the z direction
            """
            self.setDomainRange(model, model)
            self.sigmax = sigmax
            self.sigmaz = sigmaz
            self.scaling = 2.0 * np.pi * sigmax * sigmaz
            return
    
        def __str__(self):
            return "GauSmoot"
    
        def forward(self, add, model, data):
            """Forward operator"""
            self.checkDomainRange(model, data)
            if not add:
                data.zero()
            # Getting Ndarrays
            model_arr = model.getNdArray()
            data_arr = data.getNdArray()
            data_arr[:] = self.scaling * gaussian_filter(model_arr,
                                                         sigma=[self.sigmax, self.sigmaz])
            return
    
        def adjoint(self, add, model, data):
            """Self-adjoint operator"""
            self.forward(add, data, model)
            return

    model = pyVector.vectorIC(np.empty((301, 601))).set(0)
    model_arr = model.getNdArray()
    model_arr[150, 300] = 10.0
    model_arr[100, 200] = -5.0
    model_arr[280, 400] = 1.0
    
    plt.figure(figsize=(7, 3))
    plt.imshow(model_arr, cmap='gray'), plt.colorbar(), plt.title('Model')
    plt.show()
    
    G = Gauss_smooth_scipy(model, 5., 4.)
    data = G * model
    
    plt.figure(figsize=(7, 3))
    plt.imshow(data.getNdArray(), cmap='gray'), plt.colorbar(), plt.title('Data')
    plt.show()
    #
    # # CG solver
    # problemLS = ProblemL2Linear(model, data, G)
    # CG = LCGsolver(BasicStopper(niter=100))
    # CG.setDefaults()
    # CG.run(problemLS, verbose=True)
    # plt.imshow(problemLS.model.getNdArray()), plt.colorbar(), plt.title('CG Solution')
    # plt.show()
    #
    # # FISTA
    # problemFISTA = ProblemL1Lasso(model, data, G, lambda_value=.1)
    # FISTA = ISTAsolver(BasicStopper(niter=100), fast=True)
    # FISTA.setDefaults()
    # FISTA.run(problemFISTA, verbose=True)
    # plt.imshow(problemFISTA.model.getNdArray()), plt.colorbar(), plt.title('FISTA Solution')
    # plt.show()
    
    # TODO SB and ADMM do not recover the real amplitude. Why?
    # SplitBregman
    problemSB = ProblemLinearReg(model=model, data=data, op=G,
                                 regsL1=pyOperator.IdentityOp(model), epsL1=.01)

    SB = SplitBregmanSolver(BasicStopper(niter=10), niter_inner=10, niter_solver=10, steepest=False, breg_weight=1.)
    SB.setDefaults()
    SB.run(problemSB, verbose=True, inner_verbose=False)
    plt.figure(figsize=(7, 3))
    plt.imshow(problemSB.model.getNdArray(), cmap='gray'), plt.colorbar()
    plt.title(r'SplitBregman, $\lambda$=%.1e, iters=%d,%d,%d, $\beta$=%.2f'
              % (problemSB.epsL1[0],
                 SB.stopper.niter, SB.niter_inner, SB.niter_solver,
                 SB.breg_weight))
    plt.show()
    mse_sb = (model.clone() - problemSB.model).norm()**2
    print("Split-Bregman solution MSE = %.2e" % mse_sb)

    # ADMM
    problemADMM = ProblemLinearReg(model=model, data=data, op=G,
                                   regsL1=pyOperator.IdentityOp(model), epsL1=.01)
    ADMM = ADMMsolver(BasicStopper(niter=1))
    ADMM.setDefaults()
    ADMM.run(problemADMM, verbose=True, inner_verbose=True)
    
    plt.figure(figsize=(7, 3))
    plt.imshow(problemADMM.model.getNdArray(), cmap='gray'), plt.colorbar(),\
    plt.title('ADMM Solution, iters=%d,%d,%d' % (ADMM.stopper.niter, ADMM.niter_LCG, ADMM.niter_ISTA))
    plt.show()
    return 0


if __name__ == '__main__':
    main()
