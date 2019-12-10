# Module containing the definition of inverse problems where the ADMM method is used
import pyOperator
import pyVector
from pyLinearSolver import LCGsolver
from pyProblem import Problem, ProblemL1Lasso, ProblemL2LinearMultiReg
from pySolver import Solver
from pySparseSolver import ISTAsolver
from pyStopper import BasicStopper


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
        :param epsL1        : list; weights of L1 regularizers [.1]
        :param regsL1       : list; L1 regularizers of class LinearOperator [Identity]
        :param epsL2        : list; weights of L2 regularizers [None]
        :param regsL2       : list; L2 regularizers of class LinearOperator [None]
        :param dataregsL2   : vector; prior model for L2 regularization term [zeros]
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
            self.dataregsL2 = dataregsL2 if dataregsL2 is not None else self.regL2_op.range.clone()
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

    def __del__(self):
        """Default destructor"""
        return

    def objf(self, res):
        """Compute objective function based on the residuals"""
        self.obj_terms[0] = .5 * res[0].norm() ** 2  # data fidelity

        for idx in range(self.nregsL2):
            self.obj_terms[1 + idx] = self.epsL2[idx] * res[1 + idx].norm() ** 2
        for idx in range(self.nregsL1):
            self.obj_terms[1 + self.nregsL2 + idx] = self.epsL1[idx] * res[1 + self.nregsL2 + idx].norm(1)
        return sum(self.obj_terms)


def _shrinkage(x, thresh, eps=1e-10):
    """
    Shrinkage function Gamma
        y = x / (|x| + eps) * maximum(|x| - alpha, 0)
    """
    y = x.clone()
    y / x.clone().abs().addbias(eps)
    return y * x.clone().abs().addbias(-thresh).maximum(x.clone().zero())


class SplitBregmanSolver(Solver):
    """Split-Bregman solver for L1 and L2 regularized problems"""

    # Default class methods/functions
    def __init__(self, stopper, logger=None, niter_inner=3, niter_solver=5,
                 breg_weight=1., use_prev_sol=False):
        """
        Constructor for Split-Bregman Solver
        :param stopper      : stopper object
        :param logger       : logger object
        :param niter_inner  : int; number of iterations for the shrinkage loop [default 3]
        :param niter_solver : int; number of iterations for the internal CG solver [default 5]
        :param breg_weight  : float; coefficient for the Bregman update b += beta * (R*x - d) [1.]
        :param use_prev_sol : bool; use the previous solution [False]
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
        self.use_prev_sol = use_prev_sol  # as initial guess for the inner problem

        self.inner_solver = LCGsolver(BasicStopper(niter=self.niter_solver), steepest=False, logger=None)
        self.inner_solver.setDefaults(iter_sampling=1, flush_memory=True)

        self.breg_b = None
        self.breg_a = None
        self.dataregsL1 = None
        
    def __del__(self):
        print('Destructor called, Split-Bregman deleted')
    
    def _update_dataregsL1(self):
        self.dataregsL1 = self.breg_a.clone() - self.breg_b
        
    def run(self, problem, verbose=False, restart=False, initial_guess=None):
        """Running SplitBregman solver"""
        assert type(problem) == ProblemLinearReg, 'problem has to be a ProblemLinearReg'
        if problem.nregsL1 == 0:
            raise ValueError('ERROR! Provide at least one L1 regularizer!')
        
        # reset stopper before running the inversion
        self.stopper.reset()

        # initialize all the vectors and operators for Split-Bregman
        self.breg_b = problem.regL1_op.range.clone()
        self.breg_a = self.breg_b.clone()
        dataregsL1 = self.breg_b.clone()
        RL1x = problem.regL1_op.range.clone()  # store RegL1 * solution
        
        # reweight the eps according to dfw
        eps = [(e / problem.dfw)**.5 for e in problem.epsL2 + problem.epsL1]

        # inner L2 reg problem
        inner_problem = ProblemL2LinearMultiReg(
            model=problem.model,
            data=problem.data,
            op=problem.op,
            epsilon=eps,
            reg_op=pyOperator.Vstack(problem.regL2_op, problem.regL1_op),
            prior_model=pyVector.superVector(problem.dataregsL2, dataregsL1),
            minBound=problem.minBound,
            maxBound=problem.maxBound,
            boundProj=problem.boundProj
        )

        # TODO add restart and merge with initial guess
        if restart:
            msg = "Restarting previous solver run from: %s" % self.restart.restart_folder
            if verbose:
                print(msg)
            if self.logger:
                self.logger.addToLog(msg)
            self.restart.read_restart()
            outer_iter = self.restart.retrieve_parameter("iter")
            initial_obj_value = self.restart.retrieve_parameter("obj_initial")
            solution = self.restart.retrieve_vector("solution")
        
        else:
            solution = initial_guess.clone() if initial_guess is not None else problem.model.clone().zero()
            outer_iter = 0
            msg = 'SPLIT-BREGMAN ALGORITHM log file\n\n'
            msg += 90 * '#' + '\n'
            msg += "\tRestart folder: %s\n" % self.restart.restart_folder
            msg += "\tData Fidelity weight: %.2e\n" % problem.dfw
            msg += "\tL2 Regularizer weights: " + str(['%.2e' % n for n in problem.epsL2]) + "\n"
            msg += "\tL1 Regularizer weights: " + str(['%.2e' % n for n in problem.epsL1]) + "\n"
            msg += 90 * '#' + '\n'
            msg = msg.replace("'","")
            if verbose:
                print(msg.replace("log file", ""))
            if self.logger:
                self.logger.addToLog(msg)
                
        # Main iteration loop
        while True:
            obj0 = problem.get_obj(solution)
            
            for _ in range(self.niter_inner):
                
                # update L1 regularizer data for the inner problem
                self._update_dataregsL1()
                
                # solve inner problem
                if self.use_prev_sol:
                    inner_problem.model = solution
                inner_problem.setDefaults()
                inner_problem.linear = True
                self.inner_solver.run(inner_problem)
                solution = inner_problem.model

                # compute RL1*x
                problem.regsL1.forward(False, solution, RL1x)

                # update breg_a
                self.breg_a = _shrinkage(RL1x.clone() + self.breg_b, thresh=eps[-problem.nregsL1:])

            # update breg_b
            self.breg_b.scaleAdd(RL1x.clone() - self.breg_a, 1., self.breg_weight)

            # TODO save cost data, logger and all the stuff
            self.save_results(self.stopper.iter, problem, model=None, force_save=False, force_write=False)


class ADMMsolver(Solver):
    """Alternate Directions of Multipliers Method (ADMM)"""

    # Default class methods/functions
    def __init__(self, stopper, logger=None, niter_LCG=5, niter_ISTA=5, rho=None, auto_rho=True, mu=10., tau=2.,
                 use_previous_solution=False):
        """
        Constructor for ADMM Solver
        .. math ::
            dfw/2 |Op x - y|_2^2 + \sum_i epsL2_i |R2_i x - yr|_2^2 + \gamma |z|_1
                subject to z = Ax
        TODO B=-I, c=0
        :param stopper              : stopper object
        :param logger               : logger object
        :param niter_LCG            : int; number of iterations for solving the linear problem [5]
        :param niter_ISTA           : int; number of iterations for solving the lasso problem [5]
        :param rho                  : float; penalty parameter rho (if None it is initialized as 2*gamma+.1)
        :param auto_rho             : bool; update rho automatically
        :param mu                   : float; norm ratio between residuals for updating rho [10]
        :param tau                  : float; scaling factor for updating rho [2]
        :param use_previous_solution: bool; use the previous solution [False]
        """
        # Calling parent construction
        super(ADMMsolver, self).__init__()

        self.stopper = stopper
        self.logger = logger
        self.stopper.logger = self.logger

        self.niter_LCG = niter_LCG  # number of iterations for the model solver
        self.niter_ISTA = niter_ISTA  # number of iterations for the lagrangian problem

        self.solver_LCG = LCGsolver(BasicStopper(niter=self.niter_LCG), steepest=False, logger=None)
        self.solver_LCG.setDefaults(iter_sampling=1, flush_memory=True)
        self.solver_ISTA = ISTAsolver(BasicStopper(niter=self.niter_ISTA), fast=True, logger=None)
        self.solver_ISTA.setDefaults(iter_sampling=1, flush_memory=True)

        self.rho = rho      # ADMM penalty parameter
        self.mu = mu
        self.tau = tau
        self.auto_rho = auto_rho
        self.primal = None  # aka r (in the complete formulation is A x + B z - c)
        self.dual = None    # aka s (in the complete formulation is rho A.H B r)
        self.x = None       # solution vector
        self.z = None       # lagrangian vector
        self.u = None       # scaled dual variable
        self.Ax = None      # store A*x
        self.AHr = None     # store A.H*r

    def __del__(self):
        print('Destructor called, ADMM deleted')

    def compute_primal(self):
        """r = Ax - z"""
        self.primal.copy(self.Ax) - self.z

    def compute_dual(self):
        """s = -rho A.H r"""
        self.dual.copy(self.AHr).scale(-self.rho)

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

    def run(self, problem, verbose=False, restart=False, initial_guess=None):

        assert type(problem) == ProblemLinearReg, 'problem has to be a ProblemLinearReg'

        # I want to set dfw=1, so:
        epsL2 = problem.epsL2 / problem.dfw
        epsL1 = problem.epsL1 / problem.dfw
        gamma = max(epsL1)

        # A is the Vstack of the L1 reg operators, scaled by their respective eps
        A = pyOperator.Vstack(*[problem.regsL1[idx]*epsL1[idx]/gamma for idx in range(problem.nregsL1)])

        # initialize all others variables
        if self.rho is None:
            self.init_rho(gamma)
        self.x = problem.model
        self.z = A.range.clone().zero()
        self.u = A.range.clone().zero()

        # linear problem: dfw/2 |Op x - y|_2^2 + \sum_i epsL2_i |R2_i x - yr|_2^2 + rho/2 |Ax - z + u|_2^2
        # this means to solve: 1/2 | Op x - y| + eps   | R x - yr   |
        #                                        rho/2 | A x - (z-u)|
        zu = self.z.copy() - self.u
        problem_linear = ProblemL2LinearReg(
            model=self.x,
            data=problem.data,
            op=problem.op,
            epsilon=epsL2 + [self.rho/2]*A.n,
            reg_op=pyOperator.Vstack(problem.regsL2, A),
            prior_model=pyVector.superVector(problem.dataregsL2, zu),
            minBound=problem.minBound,
            maxBound=problem.maxBound,
            boundProj=problem.boundProj
        )

        # lasso problem: rho/2 | A x - z + u|_2^2 + gamma | z |_1
        self.Ax = A.range.clone()
        problem_lasso = ProblemL1Lasso(
            model=self.z,
            data=self.Ax.clone()+self.u,
            op=-pyOperator.IdentityOp(self.z),
            op_norm=None,
            lambda_value=gamma,
            minBound=problem.minBound,
            maxBound=problem.maxBound,
            boundProj=problem.boundProj
        )

        # reset stopper before running the inversion
        self.stopper.reset()

        # TODO add restart

        # Main iteration loop
        while True:
            # update x
            self.solver_LCG.run(problem_linear)

            # update z
            A.forward(False, self.x, self.Ax)
            self.solver_ISTA.run(problem_lasso)

            # update u = (u + r) / rho
            self.compute_primal()
            self.u.add(self.primal).scale(1/self.rho)
            A.adjoint(False, self.AHr, self.primal)
            self.compute_dual()
            self.update_rho()

            # TODO save cost data, logger and all the stuff
            self.save_results(self.stopper.iter, problem, model=None, force_save=False, force_write=False)


def main():
    from sys import path
    path.insert(0, '.')
    import numpy as np
    import pyVector
    import pyOperator
    
    x = pyVector.vectorIC(np.empty((100))).set(1)
    y = x.clone() * 10
    S = pyOperator.scalingOp(x, 10)
    
    problem = ProblemLinearReg(model=x, data=y, op=S,
                               regsL1=pyOperator.IdentityOp(x), epsL1=.1)
    print('Problem built!')
    SplitBregman = SplitBregmanSolver(BasicStopper(niter=10))
    SplitBregman.setDefaults()
    SplitBregman.run(problem, verbose=True)
    
    return 0


if __name__ == '__main__':
    main()
    