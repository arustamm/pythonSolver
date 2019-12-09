# Module containing the definition of inverse problems where the ADMM method is used
from pySolver import Solver
from pyLinearSolver import LCGsolver
from pySparseSolver import ISTAsolver
from pyStopperBase import BasicStopper
from pyProblem import Problem, ProblemL2LinearReg, ProblemL1Lasso
import pyOperator, pyVector


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

        # L1 Regularizations (mandatory)
        self.epsL1 = epsL1 if len(epsL1) != 0 else [.1]
        self.regsL1 = regsL1 if len(regsL1) != 0 else [pyOperator.IdentityOp(self.model)]
        self.nregsL1 = len(self.regsL1)
        self.regL1_op = pyOperator.Vstack[self.regsL1]  # for shrinkage

        # L2 Regularizations (not mandatory)
        self.regsL2 = [] if regsL2 is None else regsL2
        self.epsL2 = [] if epsL2 is None else epsL2
        self.nregsL2 = len(self.regsL2)
        if self.nregsL2 > 0:
            if dataregsL2 is None:
                self.dataregsL2 = pyVector.superVector([r2.range.clone().zero() for r2 in self.regsL2])
        else:
            self.dataregsL2 = []

        # At this point we should have:
        # - a list of L1 regularizers;
        # - a list of L1 weights (with same length of previous);
        # - a list of L2 regularizers (even empty is ok);
        # - a list of L2 weights (with same length of previous);
        # - a list of L2 dataregs (with same length of previous);

        # Last settings
        self.dres = self.res.clone()
        self.obj_terms = [None] * (1 + self.nregsL2 + self.nregsL1)
        self.res = pyVector.superVector([data.clone().zero()] * (1 + self.nregsL2 + self.nregsL1))

    def __del__(self):
        """Default destructor"""
        return

    def objective_function(self, res):
        """Compute objective function based on the residuals"""
        self.obj_terms[0] = .5 * res[0].norm() ** 2  # data fidelity

        for idx in range(self.nregsL2):
            self.obj_terms[1 + idx] = self.epsL2[idx] * res[1 + idx].norm() ** 2
        for idx in range(self.nregsL1):
            self.obj_terms[1 + self.nregsL2 + idx] = self.epsL1[idx] * res[1 + self.nregsL2 + idx].norm(1)
        return sum(self.obj_terms)


def _shrinkage(x, alpha, eps=1e-10):
    """
    Shrinkage function Gamma
        y = x / (|x| + eps) * maximum(|x| - alpha, 0)
    """
    y = x.clone()
    y / x.clone().abs().addbias(eps)
    return y * x.clone().abs().addbias(-alpha).maximum(x.clone().zero())


class SplitBregmanSolver(Solver):
    """Split-Bregman solver for L1 and L2 regularized problems"""

    # Default class methods/functions
    def __init__(self, stopper, logger=None, niter_inner=3, niter_solver=5, breg_weight=1., use_previous_solution=False):
        """
        Constructor for Split-Bregman Solver
        :param stopper              : stopper object
        :param logger               : logger object
        :param niter_inner          : int; number of iterations for the shrinkage loop [default 3]
        :param niter_solver         : int; number of iterations for the internal CG solver [default 5]
        :param breg_weight          : float; coefficient for the Bregman update b += beta * (R*x - d) [1.]
        :param use_previous_solution: bool; use the previous solution [False]
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
        self.use_previous_solution = use_previous_solution  # as initial guess for the inner problem

        self.inner_solver = LCGsolver(BasicStopper(niter=self.niter_solver), steepest=False, logger=None)
        self.inner_solver.setDefaults(iter_sampling=1, flush_memory=True)

        self.breg_coeff = None
        self.dreg_coeff = None

    def __del__(self):
        print('Destructor called, Split-Bregman deleted')

    def run(self, problem, verbose=False, restart=False, initial_guess=None):

        assert type(problem) == ProblemLinearReg, 'problem has to be a ProblemLinearReg'

        # reset stopper before running the inversion
        self.stopper.reset()

        # initialize all the vectors and operators for Split-Bregman
        self.breg_coeff = pyVector.superVector([r.range.clone().zero() for r in problem.regsL1])
        self.dreg_coeff = self.breg_coeff.clone()
        dataregsL1 = self.breg_coeff.clone()
        eps = [(e / problem.dfw)**.5 for e in problem.epsL2 + problem.epsL1]
        reg_op = pyOperator.stackOperator[problem.regsL2 + problem.regsL1]
        RL1x = problem.regsL1.range.clone()  # store RegL1 * solution

        # inner L2 reg problem
        inner_problem = ProblemL2LinearReg(
            model=problem.model,
            data=problem.data,
            op=problem.op,
            epsilon=eps,
            reg_op=reg_op,
            prior_model=pyVector.superVector(problem.dataregsL2, dataregsL1),
            minBound=problem.minBound,
            maxBound=problem.maxBound,
            boundProj=problem.boundProj
        )

        # TODO add restart and merge with initial guess
        if restart:
            pass
        else:
            sol = initial_guess if initial_guess is not None else problem.model.clone().zero()

        # Main iteration loop
        while True:
            for _ in range(self.niter_inner):

                # update dataregs for the internal problem
                dataregsL1 = self.dreg_coeff.clone() - self.breg_coeff

                if self.use_previous_solution:
                    self.problem.inner_problem.model = sol
                problem.inner_problem.setDefaults()
                self.inner_solver.run(inner_problem)

                sol = inner_problem.model
                problem.regsL1.forward(False, sol, RL1x)

                # update dreg
                self.dreg_coeff = _shrinkage(RL1x.clone() + self.breg_coeff, alpha=eps[-problem.nregsL1:])

            # update breg
            self.breg_coeff.scaleAdd(RL1x.clone() - self.dreg_coeff, 1., self.breg_weight)

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
