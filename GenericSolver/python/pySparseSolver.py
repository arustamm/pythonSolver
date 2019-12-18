from math import isnan
import numpy as np
from pySolver import Solver
from pyProblem import ProblemL1Lasso


def _soft_thresh(x, thresh):
    """
    Soft-thresholding function:
        y = sign(x) * max(abs(x) - thresh, 0)

    :param x        : vector, input values
    :param thresh   : float, soft threshold
    :return         : vector, output clipped values
    """
    return x.clone().sign() * x.clone().abs().addbias(-thresh).maximum(0.)


class ISTAsolver(Solver):
    """
    Iterative Shrikage-Thresholding Algorithm (ISTA) solver to solve:
        1/2*| y - Am |_2 + lambda*| m |_1
    """

    def __init__(self, stopper, fast=False, logger=None):
        """
        Constructor for ISTA Solver:
        :param stopper: Stopper, object to terminate inversion
        :param fast: bool, apply the Fast-ISTA [False]
        :param logger: Logger, object to write inversion log file
        """
        # Calling parent construction
        super(ISTAsolver, self).__init__()
        # Defining stopper object
        self.stopper = stopper
        # Logger object to write on log file
        self.logger = logger
        # Overwriting logger of the Stopper object
        self.stopper.logger = self.logger
        # Setting the fast flag
        self.fast = fast
        # print formatting
        self.iter_msg = "iter = %s, obj = %.5e, resnorm = %.2e, gradnorm = %.2e, feval = %d"

    def __del__(self):
        """Default destructor"""
        return

    def run(self, problem, verbose=False, restart=False):
        """Running ISTA solver"""

        self.create_msg = verbose or self.logger
        # Resetting stopper before running the inversion
        self.stopper.reset()
        # Checking if the provided problem is L1-LASSO
        if not isinstance(problem, ProblemL1Lasso):
            raise TypeError("Provided inverse problem not ProblemL1Lasso!")
        # Checking if the regularization weight was set
        if problem.lambda_value is None:
            raise ValueError("Regularization weight (lambda_value) is not set!")
        if not restart:
            if self.create_msg:
                msg = 90 * "#" + "\n"
                msg += "\t\t\t\tFAST " if self.fast else "\t\t\t\t\t"
                msg += "ITERATIVE SHRINKAGE-THRESHOLDING ALGORITHM log file\n"
                msg += "\tRestart folder: %s\n" % self.restart.restart_folder
                msg += "\tModeling Operator:\t\t%s\n" % problem.op
                msg += "\tRegularization weight:\t%.2e\n" % problem.lambda_value
                msg += 90 * "#" + "\n"
                if verbose:
                    print(msg.replace(" log file", ""))
                if self.logger:
                    self.logger.addToLog(msg)

            # Setting internal vectors (model, search direction, and previous gradient vectors)
            prblm_mdl = problem.get_model()
            ista_mdl = prblm_mdl.clone()
            # Other parameters in case FISTA is requested
            if self.fast:
                t = 1.0
                fista_mdl = prblm_mdl.clone()

            # Other internal variables
            iiter = 0
        else:
            # Retrieving parameters and vectors to restart the solver
            if self.create_msg:
                msg = "Restarting previous solver run from: %s" % self.restart.restart_folder
                if verbose:
                    print(msg)
                if self.logger:
                    self.logger.addToLog(msg)
            self.restart.read_restart()
            # Retrieving inversion parameters
            iiter = self.restart.retrieve_parameter("iter")
            initial_obj_value = self.restart.retrieve_parameter("obj_initial")
            ista_mdl = self.restart.retrieve_vector("ista_mdl")
            # Other parameters in case FISTA is requested
            if self.fast:
                t = self.restart.retrieve_parameter("t")
                fista_mdl = self.restart.retrieve_vector("fista_mdl")

        # Common variables unrelated to restart
        success = True
        ista_mdl0 = ista_mdl.clone()  # Previous model in case stepping procedure fails

        # Inversion loop
        while True:
            obj0 = problem.get_obj(ista_mdl)  # Compute objective function value
            prblm_grad = problem.get_grad(ista_mdl)  # Compute the gradient g = - A' [y - Ax]
            if iiter == 0:
                # Saving initial objective function value
                initial_obj_value = obj0
                self.restart.save_parameter("obj_initial", initial_obj_value)
                if self.create_msg:
                    msg = self.iter_msg % (str(iiter).zfill(self.stopper.zfill),
                                           obj0,
                                           problem.get_rnorm(ista_mdl),
                                           problem.get_gnorm(ista_mdl),
                                           problem.get_fevals())
                    # Writing on log file
                    if verbose:
                        print(msg)
                    if self.logger:
                        self.logger.addToLog(msg)
                # Check if either objective function value or gradient norm is NaN
                if isnan(obj0) or isnan(prblm_grad.norm()):
                    raise ValueError("Either gradient norm or objective function value NaN!")
            if problem.get_gnorm(ista_mdl) == 0.:
                print("Gradient vanishes identically")
                break

            # Saving results
            self.save_results(iiter, problem, force_save=False)

            ista_mdl0.copy(ista_mdl)  # Saving model before updating it
            if self.fast:
                # Running FISTA
                fista_mdl.scaleAdd(prblm_grad, 1.0, -1.0 / problem.op_norm)
                #########################################
                # SOFT-THRESHOLDING STEP
                # ista_mdl.copy(fista_mdl)
                # modl_arr = ista_mdl.getNdArray()
                # modl_arr[:] = _soft_thresh(modl_arr, problem.lambda_value / problem.op_norm)
                ista_mdl = _soft_thresh(fista_mdl, problem.lambda_value/problem.op_norm)

                #########################################
                # Projecting model onto the bounds (if any)
                if "bounds" in dir(problem):
                    problem.bounds.apply(ista_mdl)
                t0 = t
                t = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
                # z = x
                fista_mdl.copy(ista_mdl)
                # z = x + ((t0 - 1.) / t) * (x - xold)
                scale = (t0 - 1.0) / t
                fista_mdl.scaleAdd(ista_mdl0, 1.0 + scale, -scale)
            else:
                # Running ISTA
                ista_mdl.scaleAdd(prblm_grad, 1.0, -1.0 / problem.op_norm)  # Update model x = x + scale_precond * A' [y - Ax]
                #########################################
                # SOFT-THRESHOLDING STEP
                # modl_arr = ista_mdl.getNdArray()
                # modl_arr[:] = _soft_thresh(modl_arr, problem.lambda_value / problem.op_norm)
                ista_mdl = _soft_thresh(ista_mdl, problem.lambda_value / problem.op_norm)
                #########################################
                # Projecting model onto the bounds (if any)
                if "bounds" in dir(problem):
                    problem.bounds.apply(ista_mdl)

            obj1 = problem.get_obj(ista_mdl)
            if obj1 >= obj0:
                if self.create_msg:
                    msg = "Objective function didn't reduce, will terminate solver:\n\t" \
                          "obj_new = %.2e\tobj_cur = %.2e" % (obj1, obj0)
                    if verbose:
                        print(msg)
                    # Writing on log file
                    if self.logger:
                        self.logger.addToLog(msg)
                # Copying back to the previous solution
                ista_mdl.copy(ista_mdl0)
                break

            # Saving current model in case of restart and other parameters
            self.restart.save_parameter("iter", iiter)
            self.restart.save_vector("ista_mdl", ista_mdl)
            if self.fast:
                self.restart.save_parameter("t", t)
                self.restart.save_vector("fista_mdl", fista_mdl)

            # iteration info
            iiter = iiter + 1
            if self.create_msg:
                msg = self.iter_msg % (str(iiter).zfill(self.stopper.zfill),
                                       obj1,
                                       problem.get_rnorm(ista_mdl),
                                       problem.get_gnorm(ista_mdl),
                                       problem.get_fevals())
                if verbose:
                    print(msg)
                # Writing on log file
                if self.logger:
                    self.logger.addToLog("\n" + msg)
            # Check if either objective function value or gradient norm is NaN
            if isnan(obj1) or isnan(prblm_grad.norm()):
                raise ValueError("Either gradient norm or objective function value NaN!")
            if self.stopper.run(problem, iiter, initial_obj_value, verbose):
                break

        # Writing last inverted model
        self.save_results(iiter, problem, force_save=True, force_write=True)
        if self.create_msg:
            msg = 90 * "#" + "\n"
            msg += "\t\t\t\tFAST " if self.fast else "\t\t\t\t\t"
            msg += "ITERATIVE SHRINKAGE-THRESHOLDING ALGORITHM log file end\n"
            msg += 90 * "#" + "\n"
            if verbose:
                print(msg.replace(" log file", ""))
            if self.logger:
                self.logger.addToLog(msg)
        # Clear restart object
        self.restart.clear_restart()


class ISTCsolver(Solver):
    """ISTC solver to solve: convex problem 1/2*| y - Am |_2 + lambda*| m |_1"""

    def __init__(self, stopper, inner_it, cooling_start, cooling_end, logger=None):
        """
        Constructor for ISTC Solver
        :param stopper      : Stopper, object to terminate inversion
        :param inner_it     : int, Number of inner iterations
        :param logger       : Logger, object to write inversion log file
        :param cooling_start: float, Start of cooling continuation as fraction of size of sorted array |A'y|
        :param cooling_end  : float; End of cooling continuation as fraction of size of sorted array |A'y|
        """
        # Calling parent construction
        super(ISTCsolver, self).__init__()
        # Defining stopper object
        self.stopper = stopper
        # Logger object to write on log file
        self.logger = logger
        # Overwriting logger of the Stopper object
        self.stopper.logger = self.logger
        self.iter_msg = "Inner_iter = %s, obj = %.5e, resnorm = %.2e, gradnorm= %.2e, feval = %d"

        # ISTC parameters
        if self.stopper.niter <= 0:
            raise ValueError("niter for stopper object must be positive and greater than 0!")
        self.inner_it = inner_it  # number of inner iterations, the outer iterations are taken care by the stopper
        # cooling_start and cooling_end are numbers between 0 and 1 such that cooling_start <= cooling_end
        if not 0 <= cooling_start <= 1 or not 0 <= cooling_end <= 1 or cooling_end < cooling_start:
            raise ValueError("Cooling_start and end must be within [0,1] interval and cooling_start <= cooling_end")
        self.cooling_start = cooling_start  # start of cooling continuation as fraction of size of sorted array |A'y|
        self.cooling_end = cooling_end  # end of cooling continuation as fraction of size of sorted array |A'y|

    def __del__(self):
        """Default destructor"""
        return

    def run(self, problem, verbose=False, restart=False):
        """Running ISTC solver"""

        self.create_msg = verbose or self.logger

        # Resetting stopper before running the inversion
        self.stopper.reset()
        # Checking if the provided problem is L1-LASSO
        if not isinstance(problem, ProblemL1Lasso):
            raise TypeError("Provided inverse problem not ProblemL1Lasso!")
        # Computing preconditioning
        scale_precond = 0.99 * np.sqrt(2) / problem.op_norm  # scaling factor applied to operator A for preconditioning
        if not restart:
            if self.create_msg:
                msg = 90 * "#" + "\n"
                msg += "\t\t\tITERATIVE SOFT-THRESHOLDING WITH COOLING SOLVER log file\n"
                msg += "\tRestart folder: %s\n" % self.restart.restart_folder
                msg += "\tModeling Operator:\t\t%s\n" % problem.op
                msg += "\tRegularization weight:\t%.2e\n" % problem.lambda_value
                msg += 90 * "#" + "\n"
                if verbose:
                    print(msg.replace(" log file", ""))
                if self.logger:
                    self.logger.addToLog(msg)

            # Setting internal vectors (model, search direction, and previous gradient vectors)
            prblm_mdl = problem.get_model()
            istc_mdl = prblm_mdl.clone()

            # Inversion always starts from m = 0 (I need to understand if it is possible to start from m different than 0)
            istc_mdl.zero()  # modl = 0
            # Other internal variables
            iiter = 0
            # Computing cooling schedule for lambda values
            # istc_mdl.scale(scale_precond) #Currently unnecessary since starting model is zero
            prblm_grad = problem.get_grad(istc_mdl)
            grad_arr = np.copy(prblm_grad.getNdArray())
            grad_arr = np.abs(grad_arr.flatten())  # |A'y| and removing zero elements
            grad_arr = grad_arr[np.nonzero(grad_arr)]
            if grad_arr.size == 0:
                raise ValueError("-- A'y is returning a null vector (i.e., y in the Null space of A')")
            # Sorting the elements in descending order
            grad_arr.sort()
            grad_arr = np.flip(grad_arr, 0)
            # Setting fraction of points sampled by the outer loop (linear sampling)
            samples = np.array(np.round(np.linspace(self.cooling_start,
                                                    self.cooling_end,
                                                    self.stopper.niter)
                                        * grad_arr.size), dtype=np.uint64, copy=False)
            # Lambda values to be used during inversion for each outer loop iteration
            lambda_values = grad_arr[samples]
            # Scaling by the preconditioning
            lambda_values *= scale_precond
            # Saving the lambda values to avoid recomputation if restart is used
            self.restart.save_parameter("lambda_values", lambda_values)
        else:
            # Retrieving parameters and vectors to restart the solver
            if self.create_msg:
                msg = "Restarting previous solver run from: %s" % self.restart.restart_folder
                if verbose:
                    print(msg)
                if self.logger:
                    self.logger.addToLog(msg)
            self.restart.read_restart()
            # Retrieving lambda values and other parameters
            lambda_values = self.restart.retrieve_parameter("lambda_values")
            iiter = self.restart.retrieve_parameter("iter")
            initial_obj_value = self.restart.retrieve_parameter("obj_initial")
            istc_mdl = self.restart.retrieve_vector("istc_mdl")

        # Common variables unrelated to restart
        success = True
        istc_mdl0 = istc_mdl.clone()  # Previous model in case stepping procedure fails
        istc_mdl_save = istc_mdl0  # used also to save results

        # Outer iteration loop
        while True:
            # Setting lambda value for a given outer loop iteration
            problem.set_lambda(lambda_values[iiter])
            problem.obj_updated = False  # Lambda has been changed so objective function will change as well
            if self.create_msg:
                msg = "Outer_iter = %s\tlambda_value = %.2e" % (str(iiter).zfill(self.stopper.zfill), lambda_values[iiter])
                if verbose:
                    print(msg)
                if self.logger:
                    self.logger.addToLog(msg)
            if not restart:
                inner_iter = 0
            else:
                self.restart.retrieve_parameter("inner_iter", inner_iter)
                restart = False

            if iiter == 0:
                # Applying preconditioning
                istc_mdl.scale(scale_precond)
                obj = problem.get_obj(istc_mdl)  # Compute objective function value
                # Saving initial objective function value
                initial_obj_value = obj
                self.restart.save_parameter("obj_initial", initial_obj_value)
            while inner_iter < self.inner_it:
                obj0 = problem.get_obj(istc_mdl)  # Compute objective function value
                prblm_grad = problem.get_grad(istc_mdl)  # Compute the gradient g = - A' [y - Ax]
                if inner_iter == 0:
                    if self.create_msg:
                        msg = self.iter_msg % (str(inner_iter).zfill(self.stopper.zfill),
                                               obj0,
                                               problem.get_rnorm(istc_mdl),
                                               problem.get_gnorm(istc_mdl),
                                               problem.get_fevals())
                        # Writing on log file
                        if verbose:
                            print(msg)
                        if self.logger:
                            self.logger.addToLog(msg)
                    # Check if either objective function value or gradient norm is NaN
                    if isnan(obj0) or isnan(prblm_grad.norm()):
                        raise ValueError("Either gradient norm or objective function value NaN!")
                if problem.get_gnorm(istc_mdl) == 0.:
                    print("Gradient vanishes identically")
                    break

                # Removing preconditioning scaling factor from inverted model
                istc_mdl_save.copy(istc_mdl)
                istc_mdl_save.scale(scale_precond)
                # Saving results
                self.save_results(iiter, problem, istc_mdl_save, force_save=False)

                # Stepping for internal iteration model update
                istc_mdl0.copy(istc_mdl)  # Saving model before updating it
                istc_mdl.scaleAdd(prblm_grad, 1.0, -scale_precond)  # Update model x = x + scale_precond * A' [y - Ax]
                #########################################
                # SOFT-THRESHOLDING STEP
                istc_mdl = _soft_thresh(istc_mdl, problem.lambda_value)
                #########################################
                # Projecting model onto the bounds (if any)
                if "bounds" in dir(problem):
                    problem.bounds.apply(istc_mdl)

                obj1 = problem.get_obj(istc_mdl)
                problem.get_model().writeVec("problem_model.H")
                istc_mdl.writeVec("solver_model.H")
                if obj1 >= obj0:
                    if self.create_msg:
                        msg = "Objective function didn't reduce, will terminate solver:\n\t" \
                              "obj_new = %.2e\tobj_cur = %.2e" % (obj1, obj0)
                        if verbose:
                            print(msg)
                        # Writing on log file
                        if self.logger:
                            self.logger.addToLog(msg)
                    # Copying back to the previous solution
                    istc_mdl.copy(istc_mdl0)
                    break

                # Saving current model in case of restart and other parameters
                self.restart.save_parameter("iter", iiter)
                self.restart.save_parameter("inner_iter", inner_iter)
                self.restart.save_vector("istc_mdl", istc_mdl)

                # iteration info
                inner_iter += 1
                if self.create_msg:
                    msg = self.iter_msg % (str(inner_iter).zfill(self.stopper.zfill),
                                           obj1,
                                           problem.get_rnorm(istc_mdl),
                                           problem.get_gnorm(istc_mdl),
                                           problem.get_fevals())
                    if verbose:
                        print(msg)
                    # Writing on log file
                    if self.logger:
                        self.logger.addToLog(msg)
                # Check if either objective function value or gradient norm is NaN
                if isnan(obj1) or isnan(prblm_grad.norm()):
                    raise ValueError("Either gradient norm or objective function value NaN!")
            iiter = iiter + 1
            if self.stopper.run(problem, iiter, initial_obj_value, verbose):
                break

        # Removing preconditioning scaling factor from inverted model
        istc_mdl_save.copy(istc_mdl)
        istc_mdl_save.scale(scale_precond)
        # Writing last inverted model
        self.save_results(iiter, problem, istc_mdl_save, force_save=True, force_write=True)
        if self.create_msg:
            msg = 90 * "#" + "\n"
            msg += "\t\t\tITERATIVE SOFT-THRESHOLDING WITH COOLING SOLVER log file end\n"
            msg += 90 * "#" + "\n"
            if verbose:
                print(msg.replace(" log file", ""))
            if self.logger:
                self.logger.addToLog(msg)
        # Clear restart object
        self.restart.clear_restart()
