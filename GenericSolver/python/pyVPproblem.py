#Module containing the definition of inverse problems where Variable-Projection method is used (Golub and Pereyra, 1973)
import pyProblem as pyProb
import pyOperator as pyOp


class VpOperator(pyOp.Operator):
	"""
		Operator of the form: h(m_nl)m_lin, for Variable-projection method
	"""

	def __init__(self,h_nl,h_lin,set_nl,set_lin):
		"""
			Constructor for an operator with a linear and non-linear model component
			h_nl    = [no default] - non-linear operator class; Non-linear operator class
			h_lin   = [no default] - operator class; Linear operator class
			set_nl  = [no default] - class function pointer; Class function to set non-linear part within h_lin
			set_lin = [no default] - class function pointer; Class function to set linear part within h_nl
		"""
		return



class ProblemL2VpReg(pyProb.Problem):
	"""
       Non-linear inverse problem in which part of the model parameters define a quadratic function
       The non-linear component is solved using the variable-projection method (Golub and Pereyra, 1973)
       Problem form: phi(m) = 1/2*|g(m_nl) + h(m_nl)m_lin - d|_2 + epsilon^2/2*|g'(m_nl) + h'(m_nl)m_lin - d'|_2
    """

	def __init__(self,model_nl,model_lin,h_op,data,g_op=None,g_op_reg=None,h_op_reg=None,data_reg=None):
		"""
			Constructor for solving a inverse problem using the variable-projection method
			model_nl    = [no default] - vector class; Initial non-linear model component of the objective function
			model_lin   = [no default] - vector class; Initial quadritic (Linear) model component of the objective function
			h_op   		= [no default] - Vp operator class; Variable projection operator
			data   		= [no default] - vector class; Data vector
		"""
        return
