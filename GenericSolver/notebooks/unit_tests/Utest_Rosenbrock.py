#!/usr/bin/env python3.5
import sys
sys.path.append("/net/server/homes/sep/ettore/research/packages/pySolver/GenericSolver/python")
import pyNLCGsolver as NLCG
import pyLBFGSsolver as LBFGS
import pyVector as Vec
import pyOperator as Op
import pyProblem as Prblm
import pyStopperBase as Stopper
import numpy as np
from sys_util import logger
#Plotting library
import matplotlib.pyplot as plt




class Rosenbrock_prblm(Prblm.Problem):
	"""
	   Rosenbrock function inverse problem
	   f(x,y) = (1 - x)^2 + 100*(y -x^2)^2
	   m = [x y]'
	   res = objective function value
	"""

	def __init__(self,x_initial,y_initial):
		"""Constructor of linear problem"""
		#Setting the bounds (if any)
		super(Rosenbrock_prblm,self).__init__(None,None)
		self.model  = Vec.vectorIC(np.array((x_initial,y_initial)))
		self.dmodel = self.model.clone()
		self.dmodel.zero()
		#Gradient vector
		self.grad=self.dmodel.clone()
		#Residual vector
		self.res = Vec.vectorIC(np.array((0.,)))
		#Dresidual vector
		self.dres=self.res.clone()
		#Setting default variables
		self.setDefaults()
		self.linear=True
		return

	def objf(self,model):
		"""Objective function computation"""
		m = model.arr #Getting ndArray of the model
		obj = self.res.arr[0]
		return obj

	def resf(self,model):
		"""Residual function"""
		m = model.arr #Getting ndArray of the model
		self.res.arr[0] = (1.0 - m[0])*(1.0 - m[0]) + 100.0 * (m[1] - m[0]*m[0]) * (m[1] - m[0]*m[0])
		return self.res

	def gradf(self,model,res):
		"""Gradient computation"""
		m = model.arr #Getting ndArray of the model
		self.grad.arr[0] = - 2.0 * (1.0 - m[0]) - 400.0 * m[0] * (m[1] - m[0]*m[0])
		self.grad.arr[1] = 200.0 * (m[1] - m[0]*m[0])
		return self.grad

	def dresf(self,model,dmodel):
		"""Linear variation of the objective function"""
		m = model.arr #Getting ndArray of the model
		dm = dmodel.arr #Getting ndArray of the model
		self.dres.arr[0] = (- 2.0 * (1.0 - m[0]) - 400.0 * m[0] * (m[1] - m[0]*m[0]))* dm[0] + (200.0 * (m[1] - m[0]*m[0])) * dm[1]
		return self.dres


if __name__ == '__main__':
	x_init = -1.0
	y_init = -1.0
	#Testing solver on Rosenbrock function
	Ros_prob = Rosenbrock_prblm(x_init,y_init)
	#Create stopper
	niter = 200
	Stop  = Stopper.BasicStopper(niter=niter,tolr=1e-32,tolg=1e-32)
	#Create solver
	NLCGsolver = NLCG.NLCGsolver(Stop,logger=logger("Rosenbrock_NLCG_log.txt"))
	NLCGsolver.setDefaults(save_obj=True,save_model=True)
	NLCGsolver.run(Ros_prob,verbose=True)
	print("optimal NLCG x: ", Ros_prob.model.arr[0])
	print("optimal NLCG y: ", Ros_prob.model.arr[1])
	# plt.plot(NLCGsolver.obj)
	# plt.show()

	#Testing Steepest-descent method
	Ros_prob = Rosenbrock_prblm(x_init,y_init)
	NLSDsolver = NLCG.NLCGsolver(Stop,beta_type="SD",logger=logger("Rosenbrock_NLSD_log.txt"))
	NLSDsolver.run(Ros_prob)
	print("optimal NLSD x: ", Ros_prob.model.arr[0])
	print("optimal NLSD y: ", Ros_prob.model.arr[1])


	#Testing BFGS algorithm
	Ros_prob = Rosenbrock_prblm(x_init,y_init)
	BFGSsolver = LBFGS.LBFGSsolver(Stop,logger=logger("Rosenbrock_BFGS_log.txt"))
	BFGSsolver.run(Ros_prob,verbose=True)
	print("optimal BFGS x: ", Ros_prob.model.arr[0])
	print("optimal BFGS y: ", Ros_prob.model.arr[1])

	#Testing LBFGS algorithm
	Ros_prob = Rosenbrock_prblm(x_init,y_init)
	LBFGSsolver = LBFGS.LBFGSsolver(Stop,m_steps=1,logger=logger("Rosenbrock_LBFGS_log.txt"))
	LBFGSsolver.run(Ros_prob,verbose=True)
	print("optimal LBFGS x: ", Ros_prob.model.arr[0])
	print("optimal LBFGS y: ", Ros_prob.model.arr[1])




























#
