#!/usr/bin/env python3
import sys,os
sys.path.insert(0, "/net/server/homes/sep/ettore/research/packages/pySolver/GenericSolver/python")
import pyVector as Vec
import pyOperator as Op
import pyLCGsolver as LCG
import pySymLCGsolver as SymLCGsolver
import pyProblem as Prblm
import pyStopperBase as Stopper
from sys_util import logger
import sep_util as sep
import numpy as np

#Testing the NLCG to solver a regularized linear problem treated as if it was non linear
import pyNLCGsolver as NLCG
import pyLBFGSsolver as BFGS

class MatMult_incore(Op.Operator):
	"""Operator class to perform matrix-vector multiplication"""

	def __init__(self,A,domain,range):
		"""Constructor for the class: A = matrix to use; domain = domain vector; range = range vector"""
		if(not isinstance(domain,Vec.vector)): raise TypeError("ERROR! Domain vector not a vector object")
		if(not isinstance(range,Vec.vector)): raise TypeError("ERROR! Range vector not a vector object")
		#Setting domain and range of operator and matrix to use during application of the operator
		self.setDomainRange(domain,range)
		self.A = np.matrix(A)
		return

	def forward(self,add,model,data):
		"""Method to compute d = A m"""
		self.checkDomainRange(model,data)
		if(not isinstance(model,Vec.vectorIC)): raise TypeError("ERROR! Model vector not a vectorIC object")
		if(not isinstance(data,Vec.vectorIC)): raise TypeError("ERROR! Data vector not a vectorIC object")
		if(not add): data.zero()
		data.arr+=np.matmul(self.A,model.arr)
		return

	def adjoint(self,add,model,data):
		"""Method to compute m = A d"""
		self.checkDomainRange(model,data)
		if(not isinstance(model,Vec.vectorIC)): raise TypeError("ERROR! Model vector not a vectorIC object")
		if(not isinstance(data,Vec.vectorIC)): raise TypeError("ERROR! Data vector not a vectorIC object")
		if(not add): model.zero()
		model.arr+=np.matmul(self.A.H,data.arr)
		return

class MatMult_outcore(Op.Operator):
	"""Operator class to perform matrix-vector multiplication"""

	def __init__(self,A,domain,range):
		"""Constructor for the class: A = matrix to use; domain = domain vector; range = range vector"""
		if(not isinstance(domain,Vec.vector)): raise TypeError("ERROR! Domain vector not a vector object")
		if(not isinstance(range,Vec.vector)): raise TypeError("ERROR! Range vector not a vector object")
		#Setting domain and range of operator and matrix to use during application of the operator
		self.setDomainRange(domain,range)
		self.A = np.matrix(A)
		return

	def forward(self,add,model,data):
		"""Method to compute d = A m"""
		self.checkDomainRange(model,data)
		if(not isinstance(model,Vec.vectorOC)): raise TypeError("ERROR! Model vector not a vectorOC object")
		if(not isinstance(data,Vec.vectorOC)): raise TypeError("ERROR! Data vector not a vectorOC object")
		if(not add): data.zero()
		#Reading model and data vector files
		[model_arr,_]=sep.read_file(model.vecfile)
		[data_arr,data_axis]=sep.read_file(data.vecfile)
		data_arr+=np.matmul(self.A,model_arr)
		#writing data vector file
		sep.write_file(data.vecfile,data_arr,data_axis)
		return

	def adjoint(self,add,model,data):
		"""Method to compute m = A d"""
		self.checkDomainRange(model,data)
		if(not isinstance(model,Vec.vectorOC)): raise TypeError("ERROR! Model vector not a vectorOC object")
		if(not isinstance(data,Vec.vectorOC)): raise TypeError("ERROR! Data vector not a vectorOC object")
		if(not add): model.zero()
		#Reading model and data vector files
		[model_arr,model_axis]=sep.read_file(model.vecfile)
		[data_arr,_]=sep.read_file(data.vecfile)
		model_arr+=np.matmul(self.A.H,data_arr)
		#writing data vector file
		sep.write_file(model.vecfile,model_arr,model_axis)
		return

#Function necessary to construct a non-linear operator out of a linear one
def dummy_func(dummy_arg):
	return

if __name__ == '__main__':
	#In-core run
	#Creating model vector
	model_vec = Vec.vectorIC(np.zeros((100,1)))
	model_vec.zero()
	#Creating data vector
	data_vec  = Vec.vectorIC(np.zeros((200,1)))
	data_vec.rand()
	#Matrix to be inverted
	A = np.matrix(np.random.rand(200,100))
	#Create operator
	MatMult = MatMult_incore(A,model_vec,data_vec)
	#Create L2-norm linear problem
	L2Prob = Prblm.ProblemL2Linear(model_vec,data_vec,MatMult)
	#Create stopper
	niter = 2000
	Stop  = Stopper.BasicStopper(niter=niter)
	#Create solver
	LCGsolver = LCG.LCGsolver(Stop)
	LCGsolver.setDefaults(iter_sampling=10)
	#Running the solver
	# LCGsolver.run(L2Prob,verbose=True)

	#Out-of-core run
	#Creating model vector
	model_vecOC = Vec.vectorOC(model_vec)
	#Creating data vector
	data_vecOC  = Vec.vectorOC(data_vec)
	#Create operator
	MatMultOC = MatMult_outcore(A,model_vecOC,data_vecOC)
	#Create L2-norm linear problem
	L2Prob_outcore = Prblm.ProblemL2Linear(model_vecOC,data_vecOC,MatMultOC)

	#Running the solver
	LCGsolver.setDefaults()
	# LCGsolver.run(L2Prob_outcore)

	#Testing inversion of a symmetric matrix (second-order derivative operator)
	n=200
	A = np.matrix(np.zeros((n,n),dtype=np.float64))
	np.fill_diagonal(A, -2)
	np.fill_diagonal(A[1:], 1)
	np.fill_diagonal(A[:,1:], 1)
	model_vec_sym = Vec.vectorIC(np.zeros((n,1),dtype=np.float64))
	data_vec_sym = Vec.vectorIC(np.zeros((n,1),dtype=np.float64))
	#Constant derivative
	data_vec_sym.arr.fill(1.)
	#Create operator
	MatMultSym = MatMult_incore(A,model_vec_sym,data_vec_sym)
	#Create L2-norm linear problem
	L2Prob_sym = Prblm.ProblemL2Linear(model_vec_sym,data_vec_sym,MatMultSym)
	#Running the solver
	LCGsolver.setDefaults(iter_buffer_size=None,iter_sampling=100)
	# LCGsolver.run(L2Prob_sym)


	#Testing LCG with regularized problem
	L2Prob_reg = Prblm.ProblemL2LinearReg(model_vec_sym,data_vec_sym,MatMultSym,0.0001)
	# L2Prob_reg.estimate_epsilon(True)
	#Running the solver
	LCGsolver.setDefaults(iter_sampling=100,save_obj=True,prefix="lin_test")
	# LCGsolver.run(L2Prob_reg,verbose=True)

	#Testing LCG for symmetric systems
	low_bound = model_vec_sym.clone()
	low_bound.set(-2000.)
	SymProb = Prblm.ProblemLinearSymmetric(model_vec_sym,data_vec_sym,MatMultSym,minBound=low_bound)
	SLCG = SymLCGsolver.SymLCGsolver(Stop)
	SLCG.setDefaults(iter_sampling=5,save_obj=True,save_res=True,save_grad=True,save_model=True,prefix="test")
	# SLCG.run(SymProb,verbose=True)
	# print(SymProb.model.arr)

	#Testing Linear steepest-descent algorithm for symmetric systems
	SymProb1 = Prblm.ProblemLinearSymmetric(model_vec_sym,data_vec_sym,MatMultSym)
	SLSD = SymLCGsolver.SymLCGsolver(Stop,steepest=True)
	SLSD.setDefaults(iter_sampling=100)
	# SLSD.run(SymProb1)

	#Testing non-linear regularized problem
	non_lin_op = Op.NonLinearOperator(MatMultSym,MatMultSym,dummy_func)
	L2NLRegProb = Prblm.ProblemL2NonLinearReg(model_vec_sym,data_vec_sym,non_lin_op,0.)
	L2NLRegProb.estimate_epsilon()
	NLCGsolver = NLCG.NLCGsolver(Stop)
	NLCGsolver.setDefaults(iter_sampling=5,save_obj=True,save_res=True,save_grad=True,save_model=True,prefix="test")
	# NLCGsolver.run(L2NLRegProb,verbose=True)


	#Testing non-linear bounded problem with NLCG
	L2NLProb = Prblm.ProblemL2NonLinear(model_vec_sym,data_vec_sym,non_lin_op,minBound=low_bound)
	NLCGsolver = NLCG.NLCGsolver(Stop)
	# NLCGsolver.run(L2NLProb,verbose=False)
	# print(L2NLProb.model.arr)

	#Testing non-linear bounded problem with BFGS
	L2NLProb = Prblm.ProblemL2NonLinear(model_vec_sym,data_vec_sym,non_lin_op,minBound=low_bound)
	BFGSsolver = BFGS.LBFGSsolver(Stop)
	# BFGSsolver.run(L2NLProb,verbose=True)
	# print(L2NLProb.model.arr)

	#Bounded problem
	#Creating the bounds
	model_vec_sym.zero()
	#Create L2-norm linear problem
	L2Prob_sym = Prblm.ProblemL2Linear(model_vec_sym,data_vec_sym,MatMultSym,minBound=low_bound)
	# L2Prob_sym = Prblm.ProblemL2Linear(model_vec_sym,data_vec_sym,MatMultSym)
	#Running the solver
	# LCGsolver.run(L2Prob_sym,verbose=False)
	# print(L2Prob_sym.model.arr)












#
