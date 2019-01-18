#!/usr/bin/env python3
import sys,os
sys.path.append(os.environ.get('REPOSITORY')+"/python_solver/python_modules")
import numpy as np
import pyOperator as Op
import pyProblem as Prblm
import pyStopperBase as Stopper
import pyLCGsolver as LCG
import pyVector

class MatMult_SepVector(Op.Operator):
	"""Operator class to perform matrix-vector multiplication"""

	def __init__(self,A,domain,range):
		"""Constructor for the class: A = matrix to use; domain = domain vector; range = range vector"""
		if(not isinstance(domain,pyVector.vectorSEP)): raise TypeError("ERROR! Domain vector not a Vector object")
		if(not isinstance(range,pyVector.vectorSEP)): raise TypeError("ERROR! Range vector not a Vector object")
		#Setting domain and range of operator and matrix to use during application of the operator
		self.setDomainRange(domain,range)
		self.A = np.matrix(A)
		return

	def forward(self,add,model,data):
		"""Method to compute d = A m"""
		self.checkDomainRange(model,data)
		if(not isinstance(model,pyVector.vectorSEP)): raise TypeError("ERROR! Model vector not a Vector object")
		if(not isinstance(data,pyVector.vectorSEP)): raise TypeError("ERROR! Data vector not a Vector object")
		if(not add): data.zero()
		#Converting to numpy arrays
		data_np=np.array(data.vec.getCpp(),copy=False)
		model_np=np.array(model.vec.getCpp(),copy=False)
		data_np+=np.matmul(A,model_np)
		return

	def adjoint(self,add,model,data):
		"""Method to compute m = A d"""
		self.checkDomainRange(model,data)
		if(not isinstance(model,pyVector.vectorSEP)): raise TypeError("ERROR! Model vector not a Vector object")
		if(not isinstance(data,pyVector.vectorSEP)): raise TypeError("ERROR! Data vector not a Vector object")
		if(not add): model.zero()
		#Converting to numpy arrays
		data_np=np.array(data.vec.getCpp(),copy=False)
		model_np=np.array(model.vec.getCpp(),copy=False)
		model_np+=np.matmul(A.H,data_np)
		return



if __name__ == '__main__':
	#Create stopper
	niter = 10000
	Stop  = Stopper.BasicStopper(niter=niter)
	#Create solver
	LCGsolver = LCG.LCGsolver(Stop)
	#Create a sepVector
	model=pyVector.vectorSEP((1,200))
	data=pyVector.vectorSEP((1,200))
	n=200
	A = np.matrix(np.zeros((n,n),dtype=np.float64))
	np.fill_diagonal(A, -2)
	np.fill_diagonal(A[1:], 1)
	np.fill_diagonal(A[:,1:], 1)
	#Create operator
	MatMultSym = MatMult_SepVector(A,model,data)
	#Testing operator
	model.rand()
	print(model.norm())
	MatMultSym.forward(False,model,data)
	MatMultSym.adjoint(False,model,data)
	print(data.norm())
	print(model.norm())
	#Testing solver
	data_np = np.array(data.vec,copy=False)
	data_np.fill(1.)
	model.zero()
	#Create L2-norm linear problem
	L2Prob_sym = Prblm.ProblemL2Linear(model,data,MatMultSym)
# 	LCGsolver.setDefaults(iter_sampling=500)
	LCGsolver.run(L2Prob_sym)
