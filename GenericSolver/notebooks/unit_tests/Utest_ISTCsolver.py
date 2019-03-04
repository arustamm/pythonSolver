#!/usr/bin/env python3
import sys,os
sys.path.insert(0, "/net/server/homes/sep/ettore/research/packages/pySolver/GenericSolver/python")
sys.path.insert(0, "/net/server/homes/sep/ettore/research/packages/acoustic_isotropic_operators/local/lib/python")
import genericIO
SepVector = genericIO.SepVector
import pyLCGsolver as LCG
import pySymLCGsolver as SymLCGsolver
import pyProblem as Prblm
import pyStopperBase as Stopper
from sys_util import logger
import sep_util as sep
import numpy as np

import pyISTCsolver as ISTC
from Gaussian_smoothing import Gauss_smooth_scipy as Gauss_smooth
from spatialDerivModule import LaplacianPython

if __name__ == '__main__':
	true_model = SepVector.getSepVector(ns=[301,601])
	true_model_arr = true_model.getNdArray()
	#Adding spikes to the model
	true_model_arr[300,150] = 10.0
	true_model_arr[200,100] = -5.0
	true_model_arr[400,280] = 1.0
	# true_model.writeVec("true_model_spike.H")
	#Instantiating operator
	sigmax = 300.0
	sigmaz = 200.0
	Gauss_op = Gauss_smooth(true_model,sigmax,sigmaz)
	#Generating data
	data = true_model.clone()
	Gauss_op.forward(False,true_model,data)
	Lapla_op = LaplacianPython(true_model,true_model,0)
	genericIO.defaultIO.writeVector("gauss.H",data)
	# data.writeVec("data_spike.H")
	quit()

	Gauss_op.dotTest(True)
	Lapla_op.dotTest(True)
	Gauss_op.powerMethod(True,square=True)
	####################################################
	#L2-norm inversions
	#Create stopper
	niter = 1000
	Stop  = Stopper.BasicStopper(niter=niter)
	#Create solver
	LCGsolver = LCG.LCGsolver(Stop)
	LCGsolver.setDefaults()
	#Create L2-norm linear problem
	initial_model = true_model.clone()
	initial_model.zero()
	L2Prob = Prblm.ProblemL2Linear(initial_model,data,Gauss_op)
	# LCGsolver.run(L2Prob,verbose=True)
	# L2Prob.model.writeVec("inverted_model_L2.H")
	#
	L2ProbReg = Prblm.ProblemL2LinearReg(initial_model,data,Gauss_op,0.1,Lapla_op)
	L2ProbReg.estimate_epsilon(True)
	# LCGsolver.run(L2ProbReg,verbose=True)
	# L2ProbReg.model.writeVec("inverted_model_L2_Reg.H")

	#Running using symmetric problem
	# SymProb = Prblm.ProblemLinearSymmetric(initial_model,data,Gauss_op)
	# SLCG = SymLCGsolver.SymLCGsolver(Stop)
	# SLCG.run(SymProb,verbose=True)
	# SymProb.model.writeVec("inverted_model_L2_Sym.H")























#
