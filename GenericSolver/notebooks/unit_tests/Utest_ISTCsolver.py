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

import pyISTCsolver as ISTC
from Gaussian_smoothing import Gauss_smooth

if __name__ == '__main__':
	true_model = Vec.vectorIC((301,601))
	true_model_arr = true_model.getNdArray()
	#Adding spikes to the model
	true_model_arr[300,150] = 10.0
	true_model_arr[200,100] = -5.0
	true_model_arr[400,280] = 1.0
	true_model.writeVec("true_model_spike.H")
	#Instantiating operator
	sigmax = 2.0
	sigmaz = 1.0
	Gauss_op = Gauss_smooth(true_model,sigmax,sigmaz)
	#Generating data
	data = true_model.clone()
	Gauss_op.forward(False,true_model,data)
	data.writeVec("data_spike.H")























#
