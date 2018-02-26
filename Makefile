####################################################################################################################
#
#Created by Ettore Biondi and Guillaume Barnier to compile executables and test out-of-core python solver
####################################################################################################################
include ${SEPINC}/SEP.top

SRCDIR = ./Src
OBJDIR = ./Obj
BINDIR = ./Bin
MODDIR = ./Mod
PARDIR = ./Par
UNTEST = ./unit_tests

S = ${SRCDIR}
O = ${OBJDIR}
B = ${BINDIR}
M = ${MODDIR}
P = ${PARDIR}
U = ${UNTEST}

#Libraries to be include during the compilation of the programs
CC                  :=   icc
ICC		     =   icc
CLD                 :=   icc
OCOPTIMFLAGS         =  -O3
OCDEBUGFLAGS        :=
ifeq ($(HOST),cees-rcf.stanford.edu)
	OCDEFINES            =   -DLINUX
	OCFLAGS             :=  -O3  -c -I/opt/SEP/include
else ifeq ($(HOST),cees-rcf-tool-3)
	OCDEFINES            =   -DLINUX
	OCFLAGS             :=  -O3  -c -I/opt/SEP/include
else ifeq ($(HOST),cees-rcf-tool-4)
	OCDEFINES            =   -DLINUX
	OCFLAGS             :=  -O3  -c -I/opt/SEP/include
else
	OCDEFINES            =   -D${SEP_ARCH}
	OCFLAGS             :=  -I${SEPINCDIR}  -O3  -c
endif
OCLIBS              :=  -lsuperset -lsepaux  -lsep3d -lsep -lm

IFORTVERSION = $(shell ifort --version | grep ^ifort | sed 's/^.* //g')

#overriding F90 compiler rules
ifeq "${IFORTVERSION}" "20160415"
	F90OPTIMFLAGS   =   -xHOST -O3 -parallel
	UF90LIBS =  -lsepfftf90 -L/opt/FFTW/lib -L/opt/intel_fce_80/lib/  -lsupersetf90 -lsuperset -lsepauxf90 -lsepmathf90 -lsep3df90 -lsep3d ${GEELIB} -lsepf90 -lsep -lfftw3 -lfftw3f  -lc -qopenmp -mkl
	UF90FLAGS = -module ${M} -mkl -I/opt/FFTW/include -qopenmp
else
	F90OPTIMFLAGS   =   -O3 -parallel
	UF90LIBS =  -lsepfftf90 -L/opt/FFTW/lib -L/opt/intel_fce_80/lib/  -lsupersetf90 -lsuperset -lsepauxf90 -lsepmathf90 -lsep3df90 -lsep3d ${GEELIB} -lsepf90 -lsep -lfftw3 -lfftw3f  -lc -openmp -mkl
	UF90FLAGS = -module ${M} -mkl -I/opt/FFTW/include -openmp
endif
UF90INCLUDES	=   -I/opt/FFTW/include
OF90FLAGS 		=   -FR -I${SEPINCDIR} -c
OF90LIBS  		=   -lm -xHOST -O3

###############################################################################################################
install_rules: ./python_solver/Limit #./python_solver/Rand
	echo "add ${PWD}/python_solver to your PYTHONPATH env variable"

install:
	make -s install_rules

#Compilation
./python_solver/Limit: ${B}/Limit.x
	mv $< $@

./python_solver/Rand: ${B}/Rand.x
	mv $< $@

./python_solver/matrix%: ${B}/matrix%.x
	mv $< $@


###############################################################################################################
#Unit tests for python out-of-core solvers
LCG_test: ./python_solver/Limit ./python_solver/matrix1
	${U}/LCG.py

NLCG_test: ./python_solver/Limit ./python_solver/matrix1
	${U}/NLCG.py

#Simple lower triangular matrix test for generic linear solver
Generic_lin_test: install ./python_solver/matrix1
	./python_solver/matrix1 matrix_name=tmp_matrix.H ndata=100 nmodel=100 build_m=1 > /dev/null
	Spike n1=100 mag=0 > init_model.H
	Math file1=init_model.H exp='1.0+file1' > data.H
	./python_solver/generic_linear_prob.py fwd_cmd_file=./Par/tmp_fwd.txt adj_cmd_file=./Par/tmp_adj.txt data=data.H init_model=init_model.H dotprod=1
	./python_solver/generic_linear_prob.py fwd_cmd_file=${P}/tmp_fwd.txt adj_cmd_file=${P}/tmp_adj.txt data=data.H init_model=init_model.H inv_model=inv_model.H suffix=_problem iteration_movies=obj,model,residual niter=50 maxval=0.5 minval=0.0 log_file=Generic_lin_test.txt wrk_dir=inversion_test alpha_scale_max=100.0

#Simple lower triangular matrix plus regularization term test for generic linear solver
Generic_lin_reg_test: install ./python_solver/matrix1
	./python_solver/matrix1 matrix_name=tmp_matrix.H ndata=100 nmodel=100 build_m=1 > /dev/null
	Spike n1=100 mag=0 > init_model.H
	Math file1=init_model.H exp='1.0+file1' > data.H
	./python_solver/generic_linear_prob.py fwd_cmd_file=${P}/tmp_fwd.txt adj_cmd_file=${P}/tmp_adj.txt data=data.H init_model=init_model.H inv_model=inv_model.H log_file=Generic_lin_reg_test.txt wrk_dir=inversion_test_reg epsilon_scale=1
	./python_solver/generic_linear_prob.py fwd_cmd_file=./Par/tmp_fwd.txt adj_cmd_file=./Par/tmp_adj.txt data=data.H init_model=init_model.H epsilon=41.0 dotprod=1
	./python_solver/generic_linear_prob.py fwd_cmd_file=${P}/tmp_fwd.txt adj_cmd_file=${P}/tmp_adj.txt data=data.H init_model=init_model.H inv_model=inv_model.H suffix=_problem iteration_movies=obj,model,residual niter=10 log_file=Generic_lin_reg_test.txt wrk_dir=inversion_test_reg epsilon=41.0

#Simple symmetric matrix test for generic linear solver CG for symmetric systems on normal equations
Generic_lin_test_symmetric_normal: install ./python_solver/matrix1
	./python_solver/matrix1 matrix_name=tmp_matrix.H ndata=100 nmodel=100 build_m=1 > /dev/null
	Spike n1=100 mag=0 > init_model.H
	Math file1=init_model.H exp='1.0+file1' > data.H
	./python_solver/matrix1 matrix_name=tmp_matrix.H <data.H adj=1 > data_model.H
	./python_solver/generic_linear_prob.py fwd_cmd_file=${P}/tmp_fwd_adj.txt symmetric=y data=data_model.H init_model=init_model.H dotprod=1
	./python_solver/generic_linear_prob.py fwd_cmd_file=${P}/tmp_fwd_adj.txt symmetric=y data=data_model.H init_model=init_model.H inv_model=inv_model_normal.H suffix=_problem_sym_normal iteration_movies=obj,model,residual niter=10 log_file=Generic_lin_test_symmetric_normal.txt wrk_dir=inversion_test_sym_normal

#Simple symmetric matrix test for generic linear solver CG for symmetric systems (negative definite matrix (increasing objective function))
Generic_lin_test_symmetric: install ./python_solver/matrix2 ./python_solver/matrix1
	./python_solver/matrix2 matrix_name=tmp_matrix.H nmodel=100 build_m=1 > /dev/null
	Spike n1=100 mag=0 > init_model.H
	Math file1=init_model.H exp='1.0+file1' > data.H
	./python_solver/generic_linear_prob.py fwd_cmd_file=./Par/tmp_fwd.txt symmetric=y nmodel_ax=1 data=data.H init_model=init_model.H dotprod=1
	./python_solver/generic_linear_prob.py fwd_cmd_file=${P}/tmp_fwd.txt symmetric=y data=data.H init_model=init_model.H inv_model=inv_model.H suffix=_problem_sym iteration_movies=obj,model,residual niter=60 log_file=Generic_lin_test_symmetric.txt wrk_dir=inversion_test_sym tolobjrel=1.0


#Simple lower triangular matrix test solved using generic non-linear solver
Generic_lin_test_nonlin: install ./python_solver/matrix1
	./python_solver/matrix1 matrix_name=tmp_matrix.H ndata=100 nmodel=100 build_m=1 > /dev/null
	Spike n1=100 mag=0 > init_model.H
	Math file1=init_model.H exp='1.0+file1' > data.H
	./python_solver/generic_non_linear_prob.py fwd_nl_cmd_file=${P}/tmp_fwd.txt fwd_cmd_file=./Par/tmp_fwd.txt adj_cmd_file=./Par/tmp_adj.txt data=data.H init_model=init_model.H dotprod=1
	./python_solver/generic_non_linear_prob.py fwd_nl_cmd_file=${P}/tmp_fwd.txt fwd_cmd_file=${P}/tmp_fwd.txt adj_cmd_file=${P}/tmp_adj.txt data=data.H init_model=init_model.H inv_model=inv_model.H suffix=_problem_nonlin iteration_movies=obj,model,residual niter=50 stepper=sampler conj_method=CD log_file=Generic_lin_test_nonlin.txt wrk_dir=inversion_lin_test_nonlin alpha_scale_max=100.0

#Simple lower triangular matrix plus regularization term test solved using generic non-linear solver
Generic_lin_test_nonlin_reg: install ./python_solver/matrix1
	./python_solver/matrix1 matrix_name=tmp_matrix.H ndata=100 nmodel=100 build_m=1 > /dev/null
	Spike n1=100 mag=0.0 > init_model.H
	Spike n1=100 mag=1.0 > epsi_model.H
	Math file1=init_model.H exp='1.0+file1' > data.H
	./python_solver/generic_non_linear_prob.py fwd_nl_cmd_file=${P}/tmp_fwd.txt fwd_cmd_file=${P}/tmp_fwd.txt adj_cmd_file=${P}/tmp_adj.txt data=data.H init_model=epsi_model.H inv_model=inv_model.H log_file=Generic_lin_test_nonlin.txt wrk_dir=inversion_test epsilon_scale=1
	#Testing epsilon scale when m0 = 0.0
	Spike n1=100 mag=0.0 > epsi_model1.H
	./python_solver/generic_non_linear_prob.py fwd_nl_cmd_file=${P}/tmp_fwd.txt fwd_cmd_file=${P}/tmp_fwd.txt adj_cmd_file=${P}/tmp_adj.txt data=data.H init_model=epsi_model1.H inv_model=inv_model.H log_file=Generic_lin_test_nonlin.txt wrk_dir=inversion_test epsilon_scale=1
	./python_solver/generic_non_linear_prob.py fwd_nl_cmd_file=${P}/tmp_fwd.txt fwd_cmd_file=./Par/tmp_fwd.txt adj_cmd_file=./Par/tmp_adj.txt nmodel_ax=1 ndata_ax=1 data=data.H init_model=init_model.H epsilon=41.0 dotprod=1
	./python_solver/generic_non_linear_prob.py fwd_nl_cmd_file=${P}/tmp_fwd.txt fwd_cmd_file=${P}/tmp_fwd.txt adj_cmd_file=${P}/tmp_adj.txt data=data.H init_model=init_model.H inv_model=inv_model.H suffix=_problem iteration_movies=obj,model,residual niter=10 stepper=sampler conj_method=CD log_file=Generic_lin_test_nonlin.txt wrk_dir=inversion_test epsilon=41.0

#Simple lower triangular matrix test solved using generic L-BFGS solver
Generic_lin_test_lbfgs: install ./python_solver/matrix1
	./python_solver/matrix1 matrix_name=tmp_matrix.H ndata=100 nmodel=100 build_m=1 > /dev/null
	Spike n1=100 mag=0 > init_model.H
	Math file1=init_model.H exp='1.0+file1' > data.H
	./python_solver/generic_non_linear_prob.py fwd_nl_cmd_file=${P}/tmp_fwd.txt fwd_cmd_file=${P}/tmp_fwd.txt adj_cmd_file=${P}/tmp_adj.txt data=data.H init_model=init_model.H inv_model=inv_model_lbfgs.H suffix=_lbfgs_test iteration_movies=obj,model,residual niter=50 log_file=Generic_lin_test_lbfgs.txt wrk_dir=inversion_lbfgs_test solver=lbfgs save_estimate=1 alpha_scale_max=100.0

#Simple lower triangular matrix test solved using generic Truncated Newton solver
Generic_lin_test_truncnew: install ./python_solver/matrix1
	./python_solver/matrix1 matrix_name=tmp_matrix.H ndata=100 nmodel=100 build_m=1 > /dev/null
	Spike n1=100 mag=0 > init_model.H
	Math file1=init_model.H exp='1.0+file1' > data.H
	./python_solver/generic_non_linear_prob.py fwd_nl_cmd_file=${P}/tmp_fwd.txt fwd_cmd_file=${P}/tmp_fwd.txt adj_cmd_file=${P}/tmp_adj.txt data=data.H init_model=init_model.H inv_model=inv_model_truncnew.H suffix=_truncnew_test iteration_movies=obj,model,residual niter=20 log_file=Generic_lin_test_truncnew.txt wrk_dir=inversion_truncnew_test solver=tnewton alpha_scale_max=100.0 niter_max_lin=5 niter_min_lin=2

#Tests with Rosenbrock function
Rosenbrock_test_nlcg: install
	Spike n1=2 k1=1,2 nsp=2 mag=-1,-1> init_model_rosen.H
	Spike n1=1 mag=0.> dummy_data.H
	./python_solver/generic_non_linear_prob.py fwd_nl_cmd_file=${P}/Rosenbrock_nl.txt fwd_cmd_file=${P}/Rosenbrock_lin_fwd.txt adj_cmd_file=${P}/Rosenbrock_lin_adj.txt data=dummy_data.H init_model=init_model_rosen.H inv_model=inv_model_rosen_nlcg.H suffix=_nlcg_rosenbrock iteration_movies=obj,model,gradient niter=50 wrk_dir=nlcg_rosenbrock alpha_scale_max=100.0

Rosenbrock_test_bfgs: install
	Spike n1=2 k1=1,2 nsp=2 mag=-1,-1> init_model_rosen.H
	Spike n1=1 mag=0.> dummy_data.H
	./python_solver/generic_non_linear_prob.py fwd_nl_cmd_file=${P}/Rosenbrock_nl.txt fwd_cmd_file=${P}/Rosenbrock_lin_fwd.txt adj_cmd_file=${P}/Rosenbrock_lin_adj.txt data=dummy_data.H init_model=init_model_rosen.H inv_model=inv_model_rosen_bfgs.H suffix=_bfgs_rosenbrock iteration_movies=obj,model,gradient niter=50 wrk_dir=bfgs_rosenbrock alpha_scale_max=100.0 solver=lbfgs save_estimate=y
#Hessian estimation check
#Hessian at global solution (1,1)
#|		802    -400|
#|  -400			200|
#Inverse hessian
#|		0.5    1.0|
#|   1.0	2.005|
check_hessian_estimate.H: #Rosenbrock_test_bfgs
	Spike n1=2 k1=1,2 nsp=2 mag=1,0 > input_Hestimate.H
	./python_solver/Hessian_BFGS_apply.py estimate_dir=./bfgs_rosenbrock/hessian_vectors msteps=31 input_file=input_Hestimate.H output_file=col1_invH_rosen.H
	Spike n1=2 k1=1,2 nsp=2 mag=0,1 > input_Hestimate.H
	./python_solver/Hessian_BFGS_apply.py estimate_dir=./bfgs_rosenbrock/hessian_vectors msteps=31 input_file=input_Hestimate.H output_file=col2_invH_rosen.H
	Cat3d col1_invH_rosen.H col2_invH_rosen.H axis=2 >$@
	Rm input_Hestimate.H col1_invH_rosen.H col2_invH_rosen.H


Rosenbrock_test_tnewton: install
	Spike n1=2 k1=1,2 nsp=2 mag=-1,-1> init_model_rosen.H
	Spike n1=1 mag=0.> dummy_data.H
	./python_solver/generic_non_linear_prob.py fwd_nl_cmd_file=${P}/Rosenbrock_nl.txt fwd_cmd_file=${P}/Rosenbrock_lin_fwd.txt adj_cmd_file=${P}/Rosenbrock_lin_adj.txt data=dummy_data.H init_model=init_model_rosen.H inv_model=inv_model_rosen_tnewton.H suffix=_tnewton_rosenbrock iteration_movies=obj,model,gradient niter=50 wrk_dir=tnewton_rosenbrock solver=tnewton alpha_scale_max=100.0 niter_max_lin=5 niter_min_lin=2 H_cmd_file=${P}/Hessian_rosenbrock.txt

#Testing power method script
Power_method_test: install ./python_solver/matrix1
	./python_solver/matrix1 matrix_name=tmp_matrix.H ndata=100 nmodel=100 build_m=1 > /dev/null
	Spike n1=100 mag=0.0 > init_model.H
	./python_solver/power_method.py cmd_file=${P}/tmp_fwd_adj.txt niter=10 niter_min=500 model_vector=init_model.H max_eigenvector=maxegnvector.H min_eigenvector=minegnvector.H

###############################################################################################################
clean:
	rm -f python_solver/*.pyc
	rm -f python_solver/Limit ./python_solver/Rand python_solver/matrix1 python_solver/matrix2
	rm -f python_solver/*.pyc
	rm -f ${O}/* ${M}/* ${B}/*
	rm -f *.txt
	rm -rf inversion_test inversion_test_sym inversion_test_sym_normal inversion_test_reg inversion_lbfgs_test inversion_lin_test_nonlin nlcg_rosenbrock bfgs_rosenbrock
	Rm *.H

###############################################################################################################

include ${SEPINC}/SEP.bottom
