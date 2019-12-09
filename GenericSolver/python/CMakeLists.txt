project(python-solver-python_modules)


set(PythonFiles
pyParOperator.py pyStepper.py sep_util.py pyProblem.py pyVPproblem.py sys_util.py pyOperator.py pySolver.py pyLinearSolver.py pyNonLinearSolver.py pySparseSolver.py pyStopper.py pyVector.py pyDaskOperator.py pyDaskVector.py dask_util.py)

install(FILES ${PythonFiles}  DESTINATION lib/python)
