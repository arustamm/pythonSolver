project(python-solver-python_moudles)


set(PythonFiles1
pyParOperator.py pyStepper.py pyStopperBase.py  sep_util.py  pyLCGsolver.py pyProblem.py pyStepperParabolic.py pySymLCGsolver.py sys_util.py pyOperator.py)
set(PythonFiles2
pySolver.py pyStopper.py pyVector.py pyNLCGsolver.py pyLBFGSsolver.py pyISTCsolver.py pyISTAsolver.py)

install(FILES ${PythonFiles1}  DESTINATION lib/python)
install(FILES ${PythonFiles2}  DESTINATION lib/python)
