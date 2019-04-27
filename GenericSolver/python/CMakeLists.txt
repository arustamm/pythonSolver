project(python-solver-python_modules)


set(PythonFiles1
pyParOperator.py pyStepper.py pyStopperBase.py sep_util.py pyLCGsolver.py pyProblem.py pyVPproblem.py pyStepperParabolic.py pyStepperCvSrch.py pySymLCGsolver.py sys_util.py pyOperator.py pySolver.py pyStopper.py pyVector.py pyNLCGsolver.py pyLBFGSsolver.py pyISTCsolver.py pyISTAsolver.py)

install(FILES ${PythonFiles}  DESTINATION lib/python)
