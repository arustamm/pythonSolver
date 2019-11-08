project(python-solver-python_modules)


set(PythonFiles
pyParOperator.py pyStepper.py pyStopperBase.py sep_util.py pyLCGsolver.py pyProblem.py pyVPproblem.py pyStepperParabolic.py pyStepperCvSrch.py pySymLCGsolver.py sys_util.py pyOperator.py pySolver.py pyStopper.py pyVector.py pyNLCGsolver.py pyLBFGSsolver.py pyISTCsolver.py pyISTAsolver.py pyDaskOperator.py pyDaskVector.py)

install(FILES ${PythonFiles}  DESTINATION lib/python)
