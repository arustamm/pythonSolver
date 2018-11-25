project(python-solver-python_moudles)



set(PythonFiles
pyParOperator.py  pyStepper.py    pyStopperBase.py  sep_util.py
pyLCGsolver.py    pyProblem.py    pyStepperParabolic.py pySymLCGsolver.py sys_util.py
pyOperator.py   pySolver.py   pyStopper.py    pyVector.py   vector.py)


install(FILES ${PythonFiles}  DESTINATION lib/python)
