#Module containing Non-Linear Conjugate-Gradient Solver class
#It can also handle steppest-descent algorithm
from math import isnan
import pySolver

def betaFR(grad,grad0,dir,logger):
	"""Fletcher and Reeves method"""
	#betaFR = sum(dprod(g,g))/sum(dprod(g0,g0))
	dot_grad=grad.dot(grad)
	dot_grad0=grad0.dot(grad0)
	if (dot_grad0 == 0.): #Avoid division by zero
		beta = 0.
		if(logger): logger.addToLog("Setting beta to zero since norm of previous gradient is zero!!!")
	else:
		beta = dot_grad/dot_grad0
	return beta

def betaPRP(grad,grad0,dir,logger):
	"""Polak, Ribiere, Polyak method"""
	#betaPRP = sum(dprod(g,g-g0))/sum(dprod(g0,g0))
	tmp1=grad.clone()
	#g-g0
	tmp1.scaleAdd(grad0,1.0,-1.0)
	dot_num=tmp1.dot(grad)
	dot_grad0=grad0.dot(grad0)
	if (dot_grad0 == 0.): #Avoid division by zero
		beta = 0.
		if(logger): logger.addToLog("Setting beta to zero since norm of previous gradient is zero!!!")
	else:
		beta = dot_num/dot_grad0
	return beta

def betaHS(grad,grad0,dir,logger):
	"""Hestenes and Stiefel"""
	#betaHS = sum(dprod(g,g-g0))/sum(dprod(d,g-g0))
	tmp1=grad.clone()
	#g-g0
	tmp1.scaleAdd(grad0,1.0,-1.0)
	dot_num=tmp1.dot(grad)
	dot_denom=tmp1.dot(dir)
	if (dot_denom == 0.): #Avoid division by zero
		beta = 0.
		if(logger): logger.addToLog("Setting beta to zero since norm of denominator is zero!!!")
	else:
		beta = dot_num/dot_denom
	return beta

def betaCD(grad,grad0,dir,logger):
	"""Conjugate Descent"""
	#betaCD = -sum(dprod(g,g))/sum(dprod(d,g0))
	dot_num=grad.dot(grad)
	dot_denom=-grad0.dot(dir)
	if (dot_denom == 0.): #Avoid division by zero
		beta = 0.
		if(logger): logger.addToLog("Setting beta to zero since norm of denominator is zero!!!")
	else:
		beta = dot_num/dot_denom
	return beta

def betaLS(grad,grad0,dir,logger):
	"""Liu and Storey"""
	#betaLS = -sum(dprod(g,g-g0))/sum(dprod(d,g0))
	tmp1=grad.clone()
	#g-g0
	tmp1.scaleAdd(grad0,1.0,-1.0)
	dot_num=tmp1.dot(grad)
	dot_denom=-grad0.dot(dir)
	if (dot_denom == 0.): #Avoid division by zero
		beta = 0.
		if(logger): logger.addToLog("Setting beta to zero since norm of denominator is zero!!!")
	else:
		beta = dot_num/dot_denom
	return beta

def betaDY(grad,grad0,dir,logger):
	"""Dai and Yuan"""
	#betaDY = sum(dprod(g,g))/sum(dprod(d,g-g0))
	tmp1=grad.clone()
	#g-g0
	tmp1.scaleAdd(grad0,1.0,-1.0)
	dot_num=grad.dot(grad)
	dot_denom=tmp1.dot(dir)
	if (dot_denom == 0.): #Avoid division by zero
		beta = 0.
		if(logger): logger.addToLog("Setting beta to zero since norm of denominator is zero!!!")
	else:
		beta = dot_num/dot_denom
	return beta

def betaBAN(grad,grad0,dir,logger):
	"""Bamigbola, Ali and Nwaeze"""
	#betaDY = sum(dprod(g,g-g0))/sum(dprod(g0,g-g0))
	tmp1=grad.clone()
	#g-g0
	tmp1.scaleAdd(grad0,1.0,-1.0)
	dot_num=tmp1.dot(grad)
	dot_denom=tmp1.dot(grad0)
	if (dot_denom == 0.): #Avoid division by zero
		beta = 0.
		if(logger): logger.addToLog("Setting beta to zero since norm of denominator is zero!!!")
	else:
		beta = -dot_num/dot_denom
	return beta

def betaHZ(grad,grad0,dir,logger):
	"""Hager and Zhang"""
	#betaN = sum(dprod(g-g0-2*sum(dprod(g-g0,g-g0))*d/sum(dprod(d,g-g0)),g))/sum(dprod(d,g-g0))
	tmp1=grad.clone()
	#g-g0
	tmp1.scaleAdd(grad0,1.0,-1.0)
	#sum(dprod(g-g0,g-g0))
	dot_diff_g_g0=tmp1.dot(tmp1)
	#sum(dprod(d,g-g0))
	dot_dir_diff_g_g0=tmp1.dot(dir)
	if (dot_dir_diff_g_g0 == 0.): #Avoid division by zero
		beta = 0.
		if(logger): logger.addToLog("Setting beta to zero since norm of denominator is zero!!!")
	else:
		#g-g0-2*sum(dprod(g-g0,g-g0))*d/sum(dprod(d,g-g0))
		tmp1.scaleAdd(dir,1.0,-2.0*dot_diff_g_g0/dot_dir_diff_g_g0)
		#sum(dprod(g-g0-2*sum(dprod(g-g0,g-g0))*d/sum(dprod(d,g-g0)),g))
		dot_num=grad.dot(tmp1)
		#dot_num/sum(dprod(d,g-g0))
		beta = dot_num/dot_dir_diff_g_g0
	return beta

def betaST(grad,grad0,dir,logger):
	"""Steepest descent"""
	beta = 0.
	return beta

#Beta functions
#grad=new gradient, grad0=old, dir=search direction
#From A SURVEY OF NONLINEAR CONJUGATE GRADIENT METHODS

class NLCGsolver(pySolver.Solver):
	"""Non-Linear Conjugate Gradient and Steepest-Descent Solver parent object"""

    #Default class methods/functions
	def __init__(self,stoppr,beta_type="FR",logger=None):
		"""Constructor for LCG Solver"""
		#Defining stopper object
		self.stoppr=stoppr
		#Beta function to use during the inversion
		self.beta_type=beta_type
		#Logger object to write on log file
		self.logger=logger
		#Overwriting logger of the Stopper object
		self.stoppr.logger=self.logger
		#Setting defaults for saving results
		self.setDefaults()
		return


    def beta_func(self,grad,grad0,dir):
        """Beta function interface"""
        beta_type = self.beta_type
        if(beta_type == "FR"):
            beta = betaFR(grad,grad0,dir,self.logger)
        elif(beta_type == "PRP"):
            beta = betaPRP(grad,grad0,dir,self.logger)
		elif(beta_type == "HS"):
            beta = betaHS(grad,grad0,dir,self.logger)
		elif(beta_type == "CD"):
            beta = betaCD(grad,grad0,dir,self.logger)
		elif(beta_type == "LS"):
            beta = betaLS(grad,grad0,dir,self.logger)
		elif(beta_type == "DY"):
            beta = betaDY(grad,grad0,dir,self.logger)
		elif(beta_type == "BAN"):
            beta = betaBAN(grad,grad0,dir,self.logger)
		elif(beta_type == "HZ"):
            beta = betaHZ(grad,grad0,dir,self.logger)
		elif(beta_type == "ST"):
            beta = betaST(grad,grad0,dir,self.logger)
        else:
            raise ValueError("ERROR! Requested Beta function type not existing")
        return beta
