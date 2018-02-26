#module containing solver class and functions
import sys
import operator
import math
import sep_python as sep

class solver:
	"""Solver parent object"""
	
	def __init__(self):
		"""Dummy constructor."""
		return
	  
	  
	def run(self):
		"""Dummy solver running routine"""
		assert False, "!Implement run for solver in the derived class."
		return


class stepper:
	"""Stepper parent object"""
	
	def __init__(self):
		"""Dummy constructor."""
		return

	def run(self):
		"""Dummy stepper running routine"""
		assert False, "!Implement run solver in the derived class."
		return

class problem:
	"""Problem parent object"""
	#Internal variables
	__model=[]
	__grad=[]
	__dmodel=[]
	__res=[]
	__dres=[]
	__obj_updated=False
	__res_updated=False
	__grad_updated=False
	__dres_updated=False
	__fevals=0
	__nmodel=0
	__ndata=0
	__obj=0
	__counter=0
	#Output file names
	__obj_movie=""
	__model_movie=""
	__grad_movie=""
	__res_movie=""
	
	def __init__(self,n_model,n_data):
		"""Simple constructor."""
		#internal arrays
		self.__model=[0]*n_model
		self.__grad=[0]*n_model
		self.__dmodel=[0]*n_model
		self.__res=[0]*n_data
		self.__dres=[0]*n_data
		#other internal variables
		self.__nmodel=n_model
		self.__ndata=n_data
		return
		
	def set_movie(self,obj_movie="",model_movie="",grad_movie="",res_movie=""):
		self.__obj_movie = obj_movie
		self.__model_movie = model_movie
		self.__grad_movie = grad_movie
		self.__res_movie = res_movie
		return
		
	def set_model(self,model):
		if(not(self.__model==model)):
			self.__model=model
			self.__obj_updated=False
			self.__res_updated=False
			self.__grad_updated=False
			self.__dres_updated=False
		return
		
	def get_obj(self,model):
		if(len(model)!=len(self.__model)): sys.exit("Model size does not match: get_obj")
		self.set_model(model)
		if(not(self.__obj_updated)):
			self.__res = self.get_res(self.__model)
			self.__obj = self.objf(self.__res)
			self.__obj_updated=True
		obj=self.__obj
		return obj
		
	def get_res(self,model):
		if(len(model)!=len(self.__model)): sys.exit("Model size does not match: get_res")
		self.set_model(model)
		if(len(res)!=len(self.__res)): sys.exit("Residual size does not match: get_res")
		if(not(self.__res_updated)):
			self.__res = self.resf(self.__model)
			self.__fevals += 1
			self.__res_updated=True
		res=self.__res
		return res
		
	def get_grad(self,model):
		if(len(model)!=len(self.__model)): sys.exit("Model size does not match: get_grad")
		self.set_model(model)
		if(len(grad)!=len(self.__grad)): sys.exit("Gradient size does not match: get_grad")
		if(not(self.__grad_updated)):
			self.__res  = self.get_res(self.__model)
			self.__grad = self.gradf(self.__model,self.__res)
			self.__fevals += 2
			self.__grad_updated=True
		grad=self.__grad
		return grad
		
	def get_dres(self,model,dmodel,dres):
		if(len(model)!=len(self.__model) or len(dmodel)!=len(self.__dmodel)): sys.exit("Model or Dmodel size does not match: get_dres")
		self.set_model(model)
		if(len(dres)!=len(self.__dres)): sys.exit("Dres size does not match: get_dres")
		if(not(self.__dres_updated) or not(dmodel==self.__dmodel)):
			self.__dmodel = dmodel
			self.__dres = self.get_res(self.__model,self.__dmodel)
			self.__fevals += 2
			self.__dres_updated=True
		dres=self.__res
		return
	
	def get_fevals(self):
		fevals = self.__fevals
		return fevals

	def get_rnorm(self):
		rnorm=math.sqrt(sum(map(operator.mul,self.__res,self.__res)))
		return rnorm
		
	def get_gnorm(self):
		gnorm=math.sqrt(sum(map(operator.mul,self.__grad,self.__grad)))
		return gnorm
		
	def output(self,model):
		self.__counter += 1 
		if(self.__obj_movie!=""):
			stat=sep.sep_to_history(self.__obj_movie,[self.__counter,0,1,"iteration"],1)
			stat=sep.sep_srite(self.__obj_movie,self.__obj)
		if(self.__model_movie!=""):
			stat=sep.sep_to_history(self.__model_movie,[self.__nmodel,0,1,"Model index"],1)
			stat=sep.sep_to_history(self.__model_movie,[self.__counter,0,1,"iteration"],2)
			stat=sep.sep_srite(self.__model_movie,model)
		if(self.__grad_movie!=""):
			stat=sep.sep_to_history(self.__grad_movie,[self.__nmodel,0,1,"Gradient index"],1)
			stat=sep.sep_to_history(self.__grad_movie,[self.__counter,0,1,"iteration"],2)
			stat=sep.sep_srite(self.__grad_movie,self.__grad)
		if(self.__res_movie!=""):
			stat=sep.sep_to_history(self.__res_movie,[self.__ndata,0,1,"Residual index"],1)
			stat=sep.sep_to_history(self.__res_movie,[self.__counter,0,1,"iteration"],2)
			stat=sep.sep_srite(self.__res_movie,self.__res)
		return
		
	def objf(self,res):
		"""Dummy objf running routine"""
		assert False, "!Implement objf for problem in the derived class."
		return obj
		
	def resf(self,model):
		"""Dummy resf running routine"""
		assert False, "!Implement resf for problem in the derived class."
		return res
		
	def gradf(self,model,res):
		"""Dummy gradf running routine"""
		assert False, "!Implement gradf for problem in the derived class."
		return grad
	
	
	
















