#Module containing the definition of an abstract, in-core, and out-of-core vectors
import numpy as np
import re
import math
import time
import os
from copy import deepcopy
from shutil import copyfile
#other modules
import sys_util
import sep_util

#regex to read output of Solver_ops
re_dpr=re.compile("DOT RESULT(.*)")

class vector:
	"""Abstract python vector class"""
	def __init__(self):
		"""Default constructor"""
		return
	
	def __del__(self):
		"""Default destructor"""
		return
	
	#Class vector operations
	
	def norm(self,N=2):
		"""Function to compute vector N-norm"""
		raise NotImplementedError("norm must be overwritten")
		return	
		
	def zero(self):
		"""Function to zero out a vector"""
		raise NotImplementedError("zero must be overwritten")
		return
		
	def scale(self,sc):
		"""Function to scale a vector"""
		raise NotImplementedError("scale must be overwritten")
		return
	
	def rand(self):
		"""Function to randomize a vector"""
		raise NotImplementedError("rand must be overwritten")
		return
		
	def clone(self):
		"""Function to clone (deep copy) a vector"""
		raise NotImplementedError("clone must be overwritten")
		return
		
	#Combination of different vectors
	
	def scaleAdd(self,vec2,sc1=1.0,sc2=1.0):
		"""Function to scale two vectors and add them to the first one"""
		raise NotImplementedError("scaleAdd must be overwritten")
		return
		
	def dot(self,vec2):
		"""Function to compute dot product between two vectors"""
		raise NotImplementedError("dot must be overwritten")
		return
	
	def multiply(self,vec2):
		"""Function to multiply element-wise two vectors"""
		raise NotImplementedError("multiply must be overwritten")
		return
	
	def isDifferent(self,vec2):
		"""Function to check if two vectors are identical"""
		raise NotImplementedError("isDifferent must be overwritten")
		return
	
class vectorIC(vector):
	"""In-core python vector class"""
	def __init__(self,arr_in):
		"""VectorIC constructor: arr=np.array
		   The naxis variable is a tuple that specifies the elements in each dimension starting from the fastest to the slowest memory wise
		   The array contained in this class are stored with C memory order (i.e., row-wise sorting)"""
		#Verify that input is a numpy array or header file or vectorOC
		if(isinstance(arr_in,vectorOC)):
			#VectorOC passed to constructor
			arr,self.ax_info = sep_util.read_file(arr_in.vecfile)
			self.arr = np.array(arr,copy=False)
		elif(isinstance(arr_in,str)):
			#Header file passed to constructor
			arr,self.ax_info = sep_util.read_file(arr_in)
			self.arr = np.array(arr,copy=False)
		elif(isinstance(arr_in,np.ndarray)):
			#Numpy array passed to constructor 
			self.arr = np.array(arr_in,copy=False)
			self.ax_info = None
		else:
			#Not supported type
			raise ValueError("ERROR! Input variable not a Numpy array or Header file or VectorOC!")
		#Number of elements per axis (tuple). Checking also the memory order
		self.naxis = self.arr.shape #If fortran the first axis is the "fastest"
		if(not np.isfortran(self.arr)): self.naxis=tuple(reversed(self.naxis)) #If C last axis is the "fastest"
		#Number of axes integer
		self.ndims = len(self.naxis)
		#Total number of elements
		self.size = self.arr.size
		return
	
	def __del__(self):
		"""VectorIC destructor"""
		del self.arr
		return
		
	def norm(self,N=2):
		"""Function to compute vector N-norm using Numpy"""
		return	np.linalg.norm(self.arr.flatten(),ord=N)
	
	def zero(self):
		"""Function to zero out a vector"""
		self.arr.fill(0)
		return
	
	def scale(self,sc):
		"""Function to scale a vector"""
		self.arr*=sc
		return
	
	def rand(self,snr=1.0):
		"""Fill vector with random number (~U[1,-1]) with a given SNR"""
		rms = np.sqrt(np.mean(np.square(self.arr)))
		amp_noise = 1.0
		if(rms != 0.): amp_noise = math.sqrt(3.0/snr)*rms #sqrt(3*Power_signal/SNR)
		self.arr = amp_noise * (2.0 * np.random.random(self.naxis) - 1.0)
		return
		
	def clone(self):
		"""Function to clone (deep copy) a vector"""
		return deepcopy(self)
		
	def scaleAdd(self,vec2,sc1=1.0,sc2=1.0):
		"""Function to scale a vector"""
		#Checking whether the input is a vector or not
		assert(isinstance(vec2,vectorIC)),"ERROR! Provided input vector not a vectorIC!"
		#Checking dimensionality
		assert(self.naxis==vec2.naxis),"ERROR! Vector dimensionality mismatching: vec1 = %s; vec2 = %s"%(self.naxis,vec2.naxis)
		#Performing scaling and addition
		self.arr=sc1*self.arr+sc2*vec2.arr
		return
		
	def dot(self,vec2):
		"""Function to compute dot product between two vectors"""
		#Checking whether the input is a vector or not
		assert(isinstance(vec2,vectorIC)),"ERROR! Provided input vector not a vectorIC!"
		#Checking size (must have same number of elements)
		assert(self.size==vec2.size),"ERROR! Vector size mismatching: vec1 = %s; vec2 = %s"%(self.size,vec2.size)
		#Checking dimensionality
		if(self.naxis!=vec2.naxis): "WARNING! Dimensionality not equal: vec1 = %s; vec2 = %s"%(self.naxis,vec2.naxis)
		return np.dot(self.arr.flatten(),vec2.arr.flatten())
		
	def multiply(self,vec2):
		"""Function to multiply element-wise two vectors"""
		#Checking whether the input is a vector or not
		assert(isinstance(vec2,vectorIC)),"ERROR! Provided input vector not a vectorIC!"
		#Checking size (must have same number of elements)
		assert(self.size==vec2.size),"ERROR! Vector size mismatching: vec1 = %s; vec2 = %s"%(self.size,vec2.size)
		#Checking dimensionality
		if(self.naxis!=vec2.naxis): "WARNING! Dimensionality not equal: vec1 = %s; vec2 = %s"%(self.naxis,vec2.naxis)
		#Performing element-wise multiplication
		self.arr=np.multiply(self.arr,vec2.arr)
		return
	
	def isDifferent(self,vec2):
		"""Function to check if two vectors are identical using built-in hash function"""
		#Checking whether the input is a vector or not
		assert(isinstance(vec2,vectorIC)),"ERROR! Provided input vector not a vectorIC!"
		#First make both array buffers read-only
		self.arr.flags.writeable = False
		vec2.arr.flags.writeable = False
		chcksum1=hash(self.arr.data)
		chcksum2=hash(vec2.arr.data)
		#Remake array buffers writable
		self.arr.flags.writeable = True
		vec2.arr.flags.writeable = True
		return (chcksum1!=chcksum2)
		
class vectorOC(vector):
	"""Out-of-core python vector class"""
	def __init__(self,input):
		"""VectorOC constructor: input= numpy array, header file, vectorIC"""
		#Verify that input is a numpy array or header file or vectorOC
		if(isinstance(input,vectorIC)):
			#VectorIC passed to constructor
			#Placing temporary file into datapath folder
			tmp_vec = sep_util.datapath+"tmp_vectorOC"+str(int(time.time()*1000000))+".H"
			sep_util.write_file(tmp_vec,input.arr,input.ax_info)
			self.vecfile = tmp_vec #Assigning internal vector array
			#Removing header file? (Default behavior is to remove temporary file)
			self.remove_file = True
		elif(isinstance(input,np.ndarray)):
			#Numpy array passed to constructor 
			tmp_vec = sep_util.datapath+"tmp_vectorOC"+str(int(time.time()*1000000))+".H"
			sep_util.write_file(tmp_vec,input)
			self.vecfile = tmp_vec #Assigning internal vector array
			#Removing header file? (Default behavior is to remove temporary file)
			self.remove_file = True
		elif(isinstance(input,str)):
			#Header file passed to constructor
			self.vecfile = input #Assigning internal vector array
			#Removing header file? (Default behavior is to preserve user file)
			self.remove_file = False
		else:
			#Not supported type
			raise ValueError("ERROR! Input variable not a Numpy array or Header file or VectorIC!")
		#Assigning binary file pointer
		self.binfile = sep_util.get_binary(self.vecfile)
		#Number of axes integer
		self.ndims = sep_util.get_num_axes(self.vecfile)
		#Number of elements per axis (tuple)
		axes_info = sep_util.get_axes(self.vecfile)
		axis_elements =tuple([ii[0] for ii in axes_info[:self.ndims]])
		self.naxis = axis_elements
		self.size = np.product(self.naxis)
		return
		
	def __del__(self):
		"""VectorIC destructor"""
		if(self.remove_file): 
			#Removing both header and binary files (using os.system to make module compatible with python3.5)
			os.system("rm -f %s %s"%(self.vecfile,self.binfile))
		return
	
	def norm(self,N=2):
		"""Function to compute vector N-norm"""
		if(N != 2): raise NotImplementedError("Norm different than L2 not currently supported")
		#Running Solver_ops to compute norm value
		find = re_dpr.search(sys_util.RunShellCmd("Solver_ops file1=%s op=dot"%(self.vecfile),get_stat=False)[0])
		if find:
			return np.sqrt(float(find.group(1)))
		else:
			raise ValueError("ERROR! Trouble parsing dot product!")
		return
		
	def zero(self):
		"""Function to zero out a vector"""
		sys_util.RunShellCmd("Solver_ops file1=%s op=zero"%(self.vecfile),get_stat=False,get_output=False)
		return

	def scale(self,sc):
		"""Function to scale a vector"""
		import sys_util
		sys_util.RunShellCmd("Solver_ops file1=%s scale1_r=%s op=scale"%(self.vecfile,sc),get_stat=False,get_output=False)
		return

	def rand(self,snr=1.0):
		"""Fill vector with random number (~U[1,-1]) with a given SNR"""
		#Computing RMS amplitude of the vector
		rms=sys_util.RunShellCmd("Attr < %s want=rms param=1 maxsize=5000"%(self.vecfile),get_stat=False)[0]
		rms=float(rms.split("=")[1]) #Standard deviation of the signal
		amp_noise = 1.0
		if(rms != 0.): amp_noise = math.sqrt(3.0/snr)*rms #sqrt(3*Power_signal/SNR)
		#Filling file with random number with the proper scale
		sys_util.RunShellCmd("Noise file=%s rep=1 type=0 var=0.3333333333; Solver_ops file1=%s scale1_r=%s op=scale"%(self.vecfile,self.vecfile,amp_noise),get_stat=False,get_output=False)
		return
	
	def clone(self):
		"""Function to clone (deep copy) a vector and creating a copy of the associated header file"""
		#First performing a deep copy of the vector
		vec_clone = deepcopy(self)
		#Creating a temporary file with similar name but computer time at the end
		tmp_vec = self.vecfile.split(".H")[0].split("/")[-1] #Getting filename only
		#Placing temporary file into datapath folder
		tmp_vec = sep_util.datapath+tmp_vec+"_clone_"+str(int(time.time()*1000000))+".H"
		tmp_bin = tmp_vec+"@"
		#Copying header and binary files and setting pointers to new file
		copyfile(self.vecfile, tmp_vec) #Copying header
		copyfile(self.binfile, tmp_bin) #Copying binary
		vec_clone.vecfile = tmp_vec
		vec_clone.binfile = tmp_bin
		#"Fixing" header file 
		with open(vec_clone.vecfile,"a") as fid:
			fid.write("in='%s'"%tmp_bin)
		#By default the clone file is going to be removed once the vector is deleted
		vec_clone.remove_file = True
		return vec_clone
		
	def scaleAdd(self,vec2,sc1=1.0,sc2=1.0):
		"""Function to scale a vector"""
		#Checking whether the input is a vector or not
		assert(isinstance(vec2,vectorOC)),"ERROR! Provided input vector not a vectorOC!"
		#Checking dimensionality
		assert(self.naxis==vec2.naxis),"ERROR! Vector dimensionality mismatching: vec1 = %s; vec2 = %s"%(self.naxis,vec2.naxis)
		#Performing scaling and addition
		cmd="Solver_ops file1=%s scale1_r=%s file2=%s scale2_r=%s op=scale_addscale"%(self.vecfile,sc1,vec2.vecfile,sc2)
		sys_util.RunShellCmd(cmd,get_stat=False,get_output=False)
		return
		
	def dot(self,vec2):
		"""Function to compute dot product between two vectors"""
		#Checking whether the input is a vector or not
		assert(isinstance(vec2,vectorOC)),"ERROR! Provided input vector not a vectorOC!"
		#Checking size (must have same number of elements)
		assert(self.size==vec2.size),"ERROR! Vector size mismatching: vec1 = %s; vec2 = %s"%(self.size,vec2.size)
		#Checking dimensionality
		if(self.naxis!=vec2.naxis): "WARNING! Dimensionality not equal: vec1 = %s; vec2 = %s"%(self.naxis,vec2.naxis)
		#Running Solver_ops to compute norm value
		cmd="Solver_ops file1=%s file2=%s op=dot"%(self.vecfile,vec2.vecfile)
		find = re_dpr.search(sys_util.RunShellCmd(cmd,get_stat=False)[0])
		if find:
			return float(find.group(1))
		else:
			raise ValueError("ERROR! Trouble parsing dot product!")
		return float(out_dot)
		
	def multiply(self,vec2):
		"""Function to multiply element-wise two vectors"""
		#Checking whether the input is a vector or not
		assert(isinstance(vec2,vectorOC)),"ERROR! Provided input vector not a vectorOC!"
		#Checking size (must have same number of elements)
		assert(self.size==vec2.size),"ERROR! Vector size mismatching: vec1 = %s; vec2 = %s"%(self.size,vec2.size)
		#Checking dimensionality
		if(self.naxis!=vec2.naxis): "WARNING! Dimensionality not equal: vec1 = %s; vec2 = %s"%(self.naxis,vec2.naxis)
		#Performing scaling and addition
		cmd="Solver_ops file1=%s file2=%s op=multiply"%(self.vecfile,vec2.vecfile)
		sys_util.RunShellCmd(cmd,get_stat=False,get_output=False)
		return
	
	def isDifferent(self,vec2):
		"""Function to check if two vectors are identical using M5 hash scheme"""
		#Checking whether the input is a vector or not
		assert(isinstance(vec2,vectorOC)),"ERROR! Provided input vector not a vectorOC!"
		hashmd5_vec1=sys_util.hashfile(self.binfile)
		hashmd5_vec2=sys_util.hashfile(vec2.binfile)
		return (hashmd5_vec1!=hashmd5_vec2)
		

	
