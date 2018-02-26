#!/usr/bin/python
#module containing class for operator
import sep_python as sep
import copy

def combine_operators(op1,op2):
	"""Function to combine operator 1 to operator 2 (Op2*Op1)"""
	#Copying the two operators (one is going to be the combined operator)
	comb_operator=copy.deepcopy(op1)
	op2_deepcopy=copy.deepcopy(op2)
	Operator.operator_list.append(op2_deepcopy.name) #Keep the right instantiated operator names
	#Changing operator name
	comb_operator_name="combined operator [Op2*Op1]: "+op2.name+"*"+op1.name
	comb_operator.name=comb_operator_name
	#Combining operators by a common temp file
	temp_comb_filename=sep.tmp_file("comb_operator_op2op1_tmpfile.H")
	#Op1 input = temp_comb_filename
	comb_operator.set_input_output(output=temp_comb_filename)
	#Op2 temp_comb_filename = output
	op2_deepcopy.set_input_output(input=temp_comb_filename)
	comb_operator.cmd+=op2_deepcopy.cmd
	#Removing temporary file
	comb_operator.cmd.append("rm -f %s"%(temp_comb_filename))
	comb_operator.cmd.append("rm -f `Get <%s in parform=n`"%(temp_comb_filename))
	#Setting output file as output file of second operator
	comb_operator.output=op2_deepcopy.output
	#Adding Combined operator to list
	Operator.operator_list.append(comb_operator_name)
	del op2_deepcopy
	return comb_operator

def print_operators():
	if (len(Operator.operator_list)==0):
		print "No Operator currently instantiated"
	else:
		print "Operators currently instantiated:",Operator.operator_list
	return

class Operator:
	operator_list=[]

	def __init__(self,op_name,cmd_file,input_file,output_file,input_m0_file=None,input_aux=[]):
		"""Constructor for Operator object
			input parameters => op_name,cmd_file,input_file,output_file,input_m0_file (necessary for non-linear)
		"""
		self.name=op_name
		self.cmd_file=cmd_file
		self.input=input_file
		self.input_m0=input_m0_file
		self.output=output_file
		self.input_aux=input_aux #other auxiliary files
		self.command_read()
		Operator.operator_list.append(self.name)
		#Setting random names if tmp_rand_name[0-9]+ are used
		self.set_random_names()
		return
		
	def set_random_names(self):
		"""Method to change tmp_rand_name[0-9]+ to random temporary name"""
		ind=0
		input_name="tmp_random_name%s"
		while any(input_name%ind in s for s in self.cmd):
			rand_input_name=sep.rand_name(12)
			for ii in range(0,len(self.cmd)):
				self.cmd[ii]=self.cmd[ii].replace(input_name%ind,rand_input_name)
			ind+=1
		return

	def run(self,print_cmd=False,print_output=False):
		"""Method to run operator"""
		for cmd_to_run in self.cmd:
			sep.RunShellCmd(cmd_to_run,print_cmd,print_output)
		#Cleaning strange files called xyza.* (Not sure if this program is generating them)
		sep.RunShellCmd("rm -f %s"%(sep.sep_find_datapath())+"xyza.*")
		stat=0
		return stat

	def set_input_output(self,input=None,output=None,input_m0=None,input_aux=[]):
		"""Method to change name of operator input and/or output"""
		if(input!=None):
			#Changing input file name
			for ii in range(0,len(self.cmd)):
				self.cmd[ii]=self.cmd[ii].replace(self.input,input)
			self.input=input
		if(output!=None):
			#Changing output file name
			for ii in range(0,len(self.cmd)):
				self.cmd[ii]=self.cmd[ii].replace(self.output,output)
			self.output=output
		if((input_m0!=None) and (self.input_m0!=None)):
			#Changing input_m0 file name
			for ii in range(0,len(self.cmd)):
				self.cmd[ii]=self.cmd[ii].replace(self.input_m0,input_m0)
			self.input_m0=input_m0
		elif((input_m0!=None) and (self.input_m0==None)):
			assert False, "Changing input_m0 without inizitializing it"
		assert (isinstance(input_aux,list)), "Provived input_aux not a list: %s"%(input_aux)
		for idx, ifile in enumerate(input_aux):
			#Changing input_aux[idx] file name if index smaller than internal list length
			if(idx<len(self.input_aux)):
				for ii in range(0,len(self.cmd)):
					self.cmd[ii]=self.cmd[ii].replace(self.input_aux[idx],ifile)
				self.input_aux[idx]=ifile
			else:
				print "WARNING! %s outside of internal list %s"%(ifile,self.input_aux)
		return

	def print_commands(self):
		"""Method to print commands in the operator"""
		for cmd_to_run in self.cmd:
			print cmd_to_run
		return

	def __del__(self):
		"""Standard destructor"""
		if(self.name in Operator.operator_list): Operator.operator_list.remove(self.name)
		return

	def command_read(self):
		self.cmd=[]
		with open(self.cmd_file) as file:
			for line in file:
				if(line[0:4]=="RUN:"):
					cmd_tmp=line.split("RUN:")[1]
					cmd_tmp=cmd_tmp.replace('\r','')
					self.cmd.append(cmd_tmp.strip('\n'))
		return
