#module containing sep base command for file interaction
import array
import inspect
import sys,re,os,string
import struct
import math
import time
import subprocess
import string
import random

#Verifying if numpy is present
import imp
try:
    imp.find_module('numpy')
    use_numpy = True
except ImportError:
    use_numpy = False
if(use_numpy): import numpy as np

sepbin=os.environ["SEP"]+"/bin"
HOME=os.environ["HOME"]
open_files=dict()
debug=False #Debug flag for printing screen output of RunShellCmd as it runs commands
debug_file=None #File where debug outputs are written


def rand_name(N):
	"""function returning random sequence of N letters and numbers"""
	return ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits ) for _ in range(N))

def parse_args(args):
  """Function to parse command line"""
  eqs={}
  aout=[]
  eqs["basic_sep_io"]="0"
  eq=re.compile("^(.+?)=(.+)$")
  for arg in args:
    a=eq.search(arg)
    if a:
       eqs[a.group(1)]=a.group(2)
    else:
       aout.append(arg)
  return eqs,aout

def get_num_axes(file):
	"""Function to obtain number of axes in a *.H file"""
	eq=re.compile("n(\d)=(\d+)")
	cmd="%s/In %s | grep \'n[0-9]=[0-9]*\' "%(sepbin,file)
	stat1,out1=RunShellCmd(cmd)
	if(stat1!=0): assert False, "Error running get_num_axes system command: %s"%cmd
	axis_elements=[]
	for iaxis in out1.split('\n'):
		find = eq.search(iaxis)
		if find:
			axis_elements.append(find.group(2))
	index=[i for i,nelements in enumerate(axis_elements) if int(nelements) > 1]
	if index:
		n_axis=index[-1]+1
	else:
		n_axis=1
	return n_axis
	
def from_par_file(par_name,par_file):
	"""Function to parse parameter from a Par file"""
	#Check if file exist
	assert(os.path.isfile(par_file)),"Error: File %s not found!!!"%(par_file)
	par_var = ""
	eq=re.compile("^%s=(.*)$"%(par_name))
	for line in reversed(open(par_file).readlines()):
		find = eq.search(line)
		if find:
			par_var = find.group(1)
			break
	return par_var.strip()

def from_cmd_line(par_name,default="",conv=None,want_list=True):
	"""Function to parse command line"""
	#Parsing command line
	[pars,files]=parse_args(sys.argv)
	del pars["basic_sep_io"]
	par_var = ""
	
	#Reading from par file first, then check the parsed command line. Command line overwrites parameter file definition of variables
	if ("par" in pars):
		par_var = from_par_file(par_name,pars["par"])
	
	#Reading from parsed command line	
	if (par_name in pars):
		par_var = pars[par_name]
		#Removing extra spaces
		par_var = par_var.strip()
	elif(default!="" and par_var==""):
		par_var = default
	
	#Checking if requested parameter was found
	assert(par_var != ""), "Problem retrieving %s from command line"%(par_name)
	
	#Checking if the par-name is a list of parameters
	if(isinstance(par_var,str)):
		if(len(par_var.split(","))!=1 and want_list):
			par_var=par_var.split(",")
			par_var=[x for x in par_var if x!=""]
	#Conversion to different format since parse_args returns string only
	if(isinstance(par_var,list)):
		if(conv=="int" and par_var!=None):
			par_var=[int(x) for x in par_var]
		elif(conv=="float" and par_var!=None):
			par_var=[float(x) for x in par_var]
		elif(conv=="bool" and par_var!=None):
			par_var=[bool(x) for x in par_var]
	else:
		if(conv=="int" and par_var!=None):
			par_var=int(par_var)
		elif(conv=="float" and par_var!=None):
			par_var=float(par_var)
		elif(conv=="bool" and par_var!=None):
			par_var=bool(par_var)
	return par_var

def get_sep_axis_params(file, iaxis):
  """Note that iaxis is 1-based, returns a list of *strings* [n,o,d,label]."""
  stat1,out1=RunShellCmd("%s/Get parform=no <%s n%d"%(sepbin,file,iaxis))
  stat2,out2=RunShellCmd("%s/Get parform=no <%s o%d"%(sepbin,file,iaxis))
  stat3,out3=RunShellCmd("%s/Get parform=no <%s d%d"%(sepbin,file,iaxis))
  stat4,out4=RunShellCmd("%s/Get parform=no <%s label%d"%(sepbin,file,iaxis))
  if len(out1)==0: out1="1"
  if len(out2)==0: out2="0"
  if len(out3)==0: out3="1"
  if len(out4)==0: out4=" "
  return [out1, out2, out3, out4]



def get_sep_axes_params(file,par,suffix):
  """par is a dictionary (both as input and as returned value) containing keys like
  nsuffix_1, osuffix_1 etc."""
  file_size = 1
  for i in range(0,7):
    out1, out2, out3, out4 = get_sep_axis_params(file, i+1)
    if "n%s_%d"%(suffix,i+1) not in par: par["n%s_%d"%(suffix,i+1)]=out1
    if "o%s_%d"%(suffix,i+1) not in par: par["o%s_%d"%(suffix,i+1)]=out2
    if "d%s_%d"%(suffix,i+1) not in par: par["d%s_%d"%(suffix,i+1)]=out3
    if "label%s_%d"%(suffix,i+1) not in par: par["label%s_%d"%(suffix,i+1)]=out4
    file_size = file_size*int(par["n%s_%d"%(suffix,i+1)])
  par['filesize']=file_size
  stat,par['naxis']=RunShellCmd("%s/In %s| grep -P 'n+[0-9]' | wc -l"%(sepbin,file))
  return par

def par2axinfo(par,suffix):
	"""Function to convert par dictonary to ax_info array"""
	ax_info = []
	for i in range(int(par['naxis'])):
		ax_info.append([par["n%s_%d"%(suffix,i+1)],par["o%s_%d"%(suffix,i+1)],par["d%s_%d"%(suffix,i+1)],par["label%s_%d"%(suffix,i+1)]])
	return ax_info


def put_sep_axis_params(file, iaxis, ax_info):
  """Note that ax_info is a list of *strings* [n,o,d,label]."""
  assert iaxis > 0
  cmd = "echo n%d=%s o%d=%s d%d=%s label%d=%s >>%s" % (iaxis,ax_info[0], iaxis,ax_info[1], iaxis,ax_info[2], iaxis,ax_info[3], file)
  RunShellCmd(cmd)
  return

def sep_find_datapath():
	"""Function to find SEP datapath directory"""
	stat=os.path.isfile('.datapath')
	if stat:
		stat,out=RunShellCmd_nocheck("cat .datapath | head -n 1")
		datapath=out.split("=")[1]
	else:
		#check whether the local host has a datapath
		stat=os.path.isfile(HOME+"/.datapath")
		if stat:
			stat,out=RunShellCmd_nocheck("cat $HOME/.datapath | grep $HOST")
			if len(out)==0:
				stat,out=RunShellCmd_nocheck("cat $HOME/.datapath | head -n 1")
			datapath=out.split("=")[1]
		else:
			datapath=""
	return datapath

def sep_from_history(file):
	"""Function for reading sep history"""
	par = dict()
	suffix = ""
	get_sep_axes_params(file,par,suffix)
	ax_info=par2axinfo(par,suffix)
	return ax_info

def sep_to_history(file,ax_info,ax_id=0):
	"""Function for writing sep history"""
	if (ax_id==0):
		for iaxis in range(len(ax_info)):
			put_sep_axis_params(file, iaxis+1, ax_info[iaxis])
	else:
		put_sep_axis_params(file, ax_id, ax_info)
	stat_fun = 0
	return stat_fun

def sep_read_file(file,formatting='>%sf'):
	"""Function for reading sep files"""
	par = dict()
	suffix = ""
	get_sep_axes_params(file,par,suffix)
	stat,file_binary = RunShellCmd("%s/Get parform=no <%s in"%(sepbin,file))
	f = open(file_binary,'r+b')
	count=par['filesize']
	#Default formatting big-ending floating point number
	if(use_numpy):
		if(formatting=='>%sf'): formatting='>f'
		data=np.fromfile(f, dtype=formatting)
	else:
		data=struct.unpack_from(formatting%count,f.read(4*count))
	f.close()
	ax_info=par2axinfo(par,suffix)
	return [data, ax_info]

def sep_write_file(file,ax_info,data,formatting='>%sf'):
	"""Function for writing sep files"""
	datapath=sep_find_datapath()
	#write binary file
	#Default formatting big-ending floating point number
	with open(datapath+file.split('/')[-1]+'@','w+b') as f:
		if(use_numpy):
			if(formatting=='>%sf'): formatting='>f'
			data.flatten('F').astype(formatting).tofile(f)
		else:
			open_files[file].write(struct.pack(formatting % len(data), *data))
	f.close()
	#writing header/pointer file
	stat=sep_to_history(file,ax_info)
	out=RunShellCmd_output('echo in='+datapath+file.split('/')[-1]+'@'+'>>'+file)
	stat_fun=0
	return stat_fun


def sep_srite(file,data,formatting='>%sf'):
	"""Function for writing sep files: binary only"""
	datapath=sep_find_datapath()
	#write binary file
	file_binary = datapath+file.split('/')[-1]+'@'
	#Write binary pointer to header if file was not already open
	if(file not in open_files): 
		open_files[file]=open(file_binary,'w+b')
		out=RunShellCmd_output('echo in='+file_binary+'>>'+file)
	#Default formatting big-ending floating point number
	if(use_numpy):
		if(formatting=='>%sf'): formatting='>f'
		data.flatten('F').astype(formatting).tofile(open_files[file])
	else:
		
		open_files[file].write(struct.pack(formatting%len(data), *data))
	stat_fun = 0
	return stat_fun

def sep_sreed(file,count=0,formatting='>%sf'):
	"""Function for reading sep files: binary only
	It also opens the file and leave it open. To close call sep_close"""
	datapath=sep_find_datapath()
	par = dict()
	suffix = ""
	get_sep_axes_params(file,par,suffix)
	stat,file_binary = RunShellCmd("%s/Get parform=no <%s in"%(sepbin,file))
	if(file not in open_files): open_files[file]=open(file_binary,'r+b')
	#Default formatting big-ending floating point number
	if count==0: count=par['filesize']
	if(use_numpy):
		if(formatting=='>%sf'): formatting='>f'
		data=np.fromfile(open_files[file], dtype=formatting, count=count)
	else:
		data=struct.unpack_from(formatting%count,open_files[file].read(4*count))
	return data


def sep_sseek(file,off,whence):
	"""Function to move file pointer. Same as lseek (whence: beg=0, cur=1, end=2)"""
	global open_files
	if("file" not in open_files):
		print("%s is not open"%(file))
	else:
		open_files[file].seek(off,whence)
	stat_fun = 0
	return stat_fun

def sep_close(file=""):
	global open_files
	if bool(open_files):
		if file=="":
			for filename in open_files:
				if(not(open_files[filename].closed)):open_files[filename].close()
				open_files=dict()
		else:
			if(file in open_files):
				if(not(open_files[file].closed)):open_files[file].close()
				open_files.pop(file)
			else:
				print("%s not open" %(file))
	return

def write_in_file(log_file=None,info=None):
		"""Function to write log file output of a command"""
		#info: line to print in the log file
		if (log_file != None and debug):
			with open(log_file,"a") as f:
				f.write(info+"\n")
		return

def wait_process(process,print_output=False):
	"""Waiting a process to finish and catching stdout during process run"""
	out=""
	while (process.poll() == None):
		for line in iter(process.stdout.readline,b''):
			info=line.decode("ascii")
			out+=info
			if print_output:
				write_in_file(debug_file,info.rstrip())
				print(info.rstrip())
				sys.stdout.flush()
			process.stdout.flush()
	return out.strip()

def RunShellCmd(cmd, print_cmd=False, print_output=False, synch=True):
  if debug:
    print_cmd=True
    print_output=True #Overwrites any previous definition
  if print_cmd:
    info = "RunShellCmd: %s"%cmd
    write_in_file(debug_file,info); print(info)
  process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)
  if(synch):
    out=wait_process(process,print_output)
    stat1 = process.poll()
    if stat1 != 0:
      info = "CmdError: %s" % process.stderr.read()
      write_in_file(debug_file,info); print(info)
      info = "Shell cmd Failed: %s"%cmd
      write_in_file(debug_file,info); assert False, info
    return stat1,out
  else:
    return None,"Running command asynchronously"

#Same as RunShellCmd but returns the output
def RunShellCmd_output(cmd, print_cmd=False, print_output=False, synch=True):
  if print_cmd:
    print("RunShellCmd: %s"%cmd)
  stat1,out1=RunShellCmd(cmd, synch=synch)
  if(synch):
    if stat1 != 0:
      assert False, "Shell cmd Failed: %s"%cmd
    if print_output:
      print("CmdOutput: %s"%out1)
    return out1
  else:
    return "Running command asynchronously"

def RunShellCmd_nocheck(cmd, print_cmd=False, print_output=False, synch=True):
  if print_cmd:
    print("RunShellCmd: %s"%cmd)
  process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)
  if(synch):
    out1=wait_process(process,print_output)
    stat1 = process.poll()
    if print_output:
      print("CmdOutput: %s"%out1)
    return stat1,out1
  else:
    return None,"Running command asynchronously"

#Out of core simple operations
def Dot(file1,file2,output):
	"""Dot product between two files or two file lists"""
	islist1=isinstance(file1,list)
	islist2=isinstance(file2,list)
	if(islist1 and islist2):
		if(len(file1)!=len(file2)): assert False, 'Error provide lists with same number of elements: file1=%s file2=%s'%(file1,file2)
		dot=0.0
		for  ifile,jfile in zip(file1,file2):
			cmd="Solver_ops file1=%s file2=%s op=dot |& colrm 1 11"%(ifile,jfile)
			out=RunShellCmd_output(cmd)
			dot+=float(out)
	elif(islist1 and not islist2 or not islist1 and islist2):
		assert False, 'Error provide lists or file names: file1=%s file2=%s'%(file1,file2)
	else:
		cmd="Solver_ops file1=%s file2=%s op=dot |& colrm 1 11"%(file1,file2)
		out=RunShellCmd_output(cmd)
		dot=float(out)
	cmd="Spike n1=1 d1=1 | Scale rscale=%s >%s out=%s@"%(dot,output,output)
	RunShellCmd(cmd)
	return

def Dot_incore(file1,file2):
	"""Dot product between two files or two file lists returning a float"""
	islist1=isinstance(file1,list)
	islist2=isinstance(file2,list)
	if(islist1 and islist2):
		if(len(file1)!=len(file2)): assert False, 'Error provide lists with same number of elements: file1=%s file2=%s'%(file1,file2)
		dot=0.0
		for  ifile,jfile in zip(file1,file2):
			cmd="Solver_ops file1=%s file2=%s op=dot |& colrm 1 11"%(ifile,jfile)
			out=RunShellCmd_output(cmd)
			dot+=float(out)
	elif(islist1 and not islist2 or not islist1 and islist2):
		assert False, 'Error provide lists or file names: file1=%s file2=%s'%(file1,file2)
	else:
		cmd="Solver_ops file1=%s file2=%s op=dot |& colrm 1 11"%(file1,file2)
		out=RunShellCmd_output(cmd)
		dot=float(out)
	return dot

def Norm(file,output):
	"""L2 Norm of a given file or file list"""
	islist=isinstance(file,list)
	if(islist):
		norm=0.0
		for ifile in file:
			cmd="Solver_ops file1=%s op=dot |& colrm 1 11"%(ifile)
			out=RunShellCmd_output(cmd)
			norm+=float(out)
	else:
		cmd="Solver_ops file1=%s op=dot |& colrm 1 11"%(file)
		out=RunShellCmd_output(cmd)
		norm=float(out)
	norm=math.sqrt(float(norm))
	cmd="Spike n1=1 d1=1 | Scale rscale=%s >%s out=%s@"%(norm,output,output)
	RunShellCmd(cmd)
	return

def Norm_incore(file):
	"""L2 Norm of a given file or file list returning a float"""
	islist=isinstance(file,list)
	if(islist):
		norm=0.0
		for ifile in file:
			cmd="Solver_ops file1=%s op=dot |& colrm 1 11"%(ifile)
			out=RunShellCmd_output(cmd)
			norm+=float(out)
	else:
		cmd="Solver_ops file1=%s op=dot |& colrm 1 11"%(file)
		out=RunShellCmd_output(cmd)
		norm=float(out)
	norm=math.sqrt(float(norm))
	return norm

def Is_different(file1,file2):
	"""Check whether file1 and file2 are the same (checking the maximum of the absolute value of the difference)"""
	islist1=isinstance(file1,list)
	islist2=isinstance(file2,list)
	mean=0.0
	if(islist1 and islist2):
		if(len(file1)!=len(file2)): assert False, 'Error provide lists with same number of elements: file1=%s file2=%s'%(file1,file2)
		for ifile,jfile in zip(file1,file2):
			if(ifile!=jfile):
				cmd="Math file1=%s file2=%s exp='@ABS(file1-file2)' | Attr want=max param=1 | Get parform=n maxval"%(ifile,jfile)
				out=RunShellCmd_output(cmd)
				mean+=float(out)
	elif(islist1 and not islist2 or not islist1 and islist2):
		assert False, 'Error provide lists or file names: file1=%s file2=%s'%(file1,file2)
	else:
		if(file1!=file2):
			cmd="Math file1=%s file2=%s exp='@ABS(file1-file2)' | Attr want=max param=1 | Get parform=n maxval"%(file1,file2)
			out=RunShellCmd_output(cmd)
			mean=float(out)
	if (mean==0):
		different = False
	else:
		different = True
	return different

def Scale(file,scalar):
	"""Multiply file or file list by a scalar"""
	islist=isinstance(file,list)
	if(islist):
		for ifile in file:
			cmd="Solver_ops file1=%s scale1_r=%s op=scale"%(ifile,scalar)
			RunShellCmd(cmd)
	else:
		cmd="Solver_ops file1=%s scale1_r=%s op=scale"%(file,scalar)
		RunShellCmd(cmd)
	return

def Sum(file1,file2,scale1=1.0,scale2=1.0):
	"""Add two files or two file lists with scalar multiplication (overwrites file1)"""
	islist1=isinstance(file1,list)
	islist2=isinstance(file2,list)
	if(islist1 and islist2):
		if(len(file1)!=len(file2)): assert False, 'Error provide lists with same number of elements: file1=%s file2=%s'%(file1,file2)
		for ifile,jfile in zip(file1,file2):
			cmd="Solver_ops file1=%s scale1_r=%s file2=%s scale2_r=%s op=scale_addscale"%(ifile,scale1,jfile,scale2)
			RunShellCmd(cmd)
	elif(islist1 and not islist2 or not islist1 and islist2):
		assert False, 'Error provide lists or file names: file1=%s file2=%s'%(file1,file2)
	else:
		cmd="Solver_ops file1=%s scale1_r=%s file2=%s scale2_r=%s op=scale_addscale"%(file1,scale1,file2,scale2)
		RunShellCmd(cmd)
	return

def Add(file1,file2,fileout,scale1=1.0,scale2=1.0):
	"""Add two files or two file lists with scalar multiplication and place the result in fileout"""
	islist1=isinstance(file1,list)
	islist2=isinstance(file2,list)
	islistout=isinstance(fileout,list)
	if(islist1 and islist2 and islistout):
		if(len(file1)!=len(file2)!=len(fileout)): assert False, 'Error provide lists with same number of elements: file1=%s file2=%s fileout=%s'%(file1,file2,fileout)
		for ifile,jfile,ifileout in zip(file1,file2,fileout):
			cmd="Add %s %s scale=%s,%s >%s"%(ifile,jfile,scale1,scale2,ifileout)
			RunShellCmd(cmd)
	elif(islist1 and not islist2 or not islist1 and islist2):
		assert False, 'Error provide lists or file names: file1=%s file2=%s'%(file1,file2)
	elif(islist1 and islist2 and not islistout):
		assert False, 'Error provide lists for fileout: fileout=%s'%(fileout)
	elif(not (islist1 and islist2) and islistout):
		assert False, 'Error provide only file name for fileout: fileout=%s'%(fileout)
	else:
		cmd="Add %s %s scale=%s,%s >%s"%(file1,file2,scale1,scale2,fileout)
		RunShellCmd(cmd)
	return

def Zero(file):
	"""Set file or file list to zeros"""
	islist=isinstance(file,list)
	if(islist):
		for ifile in file:
			cmd="Solver_ops file1=%s op=zero"%(ifile)
			RunShellCmd(cmd)
	else:
		cmd="Solver_ops file1=%s op=zero"%(file)
		RunShellCmd(cmd)
	return


def Cp(file1,file2):
	"""Copy file1 into file2"""
	islist1=isinstance(file1,list)
	islist2=isinstance(file2,list)
	if(islist1 and islist2):
		if(len(file1)!=len(file2)): assert False, 'Error provide lists with same number of elements: file1=%s file2=%s'%(file1,file2)
		for ifile,jfile in zip(file1,file2):
			#If file names are equal it won't copy the file
			if(ifile!=jfile):
				cmd="Cp %s %s"%(ifile,jfile)
				RunShellCmd(cmd)
	elif(islist1 and not islist2 or not islist1 and islist2):
		assert False, 'Error provide lists or file names: file1=%s file2=%s'%(file1,file2)
	else:
		#If file names are equal it won't copy the file
		if(file1!=file2):
			cmd="Cp %s %s"%(file1,file2)
			RunShellCmd(cmd)
	return

def Rm(file):
	"""Remove file"""
	islist=isinstance(file,list)
	if(islist):
		for ifile in file:
# 			cmd="Rm %s"%(ifile)
# 			RunShellCmd(cmd)
			rm_sep_file(ifile)
	else:
# 		cmd="Rm %s"%(file)
# 		RunShellCmd(cmd)
		rm_sep_file(file)
	return

def rm_sep_file(file):
	cmd1="rm -f `Get <%s in parform=n`"%(file)
	cmd2="rm -f %s"%(file)
	RunShellCmd(cmd1)
	RunShellCmd(cmd2)
	return

def tmp_file(file):
	"""Function to return a file name with current time appended"""
	islist=isinstance(file,list)
	if(islist): assert False, "Function tmp_file does not support file list: %s"%(file)
	datapath=sep_find_datapath()
	current_time=str(int(time.time()*1000000))
	file_name=datapath+file+current_time
	return file_name

def Get_value(file,nelements=1):
	"""Function to obtain file values"""
	islist=isinstance(file,list)
	if(islist): assert False, "Function Get_value does not support file list: %s"%(file)
	cmd="Disfil < %s number=n count=%s col=1 type=f format=%%50.45e "%(file,nelements)
	out=RunShellCmd_output(cmd)
	if nelements==1:
		value=float(out)
	else:
		value=[]
		value=[0]*nelements
		outsplit=out.split("\n")
		for ii in range(0,nelements):
			value[ii]=float(outsplit[ii])
	return value

def Rand(file,snr=1.0):
	"""Fill file with random number (~U[1,-1]) with a given SNR"""
	islist=isinstance(file,list)
	if(islist):
		for ifile in file:
			stat,rms=RunShellCmd("Attr < %s want=rms param=1"%(ifile))
			rms=float(rms.split("=")[1]) #Standard deviation of the signal
			if(rms==0.):
				amp_noise=1.0
			else:
				amp_noise=math.sqrt(3.0/snr)*rms #sqrt(3*Power_signal/SNR)
			cmd="Noise file=%s rep=1 type=0 var=0.3333333333; Solver_ops file1=%s scale1_r=%s op=scale"%(ifile,ifile,amp_noise)
			RunShellCmd(cmd)
	else:
		stat,rms=RunShellCmd("Attr < %s want=rms param=1"%(file))
		rms=float(rms.split("=")[1]) #Standard deviation of the signal
		if(rms==0.):
			amp_noise=1.0
		else:
			amp_noise=math.sqrt(3.0/snr)*rms #sqrt(3*Power_signal/SNR)
		cmd="Noise file=%s rep=1 type=0 var=0.3333333333 >/dev/null; Solver_ops file1=%s scale1_r=%s op=scale"%(file,file,amp_noise)
		RunShellCmd(cmd)
	return
