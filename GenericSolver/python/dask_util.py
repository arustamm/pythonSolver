#Module containing useful functions to interact with the Dask module
import dask.distributed as daskD
import os
import subprocess
import random
import atexit
import time

DEVNULL = open(os.devnull,'wb')

class DaskClient:
	"""
	   Class useful to construct a Dask Client to be used with Dask vectors and operators
	"""

	def __init__(self,hostnames):
		"""
		   Constructor for obtaining a client to be used when Dask is necesary
		   hostnames = [no default] - list; list of strings containing the hostnames or IP addresses of the machines that the user wants to use in their cluster/client
		"""
		#Starting dask-ssh using the provided list of IP addresses
		self.scheduler_host = hostnames[0]
		#Random port number
		self.port = ''.join([str(random.randint(0,9)) for ii in range(4)])
		cmd = ["dask-ssh"]+hostnames+["--scheduler-port"]+[self.port]
		self.dask_ssh_proc = subprocess.Popen(cmd,stdout=DEVNULL)
		self.client = daskD.Client("tcp://"+self.scheduler_host+":"+self.port)
		#Wating a little time for the scheduler to start
		time.sleep(1.0)
		#Forcing deleting of object
		atexit.register(self.__del__)
		return

	def getClient(self):
		"""
		   Accessor for obtaining the client object
		"""
		return self.client

	def __del__(self):
		"""
		   Destructor to kill the dask-ssh running process associated with the given Client
		"""
		#killing the dask-ssh process if object is deleted
		self.dask_ssh_proc.kill()
		return
