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
		#Waiting until all the requested workers are up and running
		workers=0
		requested=len(hostnames)
		t0 = time.time()
		while(workers<requested):
			workers=len(self.client.get_worker_logs().keys())
			#If the number of workers is not reached in 5 minutes raise exception
			if(time.time()-t0 > 300.0): raise SystemError("ERROR! dask-ssh cannot start the requested workers within 5 minutes! Try different hostnames.")
		#Forcing deleting of object
		atexit.register(self.__del__)
		return

	def getClient(self):
		"""
		   Accessor for obtaining the client object
		"""
		return self.client

	def getWorkerIds(self):
		"""
		   Accessor for obtaining the worker IDs
		"""
		return list(self.client.get_worker_logs().keys())

	def getNworkers(self):
		"""
		   Accessor for obtaining the number of workers
		"""
		return len(self.getWorkerIds())

	def __del__(self):
		"""
		   Destructor to kill the dask-ssh running process associated with the given Client
		"""
		#killing the dask-ssh process if object is deleted
		self.dask_ssh_proc.kill()
		return
