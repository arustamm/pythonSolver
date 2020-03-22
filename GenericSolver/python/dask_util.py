# Module containing useful functions to interact with the Dask module
import atexit
import random
import socket
import subprocess
import os
import time
import json

DEVNULL = open(os.devnull, 'wb')
import dask.distributed as daskD
from dask_jobqueue import PBSCluster


def get_tcp_info(filename):
    """Function to obtain scheduler tcp information"""
    tcp_info = None
    with open(filename) as json_file:
        data = json.load(json_file)
    if "address" in data:
        tcp_info = data["address"]
    return tcp_info


def create_hostnames(machine_names, Nworkers):
    """
	Function to create hostnames variables (i.e., list of ip addresses)
	from machine names and number of wokers per machine
	"""
    
    ip_adds = []
    for host in machine_names:
        ip_adds.append(socket.gethostbyname(host))
    
    if len(Nworkers) != len(ip_adds):
        raise ValueError("Lenght of number of workers (%s) not consistent with number of machines available (%s)"
                         % (len(Nworkers), len(ip_adds)))
    
    hostnames = []
    for idx, ip in enumerate(ip_adds):
        hostnames += [ip] * Nworkers[idx]
    return hostnames


class DaskClient:
    """
	Class useful to construct a Dask Client to be used with Dask vectors and operators
	"""
    
    def __init__(self, **kwargs):
        """
		Constructor for obtaining a client to be used when Dask is necesary
		1) Cluster with shared file system and ssh capability:
		:param hostnames : - list; list of strings containing the host names or IP addresses of the machines that
			the user wants to use in their cluster/client (First hostname will be running the scheduler!) [None]
		:param scheduler_file_prefix : string; prefix to used to create dask scheduler-file.
			Must be a mounted path on all the machines. Necessary if hostnames are provided [$HOME/scheduler-]
		2) PBS cluster:
		:param pbs_params : - dict; dictionary containing PBS Cluster options (see help(PBSCluster) for help) [None]
		:param n_workers : - int; number of workers to be submitted to the cluster or to be used on each job (if n_jobs is provided)
		:param n_jobs : - int; number of jobs to be submitted to the cluster; if provided you will have n_workers per job (i.e., n_workers*n_jobs = dask_workers) [None]
		"""
        hostnames = kwargs.get("hostnames", None)
        pbs_params = kwargs.get("pbs_params", None)
        # Checking interface to be used
        if hostnames:
            if not isinstance(hostnames, list):
                raise ValueError("User must provide a list with host names")
            scheduler_file_prefix = kwargs.get("scheduler_file_prefix", os.path.expanduser("~") + "/scheduler-")
            # Random port number
            self.port = ''.join(["1"] + [str(random.randint(0, 9)) for _ in range(3)])
            # Starting scheduler
            scheduler_file = "%s%s" % (scheduler_file_prefix, self.port) + ".json"
            cmd = ["ssh"] + [hostnames[0]] + \
                  ["dask-scheduler"] + ["--scheduler-file"] + [scheduler_file] + \
                  ["--port"] + [self.port]
            self.scheduler_proc = subprocess.Popen(cmd, stdout=DEVNULL, stderr=DEVNULL)
            # Checking if scheduler has started and getting tpc information
            t0 = time.time()
            while True:
                if os.path.isfile(scheduler_file):
                    if get_tcp_info(scheduler_file): break
                # If the dask scheduler is not started in 5 minutes raise exception
                if time.time() - t0 > 300.0:
                    raise SystemError("Dask could not start scheduler! Try different first host name.")
            # Creating dask Client
            self.client = daskD.Client(scheduler_file=scheduler_file)
            # Starting workers on all the other hosts
            self.worker_procs = []
            worker_ips = []
            for hostname in hostnames:
                cmd = ["ssh"] + [hostname] + ["dask-worker"] + ["--scheduler-file"] + [scheduler_file]
                # Starting worker
                self.worker_procs.append(subprocess.Popen(cmd, stdout=DEVNULL, stderr=DEVNULL))
                # Obtaining IP address of host for the started worker (necessary to resort workers)
                worker_ips.append(
                    subprocess.check_output(
                        ["ssh"] + [hostname] + ["hostname -I"] + ["| awk '{print $1}'"]).rstrip().decode("utf-8"))
            # Waiting until all the requested workers are up and running
            workers = 0
            requested = len(hostnames)
            t0 = time.time()
            while workers < requested:
                workers = len(self.client.get_worker_logs().keys())
                # If the number of workers is not reached in 5 minutes raise exception
                if time.time() - t0 > 300.0:
                    raise SystemError(
                        "Dask could not start the requested workers within 5 minutes! Try different hostnames.")
            # Resorting worker IDs according to user-provided list
            self.WorkerIds = []
            wrkIds = list(self.client.get_worker_logs().keys())  # Unsorted workers ids
            wrk_ips = [idw.split(":")[1][2:] for idw in wrkIds]  # Unsorted ip addresses
            for ip in worker_ips:
                idx = wrk_ips.index(ip)
                self.WorkerIds.append(wrkIds[idx])
                wrkIds.pop(idx)
                wrk_ips.pop(idx)
        elif pbs_params:
            n_workers = kwargs.get("n_workers", 0)
            n_jobs = kwargs.get("n_jobs", None)
            if n_workers <= 0:
                raise ValueError("n_workers must equal or greater than 1!")
            # if n_jobs is provided then start n_workers on n_jobs
            if n_jobs:
                if n_jobs <= 0:
                    raise ValueError("n_jobs must equal or greater than 1!")
                pbs_params.update({"processes": n_workers})
                if n_workers > 1:
                    # forcing nanny to be true (otherwise, dask-worker command will fail)
                    pbs_params.update({"nanny": True})
                n_workers *= n_jobs
            else:
                n_jobs = n_workers
            self.cluster = PBSCluster(**pbs_params)
            self.cluster.scale(jobs=n_jobs)
            # Creating dask Client
            self.client = daskD.Client(self.cluster)
            workers = 0
            t0 = time.time()
            while workers < n_workers:
                workers = len(self.client.get_worker_logs().keys())
                # If the number of workers is not reached in 5 minutes raise exception
                if time.time() - t0 > 300.0:
                    raise SystemError(
                        "Dask could not start the requested workers within 5 minutes! Try different hostnames.")
            self.WorkerIds = list(self.client.get_worker_logs().keys())
        else:
            raise ValueError("Either hostnames or pbs_params must be provided!")
        # Closing dask processes
        atexit.register(self.client.shutdown)
    
    def getClient(self):
        """
		Accessor for obtaining the client object
		"""
        return self.client
    
    def getWorkerIds(self):
        """
		Accessor for obtaining the worker IDs
		"""
        return self.WorkerIds
    
    def getNworkers(self):
        """
		Accessor for obtaining the number of workers
		"""
        return len(self.getWorkerIds())
