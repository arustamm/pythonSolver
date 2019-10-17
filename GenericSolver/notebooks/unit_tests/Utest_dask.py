import pyDaskVector
import dask.distributed as daskD
from dask_util import DaskClient
import pyVector as  Vec
import numpy as np


hostnames = ["192.168.66.166","192.168.66.66","192.168.66.67"]
client = DaskClient(hostnames)
client = client.getClient()
# client = daskD.Client("tcp://192.168.66.166:8786")
vec = Vec.vectorIC((100,100))
chunks = (3,2,4)
vecD = pyDaskVector.VectorDask(client,vector_template=vec,chunks=chunks)
client.who_has(vecD.vecDask[0])

vec1 = Vec.vectorIC((200,1))
vectors = [vec1.clone(),vec1.clone()]
vec1D = pyDaskVector.VectorDask(client,vectors=vectors)

vec1D.rand()
vec1D.norm()
vecClone = vec1D.clone()
vec1D.checkSame(vecClone)
vec1D.checkSame(vec1D)

#Testing getNdArray
vec1 = Vec.vectorIC((800,500,1))
vec2 = Vec.vectorIC((800,500,1))
vec3 = Vec.vectorIC((800,500,1))
vec2D = pyDaskVector.VectorDask(client,vectors=(vec1,vec2,vec3))
arr = vec2D.getNdArray()


#Testing with GenericIO
import SepVector
vecSep = SepVector.getSepVector(ns=[200,200])
chunks = (1,)
vecD = pyDaskVector.VectorDask(client,vector_template=vecSep,chunks=chunks)


#Understanding problem with SepVector
import SepVector
from dask_util import DaskClient

hostnames = ["192.168.66.166"]
client = DaskClient(hostnames)
client = client.getClient()
vecSep = SepVector.getSepVector(ns=[100])
future_vec = client.scatter(vecSep)
