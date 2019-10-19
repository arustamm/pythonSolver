import pyDaskVector
import dask.distributed as daskD
from dask_util import DaskClient
import pyVector as  Vec
import numpy as np


hostnames = ["192.168.66.166","192.168.66.66","192.168.66.67"]
client = DaskClient(hostnames)
client = client.getClient()
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
chunks = (1,1,1)
vecD = pyDaskVector.VectorDask(client,vector_template=vecSep,chunks=chunks)

vec1 = vecSep.clone(); vec1.rand()
vec2 = vecSep.clone(); vec2.rand()
vec3 = vecSep.clone(); vec3.rand()
vectors = [vec1,vec2,vec3]
vecD1 = pyDaskVector.VectorDask(client,vectors=vectors,chunks=chunks)
vec1.norm()**2+vec2.norm()**2+vec3.norm()**2
vecD1.norm()**2
