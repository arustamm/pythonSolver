import pyDaskVector
import dask.distributed as daskD
import pyVector as  Vec
import numpy as np

client = daskD.Client("tcp://192.168.66.166:8786")
vec = Vec.vectorIC((100,100))
chunks = (1,1)
vecD = pyDaskVector.VectorDask(client,vector_template=vec,chunks=chunks)

vec1 = Vec.vectorIC((200,1))
vectors = [vec1.clone(),vec1.clone()]
vec1D = pyDaskVector.VectorDask(client,vectors=vectors)

vec1D.rand()
vec1D.norm()
vec1D.checkSame(vecD)
vec1D.checkSame(vec1D)

#Testing getNdArray
vec1 = Vec.vectorIC((200,100,30))
vec2 = Vec.vectorIC((200,100,20))
vec3 = Vec.vectorIC((200,100,10))
vec2D = pyDaskVector.VectorDask(client,vectors=(vec1,vec2,vec3))
arr = vec2D.getNdArray()
