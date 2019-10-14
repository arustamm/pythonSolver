import pyDaskVector
import dask.distributed as daskD
import pyVector as  Vec

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
